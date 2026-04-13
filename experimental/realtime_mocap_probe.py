"""
Probe whether FreeMoCap's offline pieces can support a near-real-time side path.

This script intentionally lives under experimental/ and does not touch the
existing GUI/offline processing flow. It replays synchronized videos frame by
frame, runs one tracker per camera, triangulates each small batch, and prints
timing stats.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _choose_runtime_home() -> Path:
    candidates = [
        Path(os.environ["FREEMOCAP_PROBE_HOME"]) if os.environ.get("FREEMOCAP_PROBE_HOME") else None,
        REPO_ROOT / ".venv" / ".freemocap_probe_home",
        Path(tempfile.gettempdir()) / "freemocap_probe_home",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise RuntimeError("Could not find a writable runtime directory for probe-side package config.")


PROBE_RUNTIME_HOME = _choose_runtime_home()
os.environ["HOME"] = str(PROBE_RUNTIME_HOME)
os.environ["USERPROFILE"] = str(PROBE_RUNTIME_HOME)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROBE_RUNTIME_HOME))

_SKELLYTRACKER_BITS = None


def _import_skellytracker_bits():
    global _SKELLYTRACKER_BITS
    if _SKELLYTRACKER_BITS is not None:
        return _SKELLYTRACKER_BITS

    try:
        from skellytracker.process_folder_of_videos import get_tracker
        from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
            MediapipeModelInfo,
            MediapipeTrackingParams,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing skellytracker/mediapipe dependencies. Run this probe in the project environment "
            "(Python >=3.10,<3.13, with freemocap dependencies installed)."
        ) from exc

    _SKELLYTRACKER_BITS = get_tracker, MediapipeModelInfo, MediapipeTrackingParams
    return _SKELLYTRACKER_BITS


def _load_camera_group_class():
    module_path = (
        REPO_ROOT
        / "freemocap"
        / "core_processes"
        / "capture_volume_calibration"
        / "anipose_camera_calibration"
        / "freemocap_anipose.py"
    )
    module_name = "_freemocap_probe_anipose"
    if module_name in sys.modules:
        return sys.modules[module_name].CameraGroup

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load CameraGroup module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing FreeMoCap runtime dependencies. Run this probe in the project environment "
            "(Python >=3.10,<3.13, with freemocap dependencies installed)."
        ) from exc

    return module.CameraGroup


class MediapipePoseOnlyTracker:
    """Small probe-only wrapper around MediaPipe Pose's 33 body landmarks."""

    def __init__(
        self,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        static_image_mode: bool,
    ) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing mediapipe dependency. Run this probe in the project environment "
                "(Python >=3.10,<3.13, with freemocap dependencies installed)."
            ) from exc

        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
        )

    def process_points(self, image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        width, height = image_size
        output = np.full((33, 3), np.nan, dtype=float)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self._pose.process(rgb_image)
        if results.pose_landmarks is None:
            return output

        for landmark_number, landmark_data in enumerate(results.pose_landmarks.landmark):
            output[landmark_number, 0] = landmark_data.x * width
            output[landmark_number, 1] = landmark_data.y * height
            output[landmark_number, 2] = landmark_data.z * width
        return output

    def cleanup(self) -> None:
        self._pose.close()


@dataclass
class Timings:
    read_frames_ms: List[float] = field(default_factory=list)
    resize_frames_ms: List[float] = field(default_factory=list)
    track_2d_ms: List[float] = field(default_factory=list)
    triangulate_3d_ms: List[float] = field(default_factory=list)
    total_batch_ms: List[float] = field(default_factory=list)


@dataclass
class ProbeResult:
    frames_processed: int
    batches_processed: int
    output_shape: Optional[tuple[int, ...]]
    timings: Timings
    valid_2d_point_ratios: List[float]


@dataclass
class MocapFrame:
    seq: int
    timestamp_ns: int
    points_3d: np.ndarray
    valid_2d_point_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay synchronized FreeMoCap videos through a near-real-time 2D->3D timing probe."
    )
    parser.add_argument(
        "--recording-folder",
        required=True,
        type=Path,
        help="Path to a FreeMoCap recording folder containing synchronized_videos/.",
    )
    parser.add_argument(
        "--calibration-toml",
        type=Path,
        default=None,
        help="Path to camera calibration TOML. Defaults to RecordingInfoModel's calibration lookup.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of consecutive frames to triangulate per batch.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe model complexity for the probe.",
    )
    parser.add_argument(
        "--tracker",
        default="holistic",
        choices=["holistic", "pose"],
        help="2D tracker to time. 'pose' runs MediaPipe Pose only; 'holistic' uses FreeMoCap/skellytracker's current path.",
    )
    parser.add_argument(
        "--static-image-mode",
        action="store_true",
        help="Use MediaPipe static image mode. Usually slower, but useful for independent-frame tests.",
    )
    parser.add_argument(
        "--include-holistic",
        action="store_true",
        help="Triangulate all MediaPipe holistic points. Default triangulates body points only to test the first real-time cut.",
    )
    parser.add_argument(
        "--skip-triangulation-prewarm",
        action="store_true",
        help="Skip the one-point triangulation warmup. By default this avoids counting Numba JIT compilation in probe timings.",
    )
    parser.add_argument(
        "--parallel-camera-tracking",
        action="store_true",
        help="Run per-camera 2D tracking concurrently with a thread pool.",
    )
    parser.add_argument(
        "--camera-workers",
        type=int,
        default=None,
        help="Number of thread workers for --parallel-camera-tracking. Defaults to the number of cameras.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Resize each camera frame to this width before 2D tracking, then scale points back to original pixels.",
    )
    return parser.parse_args()


def resolve_recording_info(recording_folder: Path, calibration_toml: Optional[Path]) -> tuple[Path, Path, List[Path]]:
    recording_folder = recording_folder.expanduser().resolve()
    if not recording_folder.exists():
        raise FileNotFoundError(f"Recording folder not found: {recording_folder}")

    if recording_folder.name in {"synchronized_videos", "annotated_videos", "output_data"}:
        recording_folder = recording_folder.parent

    synchronized_videos_folder = recording_folder / "synchronized_videos"
    if not synchronized_videos_folder.exists():
        raise FileNotFoundError(f"synchronized_videos folder not found: {synchronized_videos_folder}")

    video_paths = sorted(
        path
        for path in synchronized_videos_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi"}
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos found in {synchronized_videos_folder}")

    if calibration_toml is None:
        candidates = sorted(recording_folder.glob("*camera_calibration.toml"))
        if not candidates:
            candidates = sorted(recording_folder.glob("*calibration*.toml"))
        if not candidates:
            raise FileNotFoundError("No calibration TOML found. Pass --calibration-toml.")
        calibration_toml = candidates[0]
    else:
        calibration_toml = calibration_toml.expanduser().resolve()

    if not calibration_toml.exists():
        raise FileNotFoundError(f"Calibration TOML not found: {calibration_toml}")

    return recording_folder, calibration_toml, video_paths


def open_video_captures(video_paths: Sequence[Path]) -> tuple[List[cv2.VideoCapture], List[tuple[int, int]], int]:
    captures: List[cv2.VideoCapture] = []
    frame_counts: List[int] = []
    sizes: List[tuple[int, int]] = []

    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            close_captures(captures)
            raise RuntimeError(f"Could not open video: {video_path}")
        captures.append(capture)
        frame_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        sizes.append(
            (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        )

    if len(set(frame_counts)) != 1:
        print(f"Warning: video frame counts differ; using shortest count. Counts: {frame_counts}")

    return captures, sizes, min(frame_counts)


def close_captures(captures: Iterable[cv2.VideoCapture]) -> None:
    for capture in captures:
        capture.release()


def make_trackers(num_cameras: int, model_complexity: int, static_image_mode: bool, tracker_backend: str):
    if tracker_backend == "pose":
        return [
            MediapipePoseOnlyTracker(
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=static_image_mode,
            )
            for _ in range(num_cameras)
        ]

    get_tracker, _mediapipe_model_info, MediapipeTrackingParams = _import_skellytracker_bits()
    tracking_params = MediapipeTrackingParams(
        mediapipe_model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=static_image_mode,
    )
    return [
        get_tracker(tracker_name="MediapipeHolisticTracker", tracking_params=tracking_params)
        for _ in range(num_cameras)
    ]


def tracked_objects_to_array(tracked_objects: Dict[str, object], image_size: tuple[int, int]) -> np.ndarray:
    _get_tracker, MediapipeModelInfo, _MediapipeTrackingParams = _import_skellytracker_bits()
    width, height = image_size
    output = np.full((MediapipeModelInfo.num_tracked_points, 3), np.nan, dtype=float)
    landmark_number = 0

    missing_counts = {
        "pose_landmarks": MediapipeModelInfo.num_tracked_points_body,
        "face_landmarks": MediapipeModelInfo.num_tracked_points_face,
        "left_hand_landmarks": MediapipeModelInfo.num_tracked_points_left_hand,
        "right_hand_landmarks": MediapipeModelInfo.num_tracked_points_right_hand,
    }

    for tracked_object_name in MediapipeModelInfo.tracked_object_names:
        tracked_object = tracked_objects[tracked_object_name]
        landmarks = tracked_object.extra.get("landmarks")
        if landmarks is None:
            landmark_number += missing_counts[tracked_object.object_id]
            continue

        for landmark_data in landmarks.landmark:
            output[landmark_number, 0] = landmark_data.x * width
            output[landmark_number, 1] = landmark_data.y * height
            output[landmark_number, 2] = landmark_data.z * width
            landmark_number += 1

    return output


def read_frame_set(captures: Sequence[cv2.VideoCapture]) -> Optional[List[np.ndarray]]:
    frames: List[np.ndarray] = []
    for capture in captures:
        success, frame = capture.read()
        if not success or frame is None:
            return None
        frames.append(frame)
    return frames


def resize_frames_for_tracking(
    frames: Sequence[np.ndarray], resize_width: Optional[int]
) -> tuple[List[np.ndarray], List[tuple[int, int]], List[tuple[float, float]]]:
    tracking_frames: List[np.ndarray] = []
    tracking_image_sizes: List[tuple[int, int]] = []
    scale_factors: List[tuple[float, float]] = []

    for frame in frames:
        original_height, original_width = frame.shape[:2]
        if resize_width is None or resize_width >= original_width:
            tracking_frames.append(frame)
            tracking_image_sizes.append((original_width, original_height))
            scale_factors.append((1.0, 1.0))
            continue

        resize_height = max(1, round(original_height * (resize_width / original_width)))
        resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        tracking_frames.append(resized_frame)
        tracking_image_sizes.append((resize_width, resize_height))
        scale_factors.append((original_width / resize_width, original_height / resize_height))

    return tracking_frames, tracking_image_sizes, scale_factors


def restore_original_pixel_scale(points: np.ndarray, scale_factor: tuple[float, float]) -> np.ndarray:
    if scale_factor == (1.0, 1.0):
        return points
    restored = points.copy()
    restored[:, 0] *= scale_factor[0]
    restored[:, 1] *= scale_factor[1]
    restored[:, 2] *= scale_factor[0]
    return restored


def track_single_camera_frame(
    tracker, frame: np.ndarray, image_size: tuple[int, int], scale_factor: tuple[float, float]
) -> np.ndarray:
    process_points = getattr(tracker, "process_points", None)
    if callable(process_points):
        tracked_points = process_points(frame, image_size)
    else:
        tracked_objects = tracker.process_image(frame)
        tracked_points = tracked_objects_to_array(tracked_objects, image_size)
    return restore_original_pixel_scale(tracked_points, scale_factor)


def print_setup(recording_folder: Path, calibration_toml: Path, video_paths: Sequence[Path], args: argparse.Namespace):
    print("Near-real-time FreeMoCap probe")
    print(f"  recording_folder: {recording_folder}")
    print(f"  calibration_toml: {calibration_toml}")
    print(f"  cameras: {len(video_paths)}")
    for index, video_path in enumerate(video_paths):
        print(f"    cam[{index}]: {video_path.name}")
    print(f"  max_frames: {args.max_frames}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  tracker: {args.tracker}")
    print(f"  model_complexity: {args.model_complexity}")
    print(f"  static_image_mode: {args.static_image_mode}")
    if args.tracker == "pose":
        print("  points: pose-only")
    else:
        print(f"  points: {'holistic' if args.include_holistic else 'body-only'}")
    print(f"  triangulation_prewarm: {not args.skip_triangulation_prewarm}")
    print(f"  parallel_camera_tracking: {args.parallel_camera_tracking}")
    if args.parallel_camera_tracking:
        print(f"  camera_workers: {args.camera_workers or len(video_paths)}")
    print(f"  resize_width: {args.resize_width}")


def prewarm_triangulation(calibration, num_cameras: int) -> None:
    dummy_points_2d = np.zeros((num_cameras, 1, 2), dtype=float)
    for camera_index in range(num_cameras):
        dummy_points_2d[camera_index, 0, :] = (camera_index + 1, camera_index + 1)
    try:
        calibration.triangulate(dummy_points_2d, progress=False)
    except Exception as exc:
        print(f"Warning: triangulation warmup failed; continuing without warmup. Error: {exc}")


def iter_mocap_3d_frames(
    recording_folder: Path,
    calibration_toml: Optional[Path] = None,
    max_frames: Optional[int] = None,
    model_complexity: int = 1,
    tracker: str = "pose",
    static_image_mode: bool = False,
    parallel_camera_tracking: bool = True,
    camera_workers: Optional[int] = None,
    resize_width: Optional[int] = None,
    include_holistic: bool = False,
    prewarm: bool = True,
) -> Iterator[MocapFrame]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if camera_workers is not None and camera_workers <= 0:
        raise ValueError("camera_workers must be positive")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive")
    if tracker not in {"pose", "holistic"}:
        raise ValueError("tracker must be 'pose' or 'holistic'")

    _recording_folder, calibration_toml, video_paths = resolve_recording_info(recording_folder, calibration_toml)
    CameraGroup = _load_camera_group_class()
    captures, _image_sizes, shortest_video_frame_count = open_video_captures(video_paths)
    calibration = CameraGroup.load(str(calibration_toml))
    if prewarm:
        prewarm_triangulation(calibration, num_cameras=len(video_paths))

    trackers = make_trackers(
        num_cameras=len(video_paths),
        model_complexity=model_complexity,
        static_image_mode=static_image_mode,
        tracker_backend=tracker,
    )
    frame_limit = shortest_video_frame_count if max_frames is None else min(max_frames, shortest_video_frame_count)
    body_point_count = 33
    camera_executor = None
    if parallel_camera_tracking:
        camera_executor = ThreadPoolExecutor(max_workers=camera_workers or len(video_paths))

    try:
        for frame_index in range(frame_limit):
            frames = read_frame_set(captures)
            if frames is None:
                break

            tracking_frames, tracking_image_sizes, scale_factors = resize_frames_for_tracking(frames, resize_width)
            if parallel_camera_tracking:
                per_camera_2d = list(
                    camera_executor.map(
                        track_single_camera_frame,
                        trackers,
                        tracking_frames,
                        tracking_image_sizes,
                        scale_factors,
                    )
                )
            else:
                per_camera_2d = [
                    track_single_camera_frame(tracker_instance, frame, image_size, scale_factor)
                    for tracker_instance, frame, image_size, scale_factor in zip(
                        trackers,
                        tracking_frames,
                        tracking_image_sizes,
                        scale_factors,
                    )
                ]

            tracked_frame = np.stack(per_camera_2d, axis=0)
            if tracker == "holistic" and not include_holistic:
                tracked_frame = tracked_frame[:, :body_point_count, :]

            valid_ratio = float(np.isfinite(tracked_frame[..., :2]).all(axis=-1).mean())
            points_2d = tracked_frame[:, :, :2].reshape(len(video_paths), -1, 2)
            points_3d = calibration.triangulate(points_2d, progress=False).reshape(tracked_frame.shape[1], 3)
            yield MocapFrame(
                seq=frame_index,
                timestamp_ns=time.monotonic_ns(),
                points_3d=points_3d.astype(np.float32, copy=False),
                valid_2d_point_ratio=valid_ratio,
            )
    finally:
        close_captures(captures)
        for tracker_instance in trackers:
            cleanup = getattr(tracker_instance, "cleanup", None)
            if callable(cleanup):
                cleanup()
        if camera_executor is not None:
            camera_executor.shutdown(wait=True)


def run_probe(args: argparse.Namespace) -> ProbeResult:
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.camera_workers is not None and args.camera_workers <= 0:
        raise ValueError("--camera-workers must be positive")
    if args.resize_width is not None and args.resize_width <= 0:
        raise ValueError("--resize-width must be positive")

    recording_folder, calibration_toml, video_paths = resolve_recording_info(args.recording_folder, args.calibration_toml)
    print_setup(recording_folder, calibration_toml, video_paths, args)

    CameraGroup = _load_camera_group_class()
    captures, image_sizes, shortest_video_frame_count = open_video_captures(video_paths)
    calibration = CameraGroup.load(str(calibration_toml))
    if not args.skip_triangulation_prewarm:
        prewarm_start = time.perf_counter()
        prewarm_triangulation(calibration, num_cameras=len(video_paths))
        print(f"  triangulation warmup: {(time.perf_counter() - prewarm_start) * 1000.0:.2f}ms")
    trackers = make_trackers(
        num_cameras=len(video_paths),
        model_complexity=args.model_complexity,
        static_image_mode=args.static_image_mode,
        tracker_backend=args.tracker,
    )
    timings = Timings()
    output_shape = None
    frame_limit = min(args.max_frames, shortest_video_frame_count)
    body_point_count = 33

    frames_processed = 0
    batches_processed = 0
    pending_2d_frames: List[np.ndarray] = []
    valid_2d_point_ratios: List[float] = []
    camera_executor = None
    if args.parallel_camera_tracking:
        camera_executor = ThreadPoolExecutor(max_workers=args.camera_workers or len(video_paths))

    try:
        while frames_processed < frame_limit:
            batch_start = time.perf_counter()

            read_start = time.perf_counter()
            frames = read_frame_set(captures)
            read_end = time.perf_counter()
            if frames is None:
                break

            resize_start = time.perf_counter()
            tracking_frames, tracking_image_sizes, scale_factors = resize_frames_for_tracking(frames, args.resize_width)
            resize_end = time.perf_counter()

            track_start = time.perf_counter()
            if args.parallel_camera_tracking:
                per_camera_2d = list(
                    camera_executor.map(track_single_camera_frame, trackers, tracking_frames, tracking_image_sizes, scale_factors)
                )
            else:
                per_camera_2d = [
                    track_single_camera_frame(tracker, frame, image_size, scale_factor)
                    for tracker, frame, image_size, scale_factor in zip(
                        trackers, tracking_frames, tracking_image_sizes, scale_factors
                    )
                ]
            tracked_frame = np.stack(per_camera_2d, axis=0)
            if args.tracker == "holistic" and not args.include_holistic:
                tracked_frame = tracked_frame[:, :body_point_count, :]
            valid_2d_point_ratios.append(float(np.isfinite(tracked_frame[..., :2]).all(axis=-1).mean()))
            pending_2d_frames.append(tracked_frame)
            track_end = time.perf_counter()

            frames_processed += 1
            should_flush = len(pending_2d_frames) >= args.batch_size or frames_processed >= frame_limit
            if should_flush:
                triangulate_start = time.perf_counter()
                image_2d_data = np.stack(pending_2d_frames, axis=1)
                points_2d = image_2d_data[:, :, :, :2].reshape(len(video_paths), -1, 2)
                points_3d = calibration.triangulate(points_2d, progress=False)
                output_shape = points_3d.reshape(len(pending_2d_frames), image_2d_data.shape[2], 3).shape
                triangulate_end = time.perf_counter()
                pending_2d_frames.clear()
                timings.triangulate_3d_ms.append((triangulate_end - triangulate_start) * 1000.0)
                batches_processed += 1
            else:
                triangulate_end = track_end

            timings.read_frames_ms.append((read_end - read_start) * 1000.0)
            timings.resize_frames_ms.append((resize_end - resize_start) * 1000.0)
            timings.track_2d_ms.append((track_end - track_start) * 1000.0)
            timings.total_batch_ms.append((triangulate_end - batch_start) * 1000.0)

            if frames_processed == 1 or frames_processed % 30 == 0:
                print(
                    f"frame {frames_processed:5d}: "
                    f"read={timings.read_frames_ms[-1]:7.2f}ms "
                    f"resize={timings.resize_frames_ms[-1]:7.2f}ms "
                    f"track={timings.track_2d_ms[-1]:7.2f}ms "
                    f"total={timings.total_batch_ms[-1]:7.2f}ms"
                )
    finally:
        close_captures(captures)
        for tracker in trackers:
            cleanup = getattr(tracker, "cleanup", None)
            if callable(cleanup):
                cleanup()
        if camera_executor is not None:
            camera_executor.shutdown(wait=True)

    return ProbeResult(
        frames_processed=frames_processed,
        batches_processed=batches_processed,
        output_shape=output_shape,
        timings=timings,
        valid_2d_point_ratios=valid_2d_point_ratios,
    )


def p95(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), 95))


def print_metric(name: str, values: Sequence[float], divisor: int = 1) -> None:
    if not values:
        print(f"  {name:<18} no samples")
        return
    per_frame_values = [value / divisor for value in values]
    print(
        f"  {name:<18} "
        f"mean={statistics.mean(per_frame_values):8.2f}ms "
        f"p95={p95(per_frame_values):8.2f}ms "
        f"min={min(per_frame_values):8.2f}ms "
        f"max={max(per_frame_values):8.2f}ms"
    )


def print_summary(result: ProbeResult, batch_size: int) -> None:
    native_resize_total_ms = [
        total_ms - resize_ms
        for total_ms, resize_ms in zip(result.timings.total_batch_ms, result.timings.resize_frames_ms)
    ]

    print("\nSummary")
    print(f"  frames_processed:  {result.frames_processed}")
    print(f"  batches_processed: {result.batches_processed}")
    print(f"  last_3d_shape:     {result.output_shape}")
    if result.valid_2d_point_ratios:
        print(
            f"  valid_2d_points:   {statistics.mean(result.valid_2d_point_ratios) * 100.0:8.2f}% "
            f"mean, {min(result.valid_2d_point_ratios) * 100.0:8.2f}% min"
        )
    if result.timings.total_batch_ms:
        mean_frame_ms = statistics.mean(result.timings.total_batch_ms)
        print(f"  estimated_fps:     {1000.0 / mean_frame_ms:8.2f}")
    if len(result.timings.total_batch_ms) > 1:
        steady_state_mean_ms = statistics.mean(result.timings.total_batch_ms[1:])
        print(f"  steady_state_fps:  {1000.0 / steady_state_mean_ms:8.2f}  (excluding first frame)")
    if native_resize_total_ms:
        mean_native_resize_ms = statistics.mean(native_resize_total_ms)
        print(f"  native_res_fps:    {1000.0 / mean_native_resize_ms:8.2f}  (excluding resize cost)")
    if len(native_resize_total_ms) > 1:
        steady_native_resize_ms = statistics.mean(native_resize_total_ms[1:])
        print(f"  native_res_steady: {1000.0 / steady_native_resize_ms:8.2f}  (excluding first frame + resize)")
    print_metric("read_frames", result.timings.read_frames_ms)
    print_metric("resize_frames", result.timings.resize_frames_ms)
    print_metric("track_2d", result.timings.track_2d_ms)
    print_metric("triangulate_3d", result.timings.triangulate_3d_ms, divisor=batch_size)
    print_metric("total_loop", result.timings.total_batch_ms)
    print_metric("native_res_loop", native_resize_total_ms)
    if len(result.timings.total_batch_ms) > 1:
        print("\nSteady state, excluding first frame")
        print_metric("read_frames", result.timings.read_frames_ms[1:])
        print_metric("resize_frames", result.timings.resize_frames_ms[1:])
        print_metric("track_2d", result.timings.track_2d_ms[1:])
        print_metric("triangulate_3d", result.timings.triangulate_3d_ms[1:], divisor=batch_size)
        print_metric("total_loop", result.timings.total_batch_ms[1:])
        print_metric("native_res_loop", native_resize_total_ms[1:])


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    print_summary(result, args.batch_size)


if __name__ == "__main__":
    main()
