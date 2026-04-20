"""
Experimental skellycam live source for the FreeMoCap -> GMR bridge.

This module intentionally exposes the same MocapFrame stream shape as
realtime_mocap_probe.iter_mocap_3d_frames, so the bridge can swap the input
source without changing the retarget/ZMQ path.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from realtime_mocap_probe import (
    MocapFrame,
    _load_camera_group_class,
    make_trackers,
    prewarm_triangulation,
    resize_frames_for_tracking,
    track_single_camera_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_CAMERA_SKEW_MS = 50.0
SUPPORTED_TRIANGULATE_METHODS = ("simple", "ransac")
SUPPORTED_ROTATION_DEGREES = (0, 90, 180, 270)


def _prepend_experimental_path_for_spawned_processes() -> None:
    experimental_path = str(Path(__file__).resolve().parent)
    if experimental_path not in sys.path:
        sys.path.insert(0, experimental_path)

    pythonpath_parts = [
        part
        for part in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if part
    ]
    if experimental_path not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([experimental_path, *pythonpath_parts])


def install_skellycam_runtime_patches() -> None:
    _prepend_experimental_path_for_spawned_processes()
    try:
        from skellycam_capture_config_patch import install_skellycam_capture_config_patch

        install_skellycam_capture_config_patch()
    except Exception:
        logging = __import__("logging")
        logging.getLogger(__name__).debug(
            "Could not install skellycam runtime patches in this process",
            exc_info=True,
        )


def choose_skellycam_runtime_home(skellycam_home: Optional[Path] = None) -> Path:
    candidates = [
        skellycam_home.expanduser() if skellycam_home is not None else None,
        Path(os.environ["SKELLYCAM_HOME"]).expanduser() if os.environ.get("SKELLYCAM_HOME") else None,
        REPO_ROOT / ".venv" / ".skellycam_live_home",
        Path(tempfile.gettempdir()) / "skellycam_live_home",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate.resolve()
        except OSError:
            continue
    raise RuntimeError("Could not find a writable runtime directory for skellycam.")


def configure_skellycam_runtime_home(skellycam_home: Optional[Path] = None) -> Path:
    runtime_home = choose_skellycam_runtime_home(skellycam_home)
    os.environ["SKELLYCAM_HOME"] = str(runtime_home)
    os.environ["HOME"] = str(runtime_home)
    os.environ["USERPROFILE"] = str(runtime_home)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_home))
    install_skellycam_runtime_patches()
    return runtime_home


def _import_skellycam_bits(skellycam_home: Optional[Path] = None) -> tuple[type, type]:
    configure_skellycam_runtime_home(skellycam_home)
    from skellycam.opencv.camera.models.camera_config import CameraConfig
    from skellycam.opencv.group.camera_group import CameraGroup as SkellyCameraGroup

    return SkellyCameraGroup, CameraConfig


def parse_camera_ids(camera_ids: Optional[str | Sequence[str]]) -> Optional[List[str]]:
    if camera_ids is None:
        return None
    if isinstance(camera_ids, str):
        cleaned = [part.strip() for part in camera_ids.split(",")]
        parsed = [part for part in cleaned if part]
    else:
        parsed = [str(part).strip() for part in camera_ids if str(part).strip()]
    return parsed or None


def load_camera_config_json(camera_config_json: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if camera_config_json is None:
        return {}
    path = camera_config_json.expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("--camera-config-json must contain a JSON object keyed by camera id.")
    return {str(camera_id): dict(config) for camera_id, config in raw.items()}


def load_anipose_calibration(calibration_toml: Path) -> Any:
    calibration_toml = calibration_toml.expanduser().resolve()
    AniposeCameraGroup = _load_camera_group_class()
    return AniposeCameraGroup.load(str(calibration_toml))


def describe_calibration_cameras(calibration: Any) -> Dict[str, Any]:
    cameras = list(getattr(calibration, "cameras", []))
    return {
        "camera_count": len(cameras),
        "camera_names": [getattr(camera, "get_name", lambda: None)() for camera in cameras],
        "camera_sizes": [
            list(getattr(camera, "get_size", lambda: None)() or [])
            for camera in cameras
        ],
    }


def _build_camera_config_dictionary(
    camera_ids: Optional[Sequence[str]],
    camera_config_json: Optional[Path],
    camera_config_class: type,
    expected_camera_count: int,
) -> tuple[List[str], Dict[str, Any]]:
    raw_configs = load_camera_config_json(camera_config_json)
    parsed_camera_ids = parse_camera_ids(camera_ids)

    if raw_configs:
        config_dictionary = {
            str(camera_id): camera_config_class(**{**config, "camera_id": str(camera_id)})
            for camera_id, config in raw_configs.items()
        }
        if parsed_camera_ids is None:
            parsed_camera_ids = [
                camera_id
                for camera_id, config in config_dictionary.items()
                if bool(getattr(config, "use_this_camera", True))
            ]
    else:
        if parsed_camera_ids is None:
            parsed_camera_ids = [str(index) for index in range(expected_camera_count)]
        config_dictionary = {
            camera_id: camera_config_class(camera_id=camera_id)
            for camera_id in parsed_camera_ids
        }

    if len(parsed_camera_ids) != expected_camera_count:
        raise ValueError(
            f"Camera id count ({len(parsed_camera_ids)}) must match calibration camera count "
            f"({expected_camera_count}). Pass --camera-ids in calibration order."
        )

    missing_configs = [camera_id for camera_id in parsed_camera_ids if camera_id not in config_dictionary]
    if missing_configs:
        raise ValueError(f"Camera config JSON is missing camera ids: {missing_configs}")

    ordered_config_dictionary = {
        camera_id: config_dictionary[camera_id].copy(update={"use_this_camera": True})
        if hasattr(config_dictionary[camera_id], "copy")
        else config_dictionary[camera_id]
        for camera_id in parsed_camera_ids
    }
    return parsed_camera_ids, ordered_config_dictionary


def frames_are_synchronized(frame_payloads: Sequence[Any], max_camera_skew_ms: float) -> bool:
    skew_ms = camera_timestamp_skew_ms(frame_payloads)
    return skew_ms is not None and skew_ms <= max_camera_skew_ms


def camera_timestamp_skew_ms(frame_payloads: Sequence[Any]) -> Optional[float]:
    timestamps = [getattr(frame, "timestamp_ns", None) for frame in frame_payloads]
    if any(timestamp is None for timestamp in timestamps):
        return None
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if not np.isfinite(timestamp_array).all():
        return None
    return float((np.max(timestamp_array) - np.min(timestamp_array)) / 1e6)


def _payloads_to_frames_and_sizes(frame_payloads: Sequence[Any]) -> tuple[List[np.ndarray], List[tuple[int, int]]]:
    frames: List[np.ndarray] = []
    image_sizes: List[tuple[int, int]] = []
    for payload in frame_payloads:
        if not bool(getattr(payload, "success", False)):
            raise ValueError("Received unsuccessful camera frame payload.")
        image = getattr(payload, "image", None)
        if image is None:
            raise ValueError("Received camera frame payload with no image.")
        if image.ndim < 2:
            raise ValueError(f"Camera frame image must be at least 2D, got shape {image.shape}.")
        frames.append(image)
        image_sizes.append((int(image.shape[1]), int(image.shape[0])))
    return frames, image_sizes


def _drain_latest_frames(camera_group: Any, camera_ids: Sequence[str], max_drain_rounds: int = 200) -> Dict[str, Any]:
    latest: Dict[str, Any] = {}
    for _ in range(max_drain_rounds):
        got_any = False
        frame_payload_dictionary = camera_group.latest_frames()
        for camera_id in camera_ids:
            frame_payload = frame_payload_dictionary.get(camera_id)
            if frame_payload is not None:
                latest[camera_id] = frame_payload
                got_any = True
        if not got_any:
            break
    return latest


def _normalize_triangulate_method(triangulate_method: str) -> str:
    normalized = str(triangulate_method).strip().lower()
    if normalized not in SUPPORTED_TRIANGULATE_METHODS:
        raise ValueError(
            f"triangulate_method must be one of {SUPPORTED_TRIANGULATE_METHODS}, got {triangulate_method!r}"
        )
    return normalized


def triangulate_points_3d(calibration: Any, points_2d: np.ndarray, triangulate_method: str = "simple") -> np.ndarray:
    triangulate_method = _normalize_triangulate_method(triangulate_method)
    if triangulate_method == "ransac":
        return calibration.triangulate_ransac(points_2d, progress=False)
    return calibration.triangulate(points_2d, progress=False)


def compute_reprojection_diagnostics(calibration: Any, points_3d: np.ndarray, points_2d: np.ndarray) -> Dict[str, Any]:
    reprojection_mean_per_point = np.asarray(
        calibration.reprojection_error(points_3d, points_2d, mean=True),
        dtype=np.float32,
    ).reshape(-1)
    reprojection_vectors = np.asarray(
        calibration.reprojection_error(points_3d, points_2d, mean=False),
        dtype=np.float32,
    )
    if reprojection_vectors.ndim != 3 or reprojection_vectors.shape[-1] != 2:
        raise ValueError(
            f"Expected reprojection_error(mean=False) to return [camera, point, xy], got {reprojection_vectors.shape}"
        )
    reprojection_norms = np.linalg.norm(reprojection_vectors, axis=2)
    per_camera_mean = np.nanmean(reprojection_norms, axis=1).astype(np.float32)
    per_camera_max = np.nanmax(reprojection_norms, axis=1).astype(np.float32)
    return {
        "reprojection_error_px_mean_per_point": reprojection_mean_per_point,
        "reprojection_error_px_mean": float(np.nanmean(reprojection_mean_per_point))
        if np.isfinite(reprojection_mean_per_point).any()
        else None,
        "reprojection_error_px_max": float(np.nanmax(reprojection_norms))
        if np.isfinite(reprojection_norms).any()
        else None,
        "per_camera_reprojection_error_px_mean": per_camera_mean,
        "per_camera_reprojection_error_px_max": per_camera_max,
    }


def rotated_image_size(image_size: tuple[int, int], rotation_degrees: int) -> tuple[int, int]:
    normalized = int(rotation_degrees) % 360
    if normalized not in SUPPORTED_ROTATION_DEGREES:
        raise ValueError(
            f"rotation_degrees must be one of {SUPPORTED_ROTATION_DEGREES}, got {rotation_degrees!r}"
        )
    width, height = image_size
    if normalized in {90, 270}:
        return int(height), int(width)
    return int(width), int(height)


def rotate_points_2d(points_2d: np.ndarray, image_size: tuple[int, int], rotation_degrees: int) -> np.ndarray:
    normalized = int(rotation_degrees) % 360
    if normalized not in SUPPORTED_ROTATION_DEGREES:
        raise ValueError(
            f"rotation_degrees must be one of {SUPPORTED_ROTATION_DEGREES}, got {rotation_degrees!r}"
        )
    rotated = np.asarray(points_2d, dtype=np.float32).copy()
    if normalized == 0 or rotated.size == 0:
        return rotated

    width, height = image_size
    valid = np.isfinite(rotated).all(axis=1)
    if not np.any(valid):
        return rotated

    xs = rotated[valid, 0].copy()
    ys = rotated[valid, 1].copy()
    if normalized == 90:
        rotated[valid, 0] = float(height - 1) - ys
        rotated[valid, 1] = xs
    elif normalized == 180:
        rotated[valid, 0] = float(width - 1) - xs
        rotated[valid, 1] = float(height - 1) - ys
    else:
        rotated[valid, 0] = ys
        rotated[valid, 1] = float(width - 1) - xs
    return rotated


def apply_camera_rotation_variants(
    points_2d: np.ndarray,
    image_sizes: Sequence[tuple[int, int]],
    rotation_degrees_by_camera: Sequence[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    points_2d = np.asarray(points_2d, dtype=np.float32)
    if points_2d.ndim != 3 or points_2d.shape[-1] != 2:
        raise ValueError(f"points_2d must have shape [camera, point, 2], got {points_2d.shape}")
    if len(image_sizes) != points_2d.shape[0]:
        raise ValueError("image_sizes length must match the number of cameras in points_2d")
    if len(rotation_degrees_by_camera) != points_2d.shape[0]:
        raise ValueError("rotation_degrees_by_camera length must match the number of cameras in points_2d")

    transformed = np.empty_like(points_2d, dtype=np.float32)
    transformed_sizes: list[tuple[int, int]] = []
    for camera_index, (image_size, rotation_degrees) in enumerate(zip(image_sizes, rotation_degrees_by_camera)):
        transformed[camera_index] = rotate_points_2d(points_2d[camera_index], image_size, rotation_degrees)
        transformed_sizes.append(rotated_image_size(image_size, rotation_degrees))
    return transformed, transformed_sizes


def score_valid_3d_points(points_3d_mm: np.ndarray) -> float:
    points_3d_mm = np.asarray(points_3d_mm)
    if points_3d_mm.ndim != 2 or points_3d_mm.shape[1] != 3:
        raise ValueError(f"points_3d_mm must have shape [point, 3], got {points_3d_mm.shape}")
    return float(np.isfinite(points_3d_mm).all(axis=1).mean())


def _track_and_triangulate_frame_set(
    frames: Sequence[np.ndarray],
    image_sizes: Sequence[tuple[int, int]],
    trackers: Sequence[Any],
    calibration: Any,
    tracker: str,
    resize_width: Optional[int],
    parallel_camera_tracking: bool,
    camera_executor: Optional[ThreadPoolExecutor],
    include_holistic: bool = False,
    triangulate_method: str = "simple",
) -> tuple[np.ndarray, float, Dict[str, Any]]:
    total_start = time.perf_counter()
    tracking_frames, tracking_image_sizes, scale_factors = resize_frames_for_tracking(frames, resize_width)
    tracking_start = time.perf_counter()
    if parallel_camera_tracking:
        if camera_executor is None:
            raise RuntimeError("camera_executor is required when parallel_camera_tracking=True")
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
    tracking_end = time.perf_counter()

    tracked_frame = np.stack(per_camera_2d, axis=0)
    if tracker == "holistic" and not include_holistic:
        tracked_frame = tracked_frame[:, :33, :]

    valid_points = np.isfinite(tracked_frame[..., :2]).all(axis=-1)
    valid_ratio = float(valid_points.mean())
    per_camera_valid_ratios = valid_points.mean(axis=1).astype(np.float32)
    points_2d = tracked_frame[:, :, :2].reshape(len(image_sizes), -1, 2)
    triangulate_start = time.perf_counter()
    points_3d = triangulate_points_3d(
        calibration=calibration,
        points_2d=points_2d,
        triangulate_method=triangulate_method,
    ).reshape(tracked_frame.shape[1], 3)
    triangulate_end = time.perf_counter()
    reprojection = compute_reprojection_diagnostics(
        calibration=calibration,
        points_3d=points_3d,
        points_2d=points_2d,
    )
    diagnostics = {
        "points_2d": points_2d.astype(np.float32, copy=False),
        "triangulate_method": triangulate_method,
        "per_camera_2d_valid_ratios": per_camera_valid_ratios,
        "tracking_ms": (tracking_end - tracking_start) * 1000.0,
        "triangulate_3d_ms": (triangulate_end - triangulate_start) * 1000.0,
        "total_ms": (triangulate_end - total_start) * 1000.0,
        **reprojection,
    }
    return points_3d.astype(np.float32, copy=False), valid_ratio, diagnostics


def iter_skellycam_mocap_3d_frames(
    calibration_toml: Path,
    camera_ids: Optional[str | Sequence[str]] = None,
    camera_config_json: Optional[Path] = None,
    skellycam_home: Optional[Path] = None,
    max_frames: Optional[int] = None,
    model_complexity: int = 1,
    tracker: str = "pose",
    static_image_mode: bool = False,
    parallel_camera_tracking: bool = True,
    camera_workers: Optional[int] = None,
    resize_width: Optional[int] = None,
    include_holistic: bool = False,
    prewarm: bool = True,
    triangulate_method: str = "simple",
    max_camera_skew_ms: float = DEFAULT_MAX_CAMERA_SKEW_MS,
    poll_sleep_s: float = 0.001,
    skellycam_importer: Callable[[Optional[Path]], tuple[type, type]] = _import_skellycam_bits,
    diagnostics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Iterator[MocapFrame]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if camera_workers is not None and camera_workers <= 0:
        raise ValueError("camera_workers must be positive")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive")
    if max_camera_skew_ms <= 0:
        raise ValueError("max_camera_skew_ms must be positive")
    if tracker not in {"pose", "holistic"}:
        raise ValueError("tracker must be 'pose' or 'holistic'")
    triangulate_method = _normalize_triangulate_method(triangulate_method)

    calibration_toml = calibration_toml.expanduser().resolve()
    if not calibration_toml.exists():
        raise FileNotFoundError(f"Calibration TOML not found: {calibration_toml}")

    calibration = load_anipose_calibration(calibration_toml)
    expected_camera_count = len(calibration.cameras)
    if prewarm:
        prewarm_triangulation(calibration, num_cameras=expected_camera_count)

    SkellyCameraGroup, CameraConfig = skellycam_importer(skellycam_home)
    ordered_camera_ids, camera_config_dictionary = _build_camera_config_dictionary(
        camera_ids=parse_camera_ids(camera_ids),
        camera_config_json=camera_config_json,
        camera_config_class=CameraConfig,
        expected_camera_count=expected_camera_count,
    )

    trackers = make_trackers(
        num_cameras=len(ordered_camera_ids),
        model_complexity=model_complexity,
        static_image_mode=static_image_mode,
        tracker_backend=tracker,
    )
    camera_executor = None
    if parallel_camera_tracking:
        camera_executor = ThreadPoolExecutor(max_workers=camera_workers or len(ordered_camera_ids))

    camera_group = SkellyCameraGroup(
        camera_ids_list=ordered_camera_ids,
        camera_config_dictionary=camera_config_dictionary,
    )

    latest_by_camera: Dict[str, Any] = {}
    seq = 0
    try:
        camera_group.start()
        while max_frames is None or seq < max_frames:
            latest_by_camera.update(_drain_latest_frames(camera_group, ordered_camera_ids))
            if any(camera_id not in latest_by_camera for camera_id in ordered_camera_ids):
                time.sleep(poll_sleep_s)
                continue

            ordered_payloads = [latest_by_camera[camera_id] for camera_id in ordered_camera_ids]
            camera_skew_ms = camera_timestamp_skew_ms(ordered_payloads)
            if camera_skew_ms is None or camera_skew_ms > max_camera_skew_ms:
                time.sleep(poll_sleep_s)
                continue

            frames, image_sizes = _payloads_to_frames_and_sizes(ordered_payloads)
            points_3d, valid_ratio, tracking_diagnostics = _track_and_triangulate_frame_set(
                frames=frames,
                image_sizes=image_sizes,
                trackers=trackers,
                calibration=calibration,
                tracker=tracker,
                resize_width=resize_width,
                parallel_camera_tracking=parallel_camera_tracking,
                camera_executor=camera_executor,
                include_holistic=include_holistic,
                triangulate_method=triangulate_method,
            )
            timestamp_ns = int(max(float(getattr(payload, "timestamp_ns")) for payload in ordered_payloads))
            if diagnostics_callback is not None:
                diagnostics_callback(
                    {
                        "seq": int(seq),
                        "timestamp_ns": timestamp_ns,
                        "camera_ids": list(ordered_camera_ids),
                        "camera_timestamps_ns": [
                            int(float(getattr(payload, "timestamp_ns"))) for payload in ordered_payloads
                        ],
                        "camera_skew_ms": float(camera_skew_ms),
                        "image_sizes": list(image_sizes),
                        "valid_2d_point_ratio": float(valid_ratio),
                        "points_3d": points_3d,
                        **tracking_diagnostics,
                    }
                )
            yield MocapFrame(
                seq=seq,
                timestamp_ns=timestamp_ns,
                points_3d=points_3d,
                valid_2d_point_ratio=valid_ratio,
            )
            seq += 1
    finally:
        try:
            camera_group.close()
        finally:
            for tracker_instance in trackers:
                cleanup = getattr(tracker_instance, "cleanup", None)
                if callable(cleanup):
                    cleanup()
            if camera_executor is not None:
                camera_executor.shutdown(wait=True)
