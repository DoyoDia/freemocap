"""
Diagnose live skellycam -> MediaPipe pose -> 3D triangulation quality.

This script intentionally does not import GMR, MuJoCo, or ZMQ. It only checks
whether the real two-camera mocap stream is stable enough before retargeting.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from skellycam_live_source import DEFAULT_MAX_CAMERA_SKEW_MS, iter_skellycam_mocap_3d_frames

REPO_ROOT = Path(__file__).resolve().parents[1]

MEDIAPIPE = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("nose", "left_ear"),
    ("nose", "right_ear"),
]

TRACKED_POINTS_FOR_JITTER = ("pelvis", "head", "left_ankle", "right_ankle")


def _default_output_folder() -> Path:
    timestamp = datetime.now().strftime("skellycam_mocap_%Y_%m_%d_%H_%M_%S")
    return REPO_ROOT / "freemocap_data" / "diagnostics" / timestamp


def _finite_point(point: np.ndarray) -> bool:
    return bool(np.asarray(point).shape == (3,) and np.isfinite(point).all())


def _point(points_m: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(points_m[MEDIAPIPE[name]], dtype=np.float64)


def _midpoint(points_m: np.ndarray, left_name: str, right_name: str) -> np.ndarray:
    left = _point(points_m, left_name)
    right = _point(points_m, right_name)
    if _finite_point(left) and _finite_point(right):
        return 0.5 * (left + right)
    return np.full(3, np.nan, dtype=np.float64)


def _head_point(points_m: np.ndarray) -> np.ndarray:
    candidates = [
        _point(points_m, name)
        for name in ("nose", "left_ear", "right_ear")
        if _finite_point(_point(points_m, name))
    ]
    if not candidates:
        return np.full(3, np.nan, dtype=np.float64)
    return np.mean(np.stack(candidates, axis=0), axis=0)


def _distance(points_m: np.ndarray, name_a: str, name_b: str) -> Optional[float]:
    point_a = _point(points_m, name_a)
    point_b = _point(points_m, name_b)
    if not (_finite_point(point_a) and _finite_point(point_b)):
        return None
    return float(np.linalg.norm(point_a - point_b))


def _distance_between(point_a: np.ndarray, point_b: np.ndarray) -> Optional[float]:
    if not (_finite_point(point_a) and _finite_point(point_b)):
        return None
    return float(np.linalg.norm(point_a - point_b))


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _named_points(points_m: np.ndarray) -> Dict[str, list[Optional[float]]]:
    pelvis = _midpoint(points_m, "left_hip", "right_hip")
    return {
        "pelvis": _as_jsonable(pelvis),
        "head": _as_jsonable(_head_point(points_m)),
        "left_hip": _as_jsonable(_point(points_m, "left_hip")),
        "right_hip": _as_jsonable(_point(points_m, "right_hip")),
        "left_knee": _as_jsonable(_point(points_m, "left_knee")),
        "right_knee": _as_jsonable(_point(points_m, "right_knee")),
        "left_ankle": _as_jsonable(_point(points_m, "left_ankle")),
        "right_ankle": _as_jsonable(_point(points_m, "right_ankle")),
        "left_foot": _as_jsonable(_point(points_m, "left_foot_index")),
        "right_foot": _as_jsonable(_point(points_m, "right_foot_index")),
    }


def _bone_lengths(points_m: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "left_hip_knee": _distance(points_m, "left_hip", "left_knee"),
        "left_knee_ankle": _distance(points_m, "left_knee", "left_ankle"),
        "left_ankle_foot": _distance(points_m, "left_ankle", "left_foot_index"),
        "right_hip_knee": _distance(points_m, "right_hip", "right_knee"),
        "right_knee_ankle": _distance(points_m, "right_knee", "right_ankle"),
        "right_ankle_foot": _distance(points_m, "right_ankle", "right_foot_index"),
    }


def _widths(points_m: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "shoulder": _distance(points_m, "left_shoulder", "right_shoulder"),
        "hip": _distance(points_m, "left_hip", "right_hip"),
        "knee": _distance(points_m, "left_knee", "right_knee"),
        "ankle": _distance(points_m, "left_ankle", "right_ankle"),
    }


def _jitter_metrics(
    points_m: np.ndarray,
    previous_points_m: Optional[np.ndarray],
    dt_s: Optional[float],
) -> Dict[str, Optional[float]]:
    if previous_points_m is None or dt_s is None or dt_s <= 0.0:
        return {name: None for name in TRACKED_POINTS_FOR_JITTER}

    current = {
        "pelvis": _midpoint(points_m, "left_hip", "right_hip"),
        "head": _head_point(points_m),
        "left_ankle": _point(points_m, "left_ankle"),
        "right_ankle": _point(points_m, "right_ankle"),
    }
    previous = {
        "pelvis": _midpoint(previous_points_m, "left_hip", "right_hip"),
        "head": _head_point(previous_points_m),
        "left_ankle": _point(previous_points_m, "left_ankle"),
        "right_ankle": _point(previous_points_m, "right_ankle"),
    }
    return {
        name: None
        if _distance_between(current[name], previous[name]) is None
        else float(_distance_between(current[name], previous[name]) / dt_s)
        for name in TRACKED_POINTS_FOR_JITTER
    }


def _largest_bone_change(
    bone_lengths: Dict[str, Optional[float]],
    previous_bone_lengths: Optional[Dict[str, Optional[float]]],
) -> Optional[float]:
    if previous_bone_lengths is None:
        return None
    changes = []
    for key, value in bone_lengths.items():
        previous_value = previous_bone_lengths.get(key)
        if value is not None and previous_value is not None:
            changes.append(abs(float(value) - float(previous_value)))
    return max(changes) if changes else None


def _compute_frame_metrics(
    *,
    seq: int,
    timestamp_ns: int,
    points_m: np.ndarray,
    source_diagnostics: Dict[str, Any],
    previous_points_m: Optional[np.ndarray],
    previous_timestamp_ns: Optional[int],
    previous_bone_lengths: Optional[Dict[str, Optional[float]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dt_s = None
    if previous_timestamp_ns is not None:
        dt_s = max(0.0, (timestamp_ns - previous_timestamp_ns) / 1e9)
    bone_lengths = _bone_lengths(points_m)
    largest_bone_change = _largest_bone_change(bone_lengths, previous_bone_lengths)
    finite_3d = np.isfinite(points_m).all(axis=1)
    max_abs_point_m = float(np.nanmax(np.abs(points_m))) if np.isfinite(points_m).any() else None
    jitter_m_per_s = _jitter_metrics(points_m, previous_points_m, dt_s)

    flags = []
    valid_2d_ratio = float(source_diagnostics.get("valid_2d_point_ratio", np.nan))
    camera_skew_ms = source_diagnostics.get("camera_skew_ms")
    if np.isfinite(valid_2d_ratio) and valid_2d_ratio < args.low_valid_2d_ratio:
        flags.append("low_valid_2d")
    if camera_skew_ms is not None and float(camera_skew_ms) > args.max_camera_skew_ms:
        flags.append("high_camera_skew")
    if max_abs_point_m is not None and max_abs_point_m > args.max_abs_point_m:
        flags.append("point_out_of_range")
    if largest_bone_change is not None and largest_bone_change > args.max_bone_length_change_m:
        flags.append("bone_length_jump")
    for name, speed in jitter_m_per_s.items():
        if speed is not None and speed > args.max_jitter_speed_mps:
            flags.append(f"{name}_jitter")

    return {
        "seq": int(seq),
        "timestamp_ns": int(timestamp_ns),
        "valid_2d_point_ratio": valid_2d_ratio,
        "per_camera_2d_valid_ratios": _as_jsonable(source_diagnostics.get("per_camera_2d_valid_ratios", [])),
        "camera_ids": _as_jsonable(source_diagnostics.get("camera_ids", [])),
        "camera_timestamps_ns": _as_jsonable(source_diagnostics.get("camera_timestamps_ns", [])),
        "camera_skew_ms": None if camera_skew_ms is None else float(camera_skew_ms),
        "tracking_ms": float(source_diagnostics.get("tracking_ms", np.nan)),
        "triangulate_3d_ms": float(source_diagnostics.get("triangulate_3d_ms", np.nan)),
        "total_tracking_and_triangulation_ms": float(source_diagnostics.get("total_ms", np.nan)),
        "nan_3d_point_count": int((~finite_3d).sum()),
        "max_abs_point_m": max_abs_point_m,
        "named_points_m": _named_points(points_m),
        "widths_m": _widths(points_m),
        "bone_lengths_m": bone_lengths,
        "largest_bone_length_change_m": largest_bone_change,
        "jitter_m_per_s": jitter_m_per_s,
        "flags": flags,
    }


class SkeletonViewer:
    def __init__(self, viewer_range_m: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.viewer_range_m = float(viewer_range_m)
        self.center: Optional[np.ndarray] = None
        self.plt.ion()
        self.fig = self.plt.figure("Skellycam 3D mocap diagnostics")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.scatter = self.ax.scatter([], [], [], c=[], s=24)
        self.lines = [self.ax.plot([], [], [], linewidth=2)[0] for _ in SKELETON_EDGES]
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_zlabel("z (m)")

    def is_open(self) -> bool:
        return bool(self.plt.fignum_exists(self.fig.number))

    def _lock_axes(self, points_m: np.ndarray) -> None:
        if self.center is not None:
            return
        pelvis = _midpoint(points_m, "left_hip", "right_hip")
        if not _finite_point(pelvis):
            finite_points = points_m[np.isfinite(points_m).all(axis=1)]
            if finite_points.size == 0:
                pelvis = np.zeros(3, dtype=np.float64)
            else:
                pelvis = np.nanmean(finite_points, axis=0)
        self.center = pelvis.astype(np.float64)
        half_range = self.viewer_range_m / 2.0
        self.ax.set_xlim(self.center[0] - half_range, self.center[0] + half_range)
        self.ax.set_ylim(self.center[1] - half_range, self.center[1] + half_range)
        self.ax.set_zlim(self.center[2] - half_range, self.center[2] + half_range)

    def update(self, points_m: np.ndarray, title: str) -> None:
        if not self.is_open():
            return
        self._lock_axes(points_m)
        colors = []
        for index in range(points_m.shape[0]):
            if index in {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}:
                colors.append("tab:blue")
            elif index in {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}:
                colors.append("tab:red")
            else:
                colors.append("tab:gray")
        self.scatter._offsets3d = (points_m[:, 0], points_m[:, 1], points_m[:, 2])
        self.scatter.set_color(colors)
        for line, (start_name, end_name) in zip(self.lines, SKELETON_EDGES):
            start = _point(points_m, start_name)
            end = _point(points_m, end_name)
            if not (_finite_point(start) and _finite_point(end)):
                line.set_data_3d([], [], [])
                continue
            line.set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
            line.set_color("tab:blue" if start_name.startswith("left") else "tab:red")
        self.ax.set_title(title)
        self.plt.pause(0.001)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-toml", type=Path, required=True)
    parser.add_argument("--camera-ids", type=str, default=None)
    parser.add_argument("--camera-config-json", type=Path, default=None)
    parser.add_argument("--skellycam-home", type=Path, default=None)
    parser.add_argument("--tracker", choices=["pose", "holistic"], default="pose")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--parallel-camera-tracking", action="store_true")
    parser.add_argument("--camera-workers", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--max-camera-skew-ms", type=float, default=DEFAULT_MAX_CAMERA_SKEW_MS)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--viewer", dest="viewer", action="store_true", default=True)
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--viewer-range-m", type=float, default=3.0)
    parser.add_argument("--print-every-frames", type=int, default=30)
    parser.add_argument("--low-valid-2d-ratio", type=float, default=0.75)
    parser.add_argument("--max-bone-length-change-m", type=float, default=0.20)
    parser.add_argument("--max-jitter-speed-mps", type=float, default=5.0)
    parser.add_argument("--max-abs-point-m", type=float, default=10.0)
    parser.add_argument("--skip-triangulation-prewarm", action="store_true")
    return parser


def _summarize(metrics: list[Dict[str, Any]], output_folder: Path, args: argparse.Namespace) -> Dict[str, Any]:
    flag_counts = Counter(flag for metric in metrics for flag in metric.get("flags", []))
    valid_ratios = [metric["valid_2d_point_ratio"] for metric in metrics if metric["valid_2d_point_ratio"] is not None]
    camera_skews = [metric["camera_skew_ms"] for metric in metrics if metric["camera_skew_ms"] is not None]
    total_times = [
        metric["total_tracking_and_triangulation_ms"]
        for metric in metrics
        if math.isfinite(metric["total_tracking_and_triangulation_ms"])
    ]
    suspicious_frames = [
        {"seq": metric["seq"], "flags": metric["flags"]}
        for metric in metrics
        if metric.get("flags")
    ]
    duration_s = None
    if len(metrics) >= 2:
        duration_s = max(0.0, (metrics[-1]["timestamp_ns"] - metrics[0]["timestamp_ns"]) / 1e9)
    return {
        "frame_count": len(metrics),
        "duration_s": duration_s,
        "mean_mocap_fps": None
        if not duration_s
        else float((len(metrics) - 1) / duration_s),
        "valid_2d_point_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
        "valid_2d_point_ratio_min": float(np.min(valid_ratios)) if valid_ratios else None,
        "camera_skew_ms_mean": float(np.mean(camera_skews)) if camera_skews else None,
        "camera_skew_ms_max": float(np.max(camera_skews)) if camera_skews else None,
        "tracking_and_triangulation_ms_mean": float(np.mean(total_times)) if total_times else None,
        "tracking_and_triangulation_ms_max": float(np.max(total_times)) if total_times else None,
        "flag_counts": dict(flag_counts),
        "suspicious_frames": suspicious_frames[:200],
        "output_folder": str(output_folder),
        "points_3d_npy": str(output_folder / "mocap_3d_body_points.npy"),
        "diagnostics_jsonl": str(output_folder / "diagnostics.jsonl"),
        "args": vars(args),
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_folder = (args.output_folder or _default_output_folder()).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    latest_source_diagnostics: Dict[str, Any] = {}

    def diagnostics_callback(diagnostics: Dict[str, Any]) -> None:
        latest_source_diagnostics.clear()
        latest_source_diagnostics.update(diagnostics)

    viewer = SkeletonViewer(args.viewer_range_m) if args.viewer else None
    points_history: list[np.ndarray] = []
    metrics_history: list[Dict[str, Any]] = []
    previous_points_m: Optional[np.ndarray] = None
    previous_timestamp_ns: Optional[int] = None
    previous_bone_lengths: Optional[Dict[str, Optional[float]]] = None
    wall_start = time.perf_counter()

    diagnostics_jsonl_path = output_folder / "diagnostics.jsonl"
    print(f"Skellycam mocap diagnostics starting. Output: {output_folder}", flush=True)
    try:
        with diagnostics_jsonl_path.open("w", encoding="utf-8") as diagnostics_file:
            for frame in iter_skellycam_mocap_3d_frames(
                calibration_toml=args.calibration_toml,
                camera_ids=args.camera_ids,
                camera_config_json=args.camera_config_json,
                skellycam_home=args.skellycam_home,
                max_frames=args.max_frames,
                model_complexity=args.model_complexity,
                tracker=args.tracker,
                parallel_camera_tracking=args.parallel_camera_tracking,
                camera_workers=args.camera_workers,
                resize_width=args.resize_width,
                include_holistic=False,
                prewarm=not args.skip_triangulation_prewarm,
                max_camera_skew_ms=args.max_camera_skew_ms,
                diagnostics_callback=diagnostics_callback,
            ):
                points_m = np.asarray(frame.points_3d, dtype=np.float32).reshape(33, 3) * 0.001
                source_diagnostics = dict(latest_source_diagnostics)
                metrics = _compute_frame_metrics(
                    seq=int(frame.seq),
                    timestamp_ns=int(frame.timestamp_ns),
                    points_m=points_m,
                    source_diagnostics=source_diagnostics,
                    previous_points_m=previous_points_m,
                    previous_timestamp_ns=previous_timestamp_ns,
                    previous_bone_lengths=previous_bone_lengths,
                    args=args,
                )
                elapsed_s = max(time.perf_counter() - wall_start, 1e-6)
                metrics["mocap_fps"] = float((len(metrics_history) + 1) / elapsed_s)
                points_history.append(points_m.astype(np.float32, copy=True))
                metrics_history.append(metrics)
                diagnostics_file.write(json.dumps(_as_jsonable(metrics), ensure_ascii=False) + "\n")
                diagnostics_file.flush()

                previous_points_m = points_m
                previous_timestamp_ns = int(frame.timestamp_ns)
                previous_bone_lengths = metrics["bone_lengths_m"]

                if viewer is not None and viewer.is_open():
                    viewer.update(
                        points_m,
                        title=(
                            f"seq={frame.seq} valid2d={frame.valid_2d_point_ratio * 100.0:.1f}% "
                            f"flags={','.join(metrics['flags']) or 'ok'}"
                        ),
                    )

                if args.print_every_frames > 0 and (frame.seq == 0 or frame.seq % args.print_every_frames == 0):
                    print(
                        "[MocapDiagStats] "
                        f"seq={frame.seq} fps={metrics['mocap_fps']:.1f} "
                        f"valid2d={frame.valid_2d_point_ratio * 100.0:.1f}% "
                        f"skew_ms={metrics['camera_skew_ms']} "
                        f"nan3d={metrics['nan_3d_point_count']} "
                        f"flags={','.join(metrics['flags']) or 'ok'}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("Diagnostics interrupted; saving collected frames.", flush=True)
    finally:
        points_path = output_folder / "mocap_3d_body_points.npy"
        if points_history:
            np.save(points_path, np.stack(points_history, axis=0).astype(np.float32))
        else:
            np.save(points_path, np.empty((0, 33, 3), dtype=np.float32))
        summary = _summarize(metrics_history, output_folder, args)
        summary_path = output_folder / "summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(_as_jsonable(summary), file, ensure_ascii=False, indent=2)
        print(f"Saved 3D points: {points_path}", flush=True)
        print(f"Saved diagnostics: {diagnostics_jsonl_path}", flush=True)
        print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
