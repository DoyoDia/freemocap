"""
Diagnose live skellycam -> MediaPipe pose -> 3D triangulation quality.

This script intentionally does not import GMR, MuJoCo, or ZMQ. It only checks
whether the real two-camera mocap stream is stable enough before retargeting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import queue
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from skellycam_live_source import (
    DEFAULT_MAX_CAMERA_SKEW_MS,
    SUPPORTED_ROTATION_DEGREES,
    apply_camera_rotation_variants,
    compute_reprojection_diagnostics,
    describe_calibration_cameras,
    iter_skellycam_mocap_3d_frames,
    load_anipose_calibration,
    load_camera_config_json,
    parse_camera_ids,
    score_valid_3d_points,
    triangulate_points_3d,
)

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
PROBLEM_PRIORITY = (
    "point_out_of_range",
    "bad_reprojection",
    "implausible_height",
    "implausible_segment",
    "high_camera_skew",
    "low_valid_2d",
    "bone_length_jump",
    "pelvis_jitter",
    "head_jitter",
    "left_ankle_jitter",
    "right_ankle_jitter",
)
HUMAN_PLAUSIBILITY_RANGES = {
    "estimated_height_m": (1.0, 2.3),
    "shoulder_width_m": (0.25, 0.8),
    "hip_width_m": (0.15, 0.7),
    "left_thigh_length_m": (0.2, 0.8),
    "right_thigh_length_m": (0.2, 0.8),
    "left_shin_length_m": (0.2, 0.8),
    "right_shin_length_m": (0.2, 0.8),
    "left_foot_length_m": (0.05, 0.45),
    "right_foot_length_m": (0.05, 0.45),
}
DEFAULT_BAD_REPROJECTION_ERROR_PX = 15.0
DEFAULT_VIEWER_RANGE_M = 4.0
DEFAULT_ORDER_TEST_FRAMES = 120
DEFAULT_ROTATION_TEST_FRAMES = 60


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
    return _distance_between(_point(points_m, name_a), _point(points_m, name_b))


def _distance_between(point_a: np.ndarray, point_b: np.ndarray) -> Optional[float]:
    if not (_finite_point(point_a) and _finite_point(point_b)):
        return None
    return float(np.linalg.norm(point_a - point_b))


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return None
    return float(np.mean(valid))


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
        "left_ankle_heel": _distance(points_m, "left_ankle", "left_heel"),
        "left_heel_foot": _distance(points_m, "left_heel", "left_foot_index"),
        "right_hip_knee": _distance(points_m, "right_hip", "right_knee"),
        "right_knee_ankle": _distance(points_m, "right_knee", "right_ankle"),
        "right_ankle_heel": _distance(points_m, "right_ankle", "right_heel"),
        "right_heel_foot": _distance(points_m, "right_heel", "right_foot_index"),
    }


def _segment_metrics(points_m: np.ndarray) -> Dict[str, Optional[float]]:
    left_thigh = _distance(points_m, "left_hip", "left_knee")
    right_thigh = _distance(points_m, "right_hip", "right_knee")
    left_shin = _distance(points_m, "left_knee", "left_ankle")
    right_shin = _distance(points_m, "right_knee", "right_ankle")
    left_foot = _distance(points_m, "left_heel", "left_foot_index")
    right_foot = _distance(points_m, "right_heel", "right_foot_index")
    if left_foot is None:
        left_foot = _distance(points_m, "left_ankle", "left_foot_index")
    if right_foot is None:
        right_foot = _distance(points_m, "right_ankle", "right_foot_index")
    shoulder_width = _distance(points_m, "left_shoulder", "right_shoulder")
    hip_width = _distance(points_m, "left_hip", "right_hip")
    head_to_pelvis = _distance_between(_head_point(points_m), _midpoint(points_m, "left_hip", "right_hip"))
    left_leg = _mean_optional(
        [
            _distance(points_m, "left_hip", "left_knee"),
            _distance(points_m, "left_knee", "left_ankle"),
            _distance(points_m, "left_ankle", "left_heel"),
        ]
    )
    right_leg = _mean_optional(
        [
            _distance(points_m, "right_hip", "right_knee"),
            _distance(points_m, "right_knee", "right_ankle"),
            _distance(points_m, "right_ankle", "right_heel"),
        ]
    )
    estimated_height = None
    if head_to_pelvis is not None:
        leg_proxy = _mean_optional(
            [
                None if left_thigh is None or left_shin is None else left_thigh + left_shin + (_distance(points_m, "left_ankle", "left_heel") or 0.0),
                None if right_thigh is None or right_shin is None else right_thigh + right_shin + (_distance(points_m, "right_ankle", "right_heel") or 0.0),
            ]
        )
        if leg_proxy is not None:
            estimated_height = float(head_to_pelvis + leg_proxy)

    max_abs_point_m = float(np.nanmax(np.abs(points_m))) if np.isfinite(points_m).any() else None
    return {
        "estimated_height_m": estimated_height,
        "shoulder_width_m": shoulder_width,
        "hip_width_m": hip_width,
        "left_thigh_length_m": left_thigh,
        "right_thigh_length_m": right_thigh,
        "left_shin_length_m": left_shin,
        "right_shin_length_m": right_shin,
        "left_foot_length_m": left_foot,
        "right_foot_length_m": right_foot,
        "max_abs_point_m": max_abs_point_m,
        "left_leg_proxy_m": left_leg,
        "right_leg_proxy_m": right_leg,
    }


def _metric_is_plausible(metric_name: str, value: Optional[float]) -> Optional[bool]:
    if value is None or not math.isfinite(float(value)):
        return None
    lower, upper = HUMAN_PLAUSIBILITY_RANGES[metric_name]
    return bool(lower <= float(value) <= upper)


def _plausibility_metrics(points_m: np.ndarray) -> Dict[str, Any]:
    segment_metrics = _segment_metrics(points_m)
    plausibility = {
        metric_name: _metric_is_plausible(metric_name, segment_metrics.get(metric_name))
        for metric_name in HUMAN_PLAUSIBILITY_RANGES
    }
    available = [value for value in plausibility.values() if value is not None]
    plausible_count = sum(1 for value in available if value)
    segment_values = [
        value
        for name, value in plausibility.items()
        if name != "estimated_height_m" and value is not None
    ]
    return {
        "segment_metrics": segment_metrics,
        "plausibility": plausibility,
        "plausible_metric_ratio": None if not available else float(plausible_count / len(available)),
        "height_plausible": plausibility["estimated_height_m"],
        "segment_plausible": None if not segment_values else bool(all(segment_values)),
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


def _pick_last_problem(flags: Sequence[str]) -> str:
    if not flags:
        return "ok"
    for preferred in PROBLEM_PRIORITY:
        if preferred in flags:
            return preferred
    return flags[0]


def replace_latest_queue_item(latest_queue: Any, item: Any) -> None:
    while True:
        try:
            latest_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                latest_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)


def _camera_order_indices(camera_count: int, order_name: str) -> list[int]:
    parts = [int(part) for part in order_name.split(",") if part.strip()]
    if sorted(parts) != list(range(camera_count)):
        raise ValueError(f"Invalid camera order {order_name!r} for {camera_count} cameras")
    return parts


def _camera_order_label(camera_ids: Sequence[str]) -> str:
    return ",".join(str(camera_id) for camera_id in camera_ids)


def _rotation_variant_label(rotation_degrees_by_camera: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in rotation_degrees_by_camera)


def classify_likely_cause(summary: Dict[str, Any], max_camera_skew_ms: float) -> str:
    valid_ratio = summary.get("valid_2d_point_ratio_mean")
    reprojection_mean = summary.get("reprojection_error_px_mean_mean")
    plausible_ratio = summary.get("plausible_metric_ratio_mean")
    camera_skew_mean = summary.get("camera_skew_ms_mean")
    camera_skew_max = summary.get("camera_skew_ms_max")
    if valid_ratio is not None and float(valid_ratio) < 0.75:
        return "mediapipe_or_visibility_problem"
    if camera_skew_max is not None and float(camera_skew_max) > max_camera_skew_ms:
        return "camera_sync_problem"
    if (
        reprojection_mean is not None
        and float(reprojection_mean) <= DEFAULT_BAD_REPROJECTION_ERROR_PX
        and plausible_ratio is not None
        and float(plausible_ratio) < 0.5
    ):
        return "unit_or_toml_scale_problem"
    if (
        valid_ratio is not None
        and float(valid_ratio) >= 0.9
        and reprojection_mean is not None
        and float(reprojection_mean) > DEFAULT_BAD_REPROJECTION_ERROR_PX
    ):
        return "calibration_or_camera_order_or_rotation_mismatch"
    if camera_skew_mean is not None and float(camera_skew_mean) > max_camera_skew_ms * 0.75:
        return "camera_sync_problem"
    return "unknown"


def _camera_config_summary(
    camera_ids: Sequence[str],
    camera_configs: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    output = []
    for camera_id in camera_ids:
        config = dict(camera_configs.get(str(camera_id), {}))
        output.append(
            {
                "camera_id": str(camera_id),
                "resolution_width": config.get("resolution_width"),
                "resolution_height": config.get("resolution_height"),
                "framerate": config.get("framerate"),
                "fourcc": config.get("fourcc"),
                "exposure": config.get("exposure"),
                "rotate_video_cv2_code": config.get("rotate_video_cv2_code"),
                "use_this_camera": config.get("use_this_camera"),
            }
        )
    return output


def _build_startup_consistency_report(
    calibration: Any,
    camera_ids: Sequence[str],
    camera_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    calibration_summary = describe_calibration_cameras(calibration)
    live_summary = _camera_config_summary(camera_ids, camera_configs)
    warnings: list[str] = []
    if len(camera_ids) != calibration_summary["camera_count"]:
        warnings.append("live camera count differs from calibration TOML camera count")

    calibration_names = [name for name in calibration_summary["camera_names"] if name not in {None, "None"}]
    if calibration_names:
        if set(map(str, calibration_names)) == set(map(str, camera_ids)) and list(map(str, calibration_names)) != list(map(str, camera_ids)):
            warnings.append("camera id order is ambiguous: live ids match TOML names but in a different order")
    elif len(camera_ids) == 2:
        warnings.append("camera id order is ambiguous: calibration names are unavailable, test both orders")

    for index, (camera_info, calibration_size) in enumerate(zip(live_summary, calibration_summary["camera_sizes"])):
        width = camera_info.get("resolution_width")
        height = camera_info.get("resolution_height")
        if calibration_size and width is not None and height is not None:
            if [int(width), int(height)] != [int(calibration_size[0]), int(calibration_size[1])]:
                warnings.append(
                    f"camera {index} live resolution {width}x{height} differs from calibration {calibration_size[0]}x{calibration_size[1]}"
                )
        if camera_info.get("rotate_video_cv2_code") not in {None, 0}:
            warnings.append(
                f"camera {camera_info['camera_id']} has non-default rotation {camera_info.get('rotate_video_cv2_code')}"
            )

    return {
        "calibration": calibration_summary,
        "live_camera_config": live_summary,
        "warnings": warnings,
    }


def _print_startup_report(report: Dict[str, Any]) -> None:
    print(
        f"[MocapDiagSetup] toml_cameras={report['calibration']['camera_count']} "
        f"toml_names={report['calibration']['camera_names']}",
        flush=True,
    )
    for camera_info in report["live_camera_config"]:
        print(
            "[MocapDiagSetup] "
            f"camera_id={camera_info['camera_id']} "
            f"resolution={camera_info.get('resolution_width')}x{camera_info.get('resolution_height')} "
            f"fps={camera_info.get('framerate')} fourcc={camera_info.get('fourcc')} "
            f"exposure={camera_info.get('exposure')} rotation={camera_info.get('rotate_video_cv2_code')}",
            flush=True,
        )
    for warning in report["warnings"]:
        print(f"[MocapDiagWarning] {warning}", flush=True)


def _source_generator_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "calibration_toml": args.calibration_toml,
        "camera_ids": args.camera_ids,
        "camera_config_json": args.camera_config_json,
        "skellycam_home": args.skellycam_home,
        "model_complexity": args.model_complexity,
        "tracker": args.tracker,
        "parallel_camera_tracking": args.parallel_camera_tracking,
        "camera_workers": args.camera_workers,
        "resize_width": args.resize_width,
        "include_holistic": False,
        "prewarm": not args.skip_triangulation_prewarm,
        "triangulate_method": args.triangulate_method,
        "max_camera_skew_ms": args.max_camera_skew_ms,
    }


def _collect_preflight_samples(
    args: argparse.Namespace,
    frame_limit: int,
) -> list[Dict[str, Any]]:
    latest_source_diagnostics: Dict[str, Any] = {}
    collected: list[Dict[str, Any]] = []
    valid_streak = 0

    def diagnostics_callback(diagnostics: Dict[str, Any]) -> None:
        latest_source_diagnostics.clear()
        latest_source_diagnostics.update(diagnostics)

    print(f"[MocapDiagPreflight] collecting {frame_limit} real frames for candidate scoring", flush=True)
    for frame in iter_skellycam_mocap_3d_frames(
        max_frames=None,
        diagnostics_callback=diagnostics_callback,
        **_source_generator_kwargs(args),
    ):
        diagnostics = dict(latest_source_diagnostics)
        valid_2d_ratio = float(diagnostics.get("valid_2d_point_ratio", np.nan))
        if valid_2d_ratio >= float(args.start_valid_2d_ratio):
            valid_streak += 1
        else:
            valid_streak = 0
        if valid_streak < int(args.start_stable_frames):
            if frame.seq == 0 or frame.seq % max(1, args.print_every_frames) == 0:
                print(
                    "[MocapDiagPreflightWaiting] "
                    f"seq={frame.seq} valid2d={valid_2d_ratio * 100.0:.1f}% "
                    f"streak={valid_streak}/{args.start_stable_frames}",
                    flush=True,
                )
            continue

        sample = {
            "seq": int(frame.seq),
            "timestamp_ns": int(frame.timestamp_ns),
            "camera_ids": list(diagnostics.get("camera_ids", [])),
            "camera_skew_ms": diagnostics.get("camera_skew_ms"),
            "valid_2d_point_ratio": valid_2d_ratio,
            "per_camera_2d_valid_ratios": np.asarray(
                diagnostics.get("per_camera_2d_valid_ratios", []),
                dtype=np.float32,
            ),
            "image_sizes": [
                tuple(int(value) for value in image_size)
                for image_size in diagnostics.get("image_sizes", [])
            ],
            "points_2d": np.asarray(diagnostics.get("points_2d"), dtype=np.float32).copy(),
        }
        if sample["points_2d"].size == 0:
            continue
        collected.append(sample)
        if len(collected) == 1 or len(collected) % max(1, args.print_every_frames) == 0:
            print(
                "[MocapDiagPreflight] "
                f"sample={len(collected)}/{frame_limit} seq={frame.seq} "
                f"valid2d={valid_2d_ratio * 100.0:.1f}%",
                flush=True,
            )
        if len(collected) >= frame_limit:
            break
    return collected


def _evaluate_candidate_on_samples(
    *,
    samples: Sequence[Dict[str, Any]],
    calibration: Any,
    camera_order_indices: Sequence[int],
    rotation_degrees_by_camera: Sequence[int],
    triangulate_method: str,
    label: str,
) -> Dict[str, Any]:
    reprojection_means: list[float] = []
    reprojection_maxes: list[float] = []
    valid_3d_ratios: list[float] = []
    plausible_metric_ratios: list[float] = []
    height_plausible_count = 0
    segment_plausible_count = 0

    for sample in samples:
        ordered_points_2d = np.asarray(sample["points_2d"], dtype=np.float32)[list(camera_order_indices)]
        ordered_image_sizes = [sample["image_sizes"][index] for index in camera_order_indices]
        transformed_points_2d, _transformed_sizes = apply_camera_rotation_variants(
            points_2d=ordered_points_2d,
            image_sizes=ordered_image_sizes,
            rotation_degrees_by_camera=rotation_degrees_by_camera,
        )
        points_3d_mm = triangulate_points_3d(
            calibration=calibration,
            points_2d=transformed_points_2d,
            triangulate_method=triangulate_method,
        )
        reprojection = compute_reprojection_diagnostics(
            calibration=calibration,
            points_3d=points_3d_mm,
            points_2d=transformed_points_2d,
        )
        points_m = np.asarray(points_3d_mm, dtype=np.float32).reshape(-1, 3) * 0.001
        plausibility = _plausibility_metrics(points_m)

        valid_3d_ratios.append(score_valid_3d_points(points_3d_mm))
        if reprojection["reprojection_error_px_mean"] is not None:
            reprojection_means.append(float(reprojection["reprojection_error_px_mean"]))
        if reprojection["reprojection_error_px_max"] is not None:
            reprojection_maxes.append(float(reprojection["reprojection_error_px_max"]))
        if plausibility["plausible_metric_ratio"] is not None:
            plausible_metric_ratios.append(float(plausibility["plausible_metric_ratio"]))
        if plausibility["height_plausible"] is True:
            height_plausible_count += 1
        if plausibility["segment_plausible"] is True:
            segment_plausible_count += 1

    frame_count = max(len(samples), 1)
    mean_reprojection = float(np.mean(reprojection_means)) if reprojection_means else None
    mean_valid_3d_ratio = float(np.mean(valid_3d_ratios)) if valid_3d_ratios else 0.0
    mean_plausible_ratio = float(np.mean(plausible_metric_ratios)) if plausible_metric_ratios else 0.0
    reprojection_score = 0.0 if mean_reprojection is None else float(1.0 / (1.0 + (mean_reprojection / 10.0)))
    composite_score = (
        0.35 * mean_valid_3d_ratio
        + 0.35 * reprojection_score
        + 0.15 * (height_plausible_count / frame_count)
        + 0.15 * mean_plausible_ratio
    )
    return {
        "label": label,
        "camera_order_indices": list(camera_order_indices),
        "rotation_degrees_by_camera": [int(value) for value in rotation_degrees_by_camera],
        "frame_count": len(samples),
        "reprojection_error_px_mean_mean": mean_reprojection,
        "reprojection_error_px_max_max": float(np.max(reprojection_maxes)) if reprojection_maxes else None,
        "valid_3d_frame_ratio": mean_valid_3d_ratio,
        "height_plausible_frame_ratio": float(height_plausible_count / frame_count),
        "segment_plausible_frame_ratio": float(segment_plausible_count / frame_count),
        "plausible_metric_ratio_mean": mean_plausible_ratio,
        "composite_score": composite_score,
    }


def _evaluate_camera_orders(
    *,
    samples: Sequence[Dict[str, Any]],
    calibration: Any,
    triangulate_method: str,
    camera_ids: Sequence[str],
) -> Dict[str, Any]:
    camera_count = len(camera_ids)
    order_candidates: list[list[int]] = [list(range(camera_count))]
    if camera_count == 2:
        order_candidates.append([1, 0])
    seen: set[tuple[int, ...]] = set()
    candidates = []
    for indices in order_candidates:
        key = tuple(indices)
        if key in seen:
            continue
        seen.add(key)
        candidate_camera_ids = [camera_ids[index] for index in indices]
        label = _camera_order_label(candidate_camera_ids)
        result = _evaluate_candidate_on_samples(
            samples=samples,
            calibration=calibration,
            camera_order_indices=indices,
            rotation_degrees_by_camera=[0] * camera_count,
            triangulate_method=triangulate_method,
            label=label,
        )
        result["candidate_camera_ids"] = list(candidate_camera_ids)
        candidates.append(result)
        print(
            "[MocapDiagOrderTest] "
            f"order={label} score={result['composite_score']:.3f} "
            f"reproj={result['reprojection_error_px_mean_mean']} "
            f"valid3d={result['valid_3d_frame_ratio']:.3f} "
            f"height_ok={result['height_plausible_frame_ratio']:.3f} "
            f"segment_ok={result['segment_plausible_frame_ratio']:.3f}",
            flush=True,
        )
    best = max(
        candidates,
        key=lambda item: (
            item["composite_score"],
            item["valid_3d_frame_ratio"],
            -(item["reprojection_error_px_mean_mean"] or float("inf")),
        ),
    )
    print(
        f"[MocapDiagRecommend] recommended_camera_order={best['label']} score={best['composite_score']:.3f}",
        flush=True,
    )
    return {
        "recommended_camera_order": best["label"],
        "camera_order_candidates": candidates,
    }


def _evaluate_rotation_variants(
    *,
    samples: Sequence[Dict[str, Any]],
    calibration: Any,
    triangulate_method: str,
    base_camera_ids: Sequence[str],
    base_order_indices: Sequence[int],
) -> Dict[str, Any]:
    if len(base_camera_ids) != 2:
        return {
            "recommended_rotation_variant": None,
            "rotation_variant_candidates": [],
        }

    candidates = []
    for left_rotation in SUPPORTED_ROTATION_DEGREES:
        for right_rotation in SUPPORTED_ROTATION_DEGREES:
            rotations = [left_rotation, right_rotation]
            label = _rotation_variant_label(rotations)
            result = _evaluate_candidate_on_samples(
                samples=samples,
                calibration=calibration,
                camera_order_indices=base_order_indices,
                rotation_degrees_by_camera=rotations,
                triangulate_method=triangulate_method,
                label=label,
            )
            candidates.append(result)
            print(
                "[MocapDiagRotationTest] "
                f"rotations={label} score={result['composite_score']:.3f} "
                f"reproj={result['reprojection_error_px_mean_mean']} "
                f"valid3d={result['valid_3d_frame_ratio']:.3f} "
                f"height_ok={result['height_plausible_frame_ratio']:.3f} "
                f"segment_ok={result['segment_plausible_frame_ratio']:.3f}",
                flush=True,
            )

    best = max(
        candidates,
        key=lambda item: (
            item["composite_score"],
            item["valid_3d_frame_ratio"],
            -(item["reprojection_error_px_mean_mean"] or float("inf")),
        ),
    )
    print(
        f"[MocapDiagRecommend] recommended_rotation_variant={best['label']} score={best['composite_score']:.3f}",
        flush=True,
    )
    return {
        "recommended_rotation_variant": best["label"],
        "rotation_variant_candidates": candidates,
    }


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
    valid_3d_point_ratio = float(finite_3d.mean())
    jitter_m_per_s = _jitter_metrics(points_m, previous_points_m, dt_s)
    plausibility = _plausibility_metrics(points_m)
    segment_metrics = plausibility["segment_metrics"]

    flags = []
    valid_2d_ratio = float(source_diagnostics.get("valid_2d_point_ratio", np.nan))
    camera_skew_ms = source_diagnostics.get("camera_skew_ms")
    reprojection_mean = source_diagnostics.get("reprojection_error_px_mean")
    if np.isfinite(valid_2d_ratio) and valid_2d_ratio < args.low_valid_2d_ratio:
        flags.append("low_valid_2d")
    if camera_skew_ms is not None and float(camera_skew_ms) > args.max_camera_skew_ms:
        flags.append("high_camera_skew")
    if segment_metrics["max_abs_point_m"] is not None and segment_metrics["max_abs_point_m"] > args.max_abs_point_m:
        flags.append("point_out_of_range")
    if reprojection_mean is not None and float(reprojection_mean) > args.bad_reprojection_error_px:
        flags.append("bad_reprojection")
    if plausibility["height_plausible"] is False:
        flags.append("implausible_height")
    if plausibility["segment_plausible"] is False:
        flags.append("implausible_segment")
    if largest_bone_change is not None and largest_bone_change > args.max_bone_length_change_m:
        flags.append("bone_length_jump")
    for name, speed in jitter_m_per_s.items():
        if speed is not None and speed > args.max_jitter_speed_mps:
            flags.append(f"{name}_jitter")

    track_ms = float(source_diagnostics.get("tracking_ms", np.nan))
    triangulate_ms = float(source_diagnostics.get("triangulate_3d_ms", np.nan))
    total_ms = float(source_diagnostics.get("total_ms", np.nan))
    return {
        "seq": int(seq),
        "timestamp_ns": int(timestamp_ns),
        "valid_2d_point_ratio": valid_2d_ratio,
        "valid_3d_point_ratio": valid_3d_point_ratio,
        "per_camera_2d_valid_ratios": _as_jsonable(source_diagnostics.get("per_camera_2d_valid_ratios", [])),
        "camera_ids": _as_jsonable(source_diagnostics.get("camera_ids", [])),
        "camera_timestamps_ns": _as_jsonable(source_diagnostics.get("camera_timestamps_ns", [])),
        "camera_skew_ms": None if camera_skew_ms is None else float(camera_skew_ms),
        "tracking_ms": track_ms,
        "triangulate_3d_ms": triangulate_ms,
        "total_tracking_and_triangulation_ms": total_ms,
        "tracking_fps_estimate": None if not math.isfinite(track_ms) or track_ms <= 0.0 else 1000.0 / track_ms,
        "triangulate_fps_estimate": None if not math.isfinite(triangulate_ms) or triangulate_ms <= 0.0 else 1000.0 / triangulate_ms,
        "reprojection_error_px_mean": None if source_diagnostics.get("reprojection_error_px_mean") is None else float(source_diagnostics["reprojection_error_px_mean"]),
        "reprojection_error_px_max": None if source_diagnostics.get("reprojection_error_px_max") is None else float(source_diagnostics["reprojection_error_px_max"]),
        "per_camera_reprojection_error_px_mean": _as_jsonable(source_diagnostics.get("per_camera_reprojection_error_px_mean", [])),
        "per_camera_reprojection_error_px_max": _as_jsonable(source_diagnostics.get("per_camera_reprojection_error_px_max", [])),
        "nan_3d_point_count": int((~finite_3d).sum()),
        "named_points_m": _named_points(points_m),
        "bone_lengths_m": bone_lengths,
        "largest_bone_length_change_m": largest_bone_change,
        "jitter_m_per_s": jitter_m_per_s,
        "estimated_height_m": segment_metrics["estimated_height_m"],
        "shoulder_width_m": segment_metrics["shoulder_width_m"],
        "hip_width_m": segment_metrics["hip_width_m"],
        "left_thigh_length_m": segment_metrics["left_thigh_length_m"],
        "right_thigh_length_m": segment_metrics["right_thigh_length_m"],
        "left_shin_length_m": segment_metrics["left_shin_length_m"],
        "right_shin_length_m": segment_metrics["right_shin_length_m"],
        "left_foot_length_m": segment_metrics["left_foot_length_m"],
        "right_foot_length_m": segment_metrics["right_foot_length_m"],
        "max_abs_point_m": segment_metrics["max_abs_point_m"],
        "plausible_metric_ratio": plausibility["plausible_metric_ratio"],
        "height_plausible": plausibility["height_plausible"],
        "segment_plausible": plausibility["segment_plausible"],
        "flags": flags,
        "last_problem": _pick_last_problem(flags),
    }


class DiagnosticsWriter(threading.Thread):
    def __init__(self, diagnostics_jsonl_path: Path, points_path: Path) -> None:
        super().__init__(daemon=True)
        self._diagnostics_jsonl_path = diagnostics_jsonl_path
        self._points_path = points_path
        self._queue: queue.SimpleQueue[tuple[str, Any, Optional[np.ndarray]]] = queue.SimpleQueue()
        self.points_history: list[np.ndarray] = []
        self.errors: list[str] = []

    def write_frame(self, metrics: Dict[str, Any], points_m: np.ndarray) -> None:
        self._queue.put(("frame", dict(metrics), np.asarray(points_m, dtype=np.float32).copy()))

    def finish(self) -> None:
        self._queue.put(("stop", None, None))
        self.join()

    def run(self) -> None:
        try:
            with self._diagnostics_jsonl_path.open("w", encoding="utf-8") as diagnostics_file:
                while True:
                    kind, payload, points_m = self._queue.get()
                    if kind == "stop":
                        break
                    diagnostics_file.write(json.dumps(_as_jsonable(payload), ensure_ascii=False) + "\n")
                    diagnostics_file.flush()
                    if points_m is not None:
                        self.points_history.append(points_m)
        except Exception as exc:
            self.errors.append(str(exc))
        finally:
            if self.points_history:
                np.save(self._points_path, np.stack(self.points_history, axis=0).astype(np.float32))
            else:
                np.save(self._points_path, np.empty((0, 33, 3), dtype=np.float32))


def _viewer_process_main(packet_queue: Any, viewer_range_m: float, follow_pelvis: bool) -> None:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QApplication, QLabel, QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget

    class ViewerWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Skellycam 3D mocap diagnostics")
            self.resize(960, 700)
            layout = QVBoxLayout(self)
            self.overlay = QLabel()
            self.overlay.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self.overlay.setWordWrap(True)
            layout.addWidget(self.overlay)
            self.view = QGraphicsView()
            self.scene = QGraphicsScene(self)
            self.view.setScene(self.scene)
            layout.addWidget(self.view, stretch=1)
            self._latest_packet: Optional[Dict[str, Any]] = None
            self._last_sane_center = np.zeros(3, dtype=np.float64)
            self._follow_pelvis = bool(follow_pelvis)
            self._viewer_range_m = max(float(viewer_range_m), DEFAULT_VIEWER_RANGE_M)
            self._z_range_m = max(3.0, self._viewer_range_m * 0.75)

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._poll_and_render)
            self._timer.start(16)

        def _poll_and_render(self) -> None:
            while True:
                try:
                    packet = packet_queue.get_nowait()
                except queue.Empty:
                    break
                if packet is None:
                    QApplication.quit()
                    return
                self._latest_packet = packet
            if self._latest_packet is not None:
                self._render_packet(self._latest_packet)

        def _select_center(self, points_m: np.ndarray, max_abs_point_m: Optional[float]) -> np.ndarray:
            if max_abs_point_m is not None and max_abs_point_m > 10.0:
                return self._last_sane_center
            pelvis = _midpoint(points_m, "left_hip", "right_hip")
            if self._follow_pelvis and _finite_point(pelvis):
                self._last_sane_center = pelvis
                return pelvis
            finite_points = points_m[np.isfinite(points_m).all(axis=1)]
            if finite_points.size:
                center = np.nanmean(finite_points, axis=0)
                self._last_sane_center = center
                return center
            return self._last_sane_center

        def _project(self, point: np.ndarray, center: np.ndarray, view_kind: str, rect: tuple[float, float, float, float]) -> Optional[tuple[float, float]]:
            if not _finite_point(point):
                return None
            left, top, width, height = rect
            x_half = self._viewer_range_m / 2.0
            y_half = self._viewer_range_m / 2.0
            z_half = self._z_range_m / 2.0
            if view_kind == "front":
                horizontal = (point[0] - center[0]) / max(x_half, 1e-6)
                vertical = (point[1] - center[1]) / max(y_half, 1e-6)
            else:
                horizontal = (point[2] - center[2]) / max(z_half, 1e-6)
                vertical = (point[1] - center[1]) / max(y_half, 1e-6)
            px = left + ((horizontal + 1.0) * 0.5 * width)
            py = top + ((1.0 - (vertical + 1.0) * 0.5) * height)
            return px, py

        def _render_packet(self, packet: Dict[str, Any]) -> None:
            points_m = np.asarray(packet["points_m"], dtype=np.float32).reshape(33, 3)
            max_abs_point_m = packet.get("max_abs_point_m")
            center = self._select_center(points_m, max_abs_point_m)
            self.scene.clear()
            scene_width = 900.0
            scene_height = 560.0
            self.scene.setSceneRect(0.0, 0.0, scene_width, scene_height)
            front_rect = (20.0, 20.0, 400.0, 520.0)
            side_rect = (480.0, 20.0, 400.0, 520.0)
            self.scene.addRect(*front_rect, pen=QPen(QColor("#555555")))
            self.scene.addRect(*side_rect, pen=QPen(QColor("#555555")))
            self.scene.addText("Front X/Y").setPos(front_rect[0], 0.0)
            self.scene.addText("Side Z/Y").setPos(side_rect[0], 0.0)

            def point_color(index: int) -> QColor:
                if index in {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}:
                    return QColor("#1f77b4")
                if index in {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}:
                    return QColor("#d62728")
                return QColor("#888888")

            for rect, view_kind in ((front_rect, "front"), (side_rect, "side")):
                for start_name, end_name in SKELETON_EDGES:
                    start = _point(points_m, start_name)
                    end = _point(points_m, end_name)
                    start_xy = self._project(start, center, view_kind, rect)
                    end_xy = self._project(end, center, view_kind, rect)
                    if start_xy is None or end_xy is None:
                        continue
                    self.scene.addLine(
                        start_xy[0],
                        start_xy[1],
                        end_xy[0],
                        end_xy[1],
                        pen=QPen(QColor("#4c78a8"), 2.0),
                    )
                for index, point in enumerate(points_m):
                    projected = self._project(point, center, view_kind, rect)
                    if projected is None:
                        continue
                    radius = 4.0
                    self.scene.addEllipse(
                        projected[0] - radius,
                        projected[1] - radius,
                        radius * 2.0,
                        radius * 2.0,
                        pen=QPen(point_color(index)),
                        brush=point_color(index),
                    )

            warning = ""
            if max_abs_point_m is not None and max_abs_point_m > 10.0:
                warning = " | 3D out of range"
            self.overlay.setText(
                " ".join(
                    part
                    for part in [
                        f"seq={packet.get('seq')}",
                        f"valid2d={packet.get('valid_2d_point_ratio', 0.0) * 100.0:.1f}%",
                        f"reproj={packet.get('reprojection_error_px_mean')}",
                        f"height={packet.get('estimated_height_m')}",
                        f"problem={packet.get('last_problem')}",
                        f"order={packet.get('camera_order_label')}",
                        warning.strip(),
                    ]
                    if part
                )
            )

    app = QApplication([])
    _window = ViewerWindow()
    _window.show()
    app.exec()


class ViewerProcessHandle:
    def __init__(self, *, enabled: bool, viewer_range_m: float, follow_pelvis: bool) -> None:
        self.enabled = enabled
        self.process: Optional[mp.Process] = None
        self.queue: Any = None
        if not enabled:
            return
        try:
            context = mp.get_context("spawn")
            self.queue = context.Queue(maxsize=1)
            self.process = context.Process(
                target=_viewer_process_main,
                args=(self.queue, viewer_range_m, follow_pelvis),
                daemon=True,
            )
            self.process.start()
        except Exception as exc:
            print(f"[MocapDiagWarning] viewer disabled: {exc}", flush=True)
            self.enabled = False
            self.process = None
            self.queue = None

    def publish(self, packet: Dict[str, Any]) -> None:
        if not self.enabled or self.queue is None:
            return
        try:
            replace_latest_queue_item(self.queue, packet)
        except Exception:
            self.enabled = False

    def close(self) -> None:
        if self.queue is not None:
            try:
                replace_latest_queue_item(self.queue, None)
            except Exception:
                pass
        if self.process is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)


def _stats_line(metrics: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
    recommended_order = recommendations.get("recommended_camera_order")
    return (
        "[MocapDiagStats] "
        f"seq={metrics['seq']} fps={metrics['mocap_fps']:.1f} "
        f"track_fps={metrics.get('tracking_fps_estimate')} "
        f"tri_fps={metrics.get('triangulate_fps_estimate')} "
        f"valid2d={metrics['valid_2d_point_ratio'] * 100.0:.1f}% "
        f"reproj={metrics.get('reprojection_error_px_mean')} "
        f"height={metrics.get('estimated_height_m')} "
        f"skew_ms={metrics.get('camera_skew_ms')} "
        f"recommended_order={recommended_order} "
        f"last={metrics['last_problem']}"
    )


def _summarize(
    metrics: list[Dict[str, Any]],
    output_folder: Path,
    args: argparse.Namespace,
    startup_report: Dict[str, Any],
    recommendations: Dict[str, Any],
) -> Dict[str, Any]:
    flag_counts = Counter(flag for metric in metrics for flag in metric.get("flags", []))
    suspicious_frames = [
        {"seq": metric["seq"], "flags": metric["flags"], "last_problem": metric["last_problem"]}
        for metric in metrics
        if metric.get("flags")
    ]
    duration_s = None
    if len(metrics) >= 2:
        duration_s = max(0.0, (metrics[-1]["timestamp_ns"] - metrics[0]["timestamp_ns"]) / 1e9)

    def mean_of(name: str) -> Optional[float]:
        values = [float(metric[name]) for metric in metrics if metric.get(name) is not None and math.isfinite(float(metric[name]))]
        return None if not values else float(np.mean(values))

    def min_of(name: str) -> Optional[float]:
        values = [float(metric[name]) for metric in metrics if metric.get(name) is not None and math.isfinite(float(metric[name]))]
        return None if not values else float(np.min(values))

    def max_of(name: str) -> Optional[float]:
        values = [float(metric[name]) for metric in metrics if metric.get(name) is not None and math.isfinite(float(metric[name]))]
        return None if not values else float(np.max(values))

    summary = {
        "frame_count": len(metrics),
        "duration_s": duration_s,
        "mean_mocap_fps": None if not duration_s else float((len(metrics) - 1) / duration_s),
        "valid_2d_point_ratio_mean": mean_of("valid_2d_point_ratio"),
        "valid_2d_point_ratio_min": min_of("valid_2d_point_ratio"),
        "valid_3d_point_ratio_mean": mean_of("valid_3d_point_ratio"),
        "reprojection_error_px_mean_mean": mean_of("reprojection_error_px_mean"),
        "reprojection_error_px_max_max": max_of("reprojection_error_px_max"),
        "camera_skew_ms_mean": mean_of("camera_skew_ms"),
        "camera_skew_ms_max": max_of("camera_skew_ms"),
        "tracking_and_triangulation_ms_mean": mean_of("total_tracking_and_triangulation_ms"),
        "tracking_and_triangulation_ms_max": max_of("total_tracking_and_triangulation_ms"),
        "estimated_height_m_mean": mean_of("estimated_height_m"),
        "shoulder_width_m_mean": mean_of("shoulder_width_m"),
        "hip_width_m_mean": mean_of("hip_width_m"),
        "plausible_metric_ratio_mean": mean_of("plausible_metric_ratio"),
        "flag_counts": dict(flag_counts),
        "suspicious_frames": suspicious_frames[:200],
        "output_folder": str(output_folder),
        "points_3d_npy": str(output_folder / "mocap_3d_body_points.npy"),
        "diagnostics_jsonl": str(output_folder / "diagnostics.jsonl"),
        "startup_report": startup_report,
        "selected_camera_order": args.camera_ids,
        "triangulate_method": args.triangulate_method,
        "recommended_camera_order": recommendations.get("recommended_camera_order"),
        "camera_order_candidates": recommendations.get("camera_order_candidates", []),
        "recommended_rotation_variant": recommendations.get("recommended_rotation_variant"),
        "rotation_variant_candidates": recommendations.get("rotation_variant_candidates", []),
        "args": vars(args),
    }
    summary["likely_cause"] = classify_likely_cause(summary, max_camera_skew_ms=float(args.max_camera_skew_ms))
    return summary


def _run_preflight_recommendations(args: argparse.Namespace, calibration: Any, camera_ids: Sequence[str]) -> Dict[str, Any]:
    recommendations: Dict[str, Any] = {
        "recommended_camera_order": None,
        "camera_order_candidates": [],
        "recommended_rotation_variant": None,
        "rotation_variant_candidates": [],
    }
    if not args.try_camera_orders and not args.try_rotation_variants:
        return recommendations

    sample_target = max(
        args.order_test_frames if args.try_camera_orders else 0,
        args.rotation_test_frames if args.try_rotation_variants else 0,
    )
    if sample_target <= 0:
        return recommendations

    samples = _collect_preflight_samples(args, frame_limit=sample_target)
    if not samples:
        print("[MocapDiagWarning] preflight could not collect enough valid samples", flush=True)
        return recommendations

    selected_order_indices = list(range(len(camera_ids)))
    if args.try_camera_orders:
        order_report = _evaluate_camera_orders(
            samples=samples[: args.order_test_frames],
            calibration=calibration,
            triangulate_method=args.triangulate_method,
            camera_ids=camera_ids,
        )
        recommendations.update(order_report)
        if order_report["recommended_camera_order"] is not None:
            best_order_candidate = next(
                (
                    candidate
                    for candidate in order_report["camera_order_candidates"]
                    if candidate["label"] == order_report["recommended_camera_order"]
                ),
                None,
            )
            if best_order_candidate is not None:
                selected_order_indices = list(best_order_candidate["camera_order_indices"])

    if args.try_rotation_variants:
        rotation_report = _evaluate_rotation_variants(
            samples=samples[: args.rotation_test_frames],
            calibration=calibration,
            triangulate_method=args.triangulate_method,
            base_camera_ids=[camera_ids[index] for index in selected_order_indices],
            base_order_indices=selected_order_indices,
        )
        recommendations.update(rotation_report)
    return recommendations


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
    parser.add_argument("--triangulate-method", choices=["simple", "ransac"], default="simple")
    parser.add_argument("--try-camera-orders", action="store_true")
    parser.add_argument("--order-test-frames", type=int, default=DEFAULT_ORDER_TEST_FRAMES)
    parser.add_argument("--try-rotation-variants", action="store_true")
    parser.add_argument("--rotation-test-frames", type=int, default=DEFAULT_ROTATION_TEST_FRAMES)
    parser.add_argument("--max-camera-skew-ms", type=float, default=DEFAULT_MAX_CAMERA_SKEW_MS)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--viewer", dest="viewer", action="store_true", default=True)
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--viewer-range-m", type=float, default=DEFAULT_VIEWER_RANGE_M)
    parser.add_argument("--viewer-follow-pelvis", dest="viewer_follow_pelvis", action="store_true", default=True)
    parser.add_argument("--viewer-lock-first-valid", dest="viewer_follow_pelvis", action="store_false")
    parser.add_argument("--print-every-frames", type=int, default=30)
    parser.add_argument("--start-when-valid", dest="start_when_valid", action="store_true", default=True)
    parser.add_argument("--no-start-when-valid", dest="start_when_valid", action="store_false")
    parser.add_argument("--start-valid-2d-ratio", type=float, default=0.95)
    parser.add_argument("--start-stable-frames", type=int, default=5)
    parser.add_argument("--low-valid-2d-ratio", type=float, default=0.75)
    parser.add_argument("--bad-reprojection-error-px", type=float, default=DEFAULT_BAD_REPROJECTION_ERROR_PX)
    parser.add_argument("--max-bone-length-change-m", type=float, default=0.20)
    parser.add_argument("--max-jitter-speed-mps", type=float, default=5.0)
    parser.add_argument("--max-abs-point-m", type=float, default=10.0)
    parser.add_argument("--skip-triangulation-prewarm", action="store_true")
    return parser


def main() -> None:
    mp.freeze_support()
    parser = _build_arg_parser()
    args = parser.parse_args()
    output_folder = (args.output_folder or _default_output_folder()).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.order_test_frames <= 0:
        raise ValueError("--order-test-frames must be positive")
    if args.rotation_test_frames <= 0:
        raise ValueError("--rotation-test-frames must be positive")

    calibration = load_anipose_calibration(args.calibration_toml)
    camera_ids = parse_camera_ids(args.camera_ids) or [str(index) for index in range(len(calibration.cameras))]
    camera_configs = load_camera_config_json(args.camera_config_json)
    startup_report = _build_startup_consistency_report(calibration, camera_ids, camera_configs)
    _print_startup_report(startup_report)

    recommendations = _run_preflight_recommendations(args, calibration, camera_ids)
    diagnostics_jsonl_path = output_folder / "diagnostics.jsonl"
    points_path = output_folder / "mocap_3d_body_points.npy"
    writer = DiagnosticsWriter(diagnostics_jsonl_path, points_path)
    writer.start()
    viewer = ViewerProcessHandle(
        enabled=bool(args.viewer),
        viewer_range_m=float(args.viewer_range_m),
        follow_pelvis=bool(args.viewer_follow_pelvis),
    )

    latest_source_diagnostics: Dict[str, Any] = {}
    metrics_history: list[Dict[str, Any]] = []
    previous_points_m: Optional[np.ndarray] = None
    previous_timestamp_ns: Optional[int] = None
    previous_bone_lengths: Optional[Dict[str, Optional[float]]] = None
    wall_start = time.perf_counter()
    recording_started = not bool(args.start_when_valid)
    valid_start_streak = 0

    def diagnostics_callback(diagnostics: Dict[str, Any]) -> None:
        latest_source_diagnostics.clear()
        latest_source_diagnostics.update(diagnostics)

    print(f"Skellycam mocap diagnostics starting. Output: {output_folder}", flush=True)
    if args.start_when_valid:
        print(
            "Waiting for a valid person before counting diagnostic frames "
            f"(valid2d>={args.start_valid_2d_ratio}, stable_frames={args.start_stable_frames}).",
            flush=True,
        )

    try:
        for frame in iter_skellycam_mocap_3d_frames(
            max_frames=None if args.start_when_valid else args.max_frames,
            diagnostics_callback=diagnostics_callback,
            **_source_generator_kwargs(args),
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

            if not recording_started:
                has_valid_2d = float(metrics["valid_2d_point_ratio"]) >= float(args.start_valid_2d_ratio)
                if has_valid_2d:
                    valid_start_streak += 1
                else:
                    valid_start_streak = 0
                viewer.publish(
                    {
                        "seq": frame.seq,
                        "points_m": points_m,
                        "valid_2d_point_ratio": metrics["valid_2d_point_ratio"],
                        "reprojection_error_px_mean": metrics["reprojection_error_px_mean"],
                        "estimated_height_m": metrics["estimated_height_m"],
                        "last_problem": f"waiting_{valid_start_streak}/{args.start_stable_frames}",
                        "max_abs_point_m": metrics["max_abs_point_m"],
                        "camera_order_label": recommendations.get("recommended_camera_order") or args.camera_ids,
                    }
                )
                if frame.seq == 0 or frame.seq % max(1, args.print_every_frames) == 0:
                    print(
                        "[MocapDiagWaiting] "
                        f"seq={frame.seq} valid2d={frame.valid_2d_point_ratio * 100.0:.1f}% "
                        f"nan3d={metrics['nan_3d_point_count']} "
                        f"streak={valid_start_streak}/{args.start_stable_frames}",
                        flush=True,
                    )
                if valid_start_streak < int(args.start_stable_frames):
                    continue
                recording_started = True
                wall_start = time.perf_counter()
                previous_points_m = None
                previous_timestamp_ns = None
                previous_bone_lengths = None
                print(f"[MocapDiagStarted] seq={frame.seq}", flush=True)

            elapsed_s = max(time.perf_counter() - wall_start, 1e-6)
            metrics["mocap_fps"] = float((len(metrics_history) + 1) / elapsed_s)
            metrics["recommended_camera_order"] = recommendations.get("recommended_camera_order")
            metrics["recommended_rotation_variant"] = recommendations.get("recommended_rotation_variant")

            writer.write_frame(metrics, points_m)
            metrics_history.append(metrics)

            previous_points_m = points_m
            previous_timestamp_ns = int(frame.timestamp_ns)
            previous_bone_lengths = metrics["bone_lengths_m"]

            viewer.publish(
                {
                    "seq": frame.seq,
                    "points_m": points_m,
                    "valid_2d_point_ratio": metrics["valid_2d_point_ratio"],
                    "reprojection_error_px_mean": metrics["reprojection_error_px_mean"],
                    "estimated_height_m": metrics["estimated_height_m"],
                    "last_problem": metrics["last_problem"],
                    "max_abs_point_m": metrics["max_abs_point_m"],
                    "camera_order_label": recommendations.get("recommended_camera_order") or args.camera_ids,
                }
            )

            if args.print_every_frames > 0 and (frame.seq == 0 or frame.seq % args.print_every_frames == 0):
                print(_stats_line(metrics, recommendations), flush=True)
            if args.max_frames is not None and len(metrics_history) >= int(args.max_frames):
                break
    except KeyboardInterrupt:
        print("Diagnostics interrupted; saving collected frames.", flush=True)
    finally:
        viewer.close()
        writer.finish()
        summary = _summarize(metrics_history, output_folder, args, startup_report, recommendations)
        summary_path = output_folder / "summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(_as_jsonable(summary), file, ensure_ascii=False, indent=2)
        if writer.errors:
            print(f"[MocapDiagWarning] writer errors: {writer.errors}", flush=True)
        print(f"Saved 3D points: {points_path}", flush=True)
        print(f"Saved diagnostics: {diagnostics_jsonl_path}", flush=True)
        print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
