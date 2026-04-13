"""
Experimental FreeMoCap -> GMR -> sim2real bridge.

This keeps the main FreeMoCap pipeline untouched. It replays/streams pose-only
FreeMoCap 3D body points, converts them to an xrobot-like pose dictionary,
retargets them to Unitree G1 with GMR, and serves the existing sim2real
VRMotionSource ZMQ request/reply protocol.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from gmr_runtime import ensure_gmr_paths, validate_gmr_runtime
from realtime_mocap_probe import iter_mocap_3d_frames
from skellycam_live_source import iter_skellycam_mocap_3d_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
ensure_gmr_paths(REPO_ROOT)

MEDIAPIPE = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
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

XR_BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]

DATASET_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_DOF_POS_G1 = np.array(
    [
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
DEFAULT_QPOS_G1 = np.concatenate(
    [
        np.array([0.0, 0.0, 0.8], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        DEFAULT_DOF_POS_G1,
    ],
    axis=0,
)

IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class RetargetedFrame:
    recv_ns: int
    qpos: np.ndarray


def _finite_point(point: np.ndarray) -> bool:
    return bool(np.asarray(point).shape == (3,) and np.isfinite(point).all())


def _safe_norm(vector: np.ndarray, eps: float = 1e-8) -> float:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        return 0.0
    return norm


def _unit(vector: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = _safe_norm(v)
    if norm == 0.0:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return v / norm


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        return IDENTITY_QUAT_WXYZ.copy()
    return (q / norm).astype(np.float32)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return _normalize_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float64))


def _frame_quat_from_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    fallback: np.ndarray = IDENTITY_QUAT_WXYZ,
) -> np.ndarray:
    x = _unit(x_axis, [1.0, 0.0, 0.0])
    y = _unit(y_axis - x * float(np.dot(y_axis, x)), [0.0, 1.0, 0.0])
    z = np.cross(x, y)
    if _safe_norm(z) == 0.0:
        return fallback.astype(np.float32, copy=True)
    z = _unit(z, [0.0, 0.0, 1.0])
    y = _unit(np.cross(z, x), [0.0, 1.0, 0.0])
    matrix = np.stack([x, y, z], axis=1)
    if float(np.linalg.det(matrix)) < 0.0:
        y = -y
        matrix = np.stack([x, y, z], axis=1)
    return _matrix_to_quat_wxyz(matrix)


def _segment_quat(start: np.ndarray, end: np.ndarray, up_hint: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if not (_finite_point(start) and _finite_point(end)):
        return fallback.astype(np.float32, copy=True)
    x_axis = end - start
    if _safe_norm(x_axis) == 0.0:
        return fallback.astype(np.float32, copy=True)
    y_axis = np.cross(up_hint, x_axis)
    if _safe_norm(y_axis) == 0.0:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = np.cross(x_axis, y_axis)
    return _frame_quat_from_axes(x_axis, y_axis, z_axis, fallback=fallback)


def _slerp_quat_wxyz(quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(quat0).astype(np.float64)
    q1 = _normalize_quat_wxyz(quat1).astype(np.float64)
    t = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat_wxyz(q0 + t * (q1 - q0))
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if abs(sin_theta_0) < 1e-8:
        return _normalize_quat_wxyz(q0)
    theta = theta_0 * t
    return _normalize_quat_wxyz(
        q0 * (np.sin(theta_0 - theta) / sin_theta_0) + q1 * (np.sin(theta) / sin_theta_0)
    )


def _interpolate_qpos(prev_qpos: np.ndarray, next_qpos: np.ndarray, alpha: float) -> np.ndarray:
    t = float(np.clip(alpha, 0.0, 1.0))
    frame = np.asarray(prev_qpos, dtype=np.float32) * (1.0 - t) + np.asarray(next_qpos, dtype=np.float32) * t
    frame[3:7] = _slerp_quat_wxyz(prev_qpos[3:7], next_qpos[3:7], t)
    return frame.astype(np.float32)


class FreeMoCapXRobotConverter:
    def __init__(self, source_origin: np.ndarray, source_to_target_basis: np.ndarray):
        self.source_origin = np.asarray(source_origin, dtype=np.float64).reshape(3)
        self.source_to_target_basis = np.asarray(source_to_target_basis, dtype=np.float64).reshape(3, 3)
        self.previous_points_by_name: Dict[str, np.ndarray] = {}
        self.previous_quats_by_name: Dict[str, np.ndarray] = {}

    @classmethod
    def from_calibration_frames(cls, frames_m: Sequence[np.ndarray]) -> "FreeMoCapXRobotConverter":
        usable = [np.asarray(frame, dtype=np.float64).reshape(33, 3) for frame in frames_m if cls._frame_has_torso(frame)]
        if not usable:
            raise ValueError("No usable calibration frames with hips and shoulders.")
        mean_points = np.nanmean(np.stack(usable, axis=0), axis=0)
        pelvis = 0.5 * (mean_points[MEDIAPIPE["left_hip"]] + mean_points[MEDIAPIPE["right_hip"]])
        shoulder_center = 0.5 * (
            mean_points[MEDIAPIPE["left_shoulder"]] + mean_points[MEDIAPIPE["right_shoulder"]]
        )
        body_right = _unit(mean_points[MEDIAPIPE["right_hip"]] - mean_points[MEDIAPIPE["left_hip"]], [0.0, 1.0, 0.0])
        body_up = _unit(shoulder_center - pelvis, [0.0, 0.0, 1.0])
        body_forward = _unit(np.cross(body_right, body_up), [1.0, 0.0, 0.0])
        body_up = _unit(np.cross(body_forward, body_right), [0.0, 0.0, 1.0])
        basis = np.stack([body_forward, body_right, body_up], axis=1)
        return cls(source_origin=pelvis, source_to_target_basis=basis)

    @staticmethod
    def _frame_has_torso(points: np.ndarray) -> bool:
        required = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        return all(_finite_point(np.asarray(points)[MEDIAPIPE[name]]) for name in required)

    def _transform_points(self, points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64).reshape(33, 3)
        transformed = (points - self.source_origin) @ self.source_to_target_basis
        transformed[~np.isfinite(points).all(axis=1)] = np.nan
        return transformed

    def _mp(self, points: np.ndarray, name: str) -> np.ndarray:
        return np.asarray(points[MEDIAPIPE[name]], dtype=np.float64)

    def _fallback_point(self, key: str, candidate: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if _finite_point(candidate):
            point = candidate.astype(np.float64, copy=True)
        elif key in self.previous_points_by_name:
            point = self.previous_points_by_name[key].astype(np.float64, copy=True)
        else:
            point = fallback.astype(np.float64, copy=True)
        self.previous_points_by_name[key] = point
        return point

    def _hand_point(self, points: np.ndarray, side: str) -> np.ndarray:
        wrist = self._mp(points, f"{side}_wrist")
        index = self._mp(points, f"{side}_index")
        pinky = self._mp(points, f"{side}_pinky")
        candidates = [p for p in (wrist, index, pinky) if _finite_point(p)]
        if candidates:
            return np.mean(np.stack(candidates, axis=0), axis=0)
        return wrist

    def _foot_point(self, points: np.ndarray, side: str) -> np.ndarray:
        foot_index = self._mp(points, f"{side}_foot_index")
        heel = self._mp(points, f"{side}_heel")
        ankle = self._mp(points, f"{side}_ankle")
        candidates = [p for p in (foot_index, heel, ankle) if _finite_point(p)]
        if candidates:
            return np.mean(np.stack(candidates, axis=0), axis=0)
        return ankle

    def _build_positions(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        left_hip = self._mp(points, "left_hip")
        right_hip = self._mp(points, "right_hip")
        left_shoulder = self._mp(points, "left_shoulder")
        right_shoulder = self._mp(points, "right_shoulder")
        pelvis_raw = 0.5 * (left_hip + right_hip)
        shoulders_raw = 0.5 * (left_shoulder + right_shoulder)
        pelvis = self._fallback_point("Pelvis", pelvis_raw, np.zeros(3, dtype=np.float64))
        neck = self._fallback_point("Neck", shoulders_raw, pelvis + np.array([0.0, 0.0, 0.45]))

        head_candidates = [
            p for p in (self._mp(points, "left_ear"), self._mp(points, "right_ear"), self._mp(points, "nose"))
            if _finite_point(p)
        ]
        head_raw = np.mean(np.stack(head_candidates, axis=0), axis=0) if head_candidates else neck

        return {
            "Pelvis": pelvis,
            "Left_Hip": self._fallback_point("Left_Hip", left_hip, pelvis),
            "Right_Hip": self._fallback_point("Right_Hip", right_hip, pelvis),
            "Spine1": self._fallback_point("Spine1", pelvis * 0.75 + neck * 0.25, pelvis),
            "Left_Knee": self._fallback_point("Left_Knee", self._mp(points, "left_knee"), pelvis),
            "Right_Knee": self._fallback_point("Right_Knee", self._mp(points, "right_knee"), pelvis),
            "Spine2": self._fallback_point("Spine2", pelvis * 0.5 + neck * 0.5, pelvis),
            "Left_Ankle": self._fallback_point("Left_Ankle", self._mp(points, "left_ankle"), pelvis),
            "Right_Ankle": self._fallback_point("Right_Ankle", self._mp(points, "right_ankle"), pelvis),
            "Spine3": self._fallback_point("Spine3", pelvis * 0.25 + neck * 0.75, neck),
            "Left_Foot": self._fallback_point("Left_Foot", self._foot_point(points, "left"), pelvis),
            "Right_Foot": self._fallback_point("Right_Foot", self._foot_point(points, "right"), pelvis),
            "Neck": neck,
            "Left_Collar": self._fallback_point("Left_Collar", left_shoulder, neck),
            "Right_Collar": self._fallback_point("Right_Collar", right_shoulder, neck),
            "Head": self._fallback_point("Head", head_raw, neck + np.array([0.0, 0.0, 0.2])),
            "Left_Shoulder": self._fallback_point("Left_Shoulder", left_shoulder, neck),
            "Right_Shoulder": self._fallback_point("Right_Shoulder", right_shoulder, neck),
            "Left_Elbow": self._fallback_point("Left_Elbow", self._mp(points, "left_elbow"), left_shoulder),
            "Right_Elbow": self._fallback_point("Right_Elbow", self._mp(points, "right_elbow"), right_shoulder),
            "Left_Wrist": self._fallback_point("Left_Wrist", self._mp(points, "left_wrist"), left_shoulder),
            "Right_Wrist": self._fallback_point("Right_Wrist", self._mp(points, "right_wrist"), right_shoulder),
            "Left_Hand": self._fallback_point("Left_Hand", self._hand_point(points, "left"), left_shoulder),
            "Right_Hand": self._fallback_point("Right_Hand", self._hand_point(points, "right"), right_shoulder),
        }

    def _torso_quat(self, positions: Dict[str, np.ndarray]) -> np.ndarray:
        body_right = _unit(positions["Right_Hip"] - positions["Left_Hip"], [0.0, 1.0, 0.0])
        body_up = _unit(positions["Neck"] - positions["Pelvis"], [0.0, 0.0, 1.0])
        body_forward = _unit(np.cross(body_right, body_up), [1.0, 0.0, 0.0])
        body_up = _unit(np.cross(body_forward, body_right), [0.0, 0.0, 1.0])
        return _frame_quat_from_axes(body_forward, body_right, body_up)

    def _quat_for(self, key: str, quat: np.ndarray) -> np.ndarray:
        q = _normalize_quat_wxyz(quat)
        self.previous_quats_by_name[key] = q
        return q

    def _build_quats(self, positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        torso = self._torso_quat(positions)
        up_hint = _unit(positions["Neck"] - positions["Pelvis"], [0.0, 0.0, 1.0])
        quats = {
            "Pelvis": torso,
            "Spine1": torso,
            "Spine2": torso,
            "Spine3": torso,
            "Neck": torso,
            "Left_Collar": torso,
            "Right_Collar": torso,
        }
        segments = {
            "Head": ("Neck", "Head"),
            "Left_Hip": ("Pelvis", "Left_Hip"),
            "Right_Hip": ("Pelvis", "Right_Hip"),
            "Left_Knee": ("Left_Hip", "Left_Knee"),
            "Right_Knee": ("Right_Hip", "Right_Knee"),
            "Left_Ankle": ("Left_Knee", "Left_Ankle"),
            "Right_Ankle": ("Right_Knee", "Right_Ankle"),
            "Left_Foot": ("Left_Ankle", "Left_Foot"),
            "Right_Foot": ("Right_Ankle", "Right_Foot"),
            "Left_Shoulder": ("Neck", "Left_Shoulder"),
            "Right_Shoulder": ("Neck", "Right_Shoulder"),
            "Left_Elbow": ("Left_Shoulder", "Left_Elbow"),
            "Right_Elbow": ("Right_Shoulder", "Right_Elbow"),
            "Left_Wrist": ("Left_Elbow", "Left_Wrist"),
            "Right_Wrist": ("Right_Elbow", "Right_Wrist"),
            "Left_Hand": ("Left_Wrist", "Left_Hand"),
            "Right_Hand": ("Right_Wrist", "Right_Hand"),
        }
        for key, (start_key, end_key) in segments.items():
            fallback = self.previous_quats_by_name.get(key, torso)
            quats[key] = _segment_quat(positions[start_key], positions[end_key], up_hint, fallback)
        return {key: self._quat_for(key, quats.get(key, torso)) for key in XR_BODY_JOINT_NAMES}

    def to_body_pose_dict(self, points_m: np.ndarray) -> Dict[str, Any]:
        points = self._transform_points(points_m)
        positions = self._build_positions(points)
        quats = self._build_quats(positions)
        return {
            name: [positions[name].astype(float).tolist(), quats[name].astype(float).tolist()]
            for name in XR_BODY_JOINT_NAMES
        }


def _put_latest(mp_queue: mp.Queue, item: Any) -> None:
    while True:
        try:
            mp_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                mp_queue.get_nowait()
            except queue.Empty:
                return


def _mock_qpos_from_body_pose(body_pose_dict: Dict[str, Any]) -> np.ndarray:
    qpos = DEFAULT_QPOS_G1.astype(np.float32, copy=True)
    pelvis = np.asarray(body_pose_dict.get("Pelvis", [[0.0, 0.0, 0.8], IDENTITY_QUAT_WXYZ])[0], dtype=np.float32).reshape(3)
    pelvis_quat = np.asarray(
        body_pose_dict.get("Pelvis", [[0.0, 0.0, 0.8], IDENTITY_QUAT_WXYZ])[1],
        dtype=np.float32,
    ).reshape(4)
    qpos[0:3] = pelvis
    qpos[3:7] = _normalize_quat_wxyz(pelvis_quat)
    return qpos


def _retarget_worker_main(raw_queue: mp.Queue, result_queue: mp.Queue, worker_config: Dict[str, Any]) -> None:
    if bool(worker_config.get("mock_gmr", False)):
        result_queue.put({"type": "worker_ready", "mock_gmr": True})
        while True:
            packet = raw_queue.get()
            if packet is None:
                return
            try:
                qpos = _mock_qpos_from_body_pose(packet["body_pose_dict"])
                _put_latest(
                    result_queue,
                    {
                        "type": "retarget_result",
                        "seq": int(packet["seq"]),
                        "recv_ns": int(packet["recv_ns"]),
                        "qpos": qpos,
                    },
                )
            except Exception as exc:
                _put_latest(
                    result_queue,
                    {"type": "retarget_error", "seq": int(packet.get("seq", -1)), "error": str(exc)},
                )
        return

    try:
        validate_gmr_runtime(
            Path(worker_config.get("repo_root", REPO_ROOT)),
            require_mujoco=False,
            require_patch=True,
        )
    except Exception as exc:
        result_queue.put({"type": "worker_error", "error": f"GMR runtime is not ready: {exc}"})
        return

    try:
        from general_motion_retargeting import GeneralMotionRetargeting
    except ImportError as exc:
        result_queue.put({"type": "worker_error", "error": f"Failed to import GMR: {exc}"})
        return

    try:
        retarget = GeneralMotionRetargeting(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=float(worker_config["actual_human_height"]),
        )
        retarget.max_iter = int(worker_config["gmr_max_iter"])
        result_queue.put({"type": "worker_ready"})
    except Exception as exc:
        result_queue.put({"type": "worker_error", "error": f"Failed to initialize GMR: {exc}"})
        return

    while True:
        packet = raw_queue.get()
        if packet is None:
            return
        try:
            qpos = retarget.retarget(packet["body_pose_dict"], offset_to_ground=False)
            if qpos is None:
                continue
            qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
            if qpos.shape[0] < 36:
                raise ValueError(f"GMR qpos too short: {qpos.shape[0]}")
            qpos = qpos[:36].astype(np.float32, copy=True)
            qpos[3:7] = _normalize_quat_wxyz(qpos[3:7])
            _put_latest(
                result_queue,
                {
                    "type": "retarget_result",
                    "seq": int(packet["seq"]),
                    "recv_ns": int(packet["recv_ns"]),
                    "qpos": qpos,
                },
            )
        except Exception as exc:
            _put_latest(
                result_queue,
                {"type": "retarget_error", "seq": int(packet.get("seq", -1)), "error": str(exc)},
            )


def _mujoco_viewer_worker_main(
    qpos_queue: mp.Queue,
    status_queue: mp.Queue,
    viewer_config: Dict[str, Any],
) -> None:
    try:
        validate_gmr_runtime(
            Path(viewer_config.get("repo_root", REPO_ROOT)),
            require_mujoco=True,
            require_patch=True,
        )
        from general_motion_retargeting import RobotMotionViewer

        viewer = RobotMotionViewer(
            robot_type=str(viewer_config["robot"]),
            motion_fps=float(viewer_config["fps"]),
            transparent_robot=0,
            record_video=False,
        )
        _put_latest(status_queue, {"type": "mujoco_viewer_ready"})
    except Exception as exc:
        _put_latest(status_queue, {"type": "mujoco_viewer_error", "error": str(exc)})
        return

    try:
        while True:
            try:
                qpos_item = qpos_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if qpos_item is None:
                break
            try:
                qpos = np.asarray(qpos_item, dtype=np.float32).reshape(-1)
                if qpos.shape[0] < 36:
                    raise ValueError(f"MuJoCo qpos too short: {qpos.shape[0]}")
                viewer.step(
                    root_pos=qpos[0:3],
                    root_rot=_normalize_quat_wxyz(qpos[3:7]),
                    dof_pos=qpos[7:36],
                    rate_limit=True,
                    follow_camera=True,
                )
                _put_latest(status_queue, {"type": "mujoco_viewer_frame"})
            except Exception as exc:
                _put_latest(status_queue, {"type": "mujoco_viewer_error", "error": str(exc)})
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        _put_latest(status_queue, {"type": "mujoco_viewer_closed"})


class FreeMoCapToGMRBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_event = threading.Event()
        self.raw_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.mujoco_qpos_queue: Optional[mp.Queue] = None
        self.mujoco_status_queue: Optional[mp.Queue] = None
        self.retarget_process: Optional[mp.Process] = None
        self.mujoco_process: Optional[mp.Process] = None
        self.context = None
        self.req_sock = None
        self.rep_sock = None
        self.ctrl_sock = None
        self.retarget_buffer: Deque[RetargetedFrame] = deque()
        self.retarget_buffer_lock = threading.Lock()
        self.default_qpos = DEFAULT_QPOS_G1.astype(np.float32, copy=True)
        self.latest_qpos = self.default_qpos.copy()
        self.latest_retarget_recv_ns: Optional[int] = None
        self.auto_start_acknowledged = False
        self.stats_lock = threading.Lock()
        self.stats: Dict[str, Any] = {
            "mocap_frames": 0,
            "retarget_frames": 0,
            "requests": 0,
            "replies": 0,
            "retarget_errors": 0,
            "last_valid_2d_ratio": None,
            "latest_seq": None,
            "retarget_age_ms": None,
            "mujoco_viewer_frames": 0,
            "mujoco_viewer_errors": 0,
        }
        self._logged_first_mocap_frame = False
        self._logged_calibration_ready = False
        self._logged_first_retarget_frame = False

    @staticmethod
    def _default_controller_buttons(start: bool = False) -> Dict[str, Any]:
        return {
            "left_key_one": False,
            "left_key_two": False,
            "left_axis_click": False,
            "left_index_trig": False,
            "left_grip": False,
            "left_axis": [0.0, 0.0],
            "right_key_one": bool(start),
            "right_key_two": False,
            "right_axis_click": False,
            "right_index_trig": False,
            "right_grip": False,
            "right_axis": [0.0, 0.0],
        }

    @staticmethod
    def _serialize_qpos_frame(qpos: np.ndarray) -> Dict[str, Any]:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        return {
            "root_pos": q[0:3].astype(float).tolist(),
            "root_quat": _normalize_quat_wxyz(q[3:7]).astype(float).tolist(),
            "dof_pos": q[7:36].astype(float).tolist(),
        }

    def _append_retarget_frame(self, recv_ns: int, qpos: np.ndarray) -> None:
        cutoff_ns = recv_ns - int(float(self.args.retarget_buffer_window_s) * 1e9)
        with self.retarget_buffer_lock:
            self.retarget_buffer.append(RetargetedFrame(recv_ns=recv_ns, qpos=qpos.astype(np.float32, copy=True)))
            while self.retarget_buffer and self.retarget_buffer[0].recv_ns < cutoff_ns:
                self.retarget_buffer.popleft()
            self.latest_qpos = qpos.astype(np.float32, copy=True)
            self.latest_retarget_recv_ns = recv_ns

    def _send_qpos_to_mujoco_viewer(self, qpos: np.ndarray) -> None:
        if self.mujoco_qpos_queue is None:
            return
        try:
            _put_latest(self.mujoco_qpos_queue, qpos.astype(np.float32, copy=True))
        except Exception as exc:
            with self.stats_lock:
                self.stats["mujoco_viewer_errors"] += 1
            print(f"Warning: could not queue MuJoCo viewer frame: {exc}")

    def _get_retarget_frames_snapshot(self) -> List[RetargetedFrame]:
        with self.retarget_buffer_lock:
            return list(self.retarget_buffer)

    def _sample_target_qpos(self, frames: List[RetargetedFrame], target_ns: int) -> tuple[np.ndarray, str]:
        now_ns = time.monotonic_ns()
        if frames:
            latest_age_ms = (now_ns - frames[-1].recv_ns) / 1e6
            if latest_age_ms > float(self.args.stale_timeout_ms):
                print(f"Warning: retarget data stale ({latest_age_ms:.1f}ms); using default qpos")
                return self.default_qpos.copy(), "stale_default"
        if not frames:
            return self.default_qpos.copy(), "default"
        if len(frames) == 1:
            return frames[0].qpos.astype(np.float32, copy=True), "single"
        if target_ns <= frames[0].recv_ns:
            return frames[0].qpos.astype(np.float32, copy=True), "oldest"
        if target_ns >= frames[-1].recv_ns:
            return frames[-1].qpos.astype(np.float32, copy=True), "latest"
        for index in range(1, len(frames)):
            prev_frame = frames[index - 1]
            next_frame = frames[index]
            if target_ns <= next_frame.recv_ns:
                dt = int(next_frame.recv_ns - prev_frame.recv_ns)
                if dt <= 0:
                    return next_frame.qpos.astype(np.float32, copy=True), "degenerate"
                alpha = float(target_ns - prev_frame.recv_ns) / float(dt)
                return _interpolate_qpos(prev_frame.qpos, next_frame.qpos, alpha), "interpolate"
        return frames[-1].qpos.astype(np.float32, copy=True), "latest"

    def _build_reply_frames(self, req: Dict[str, Any], req_recv_ns: int) -> tuple[List[np.ndarray], str]:
        frames = self._get_retarget_frames_snapshot()
        need_frames = int(max(1, min(int(req.get("need_frames", 1)), int(self.args.max_reply_frames))))
        step_ns = int(1e9 / float(self.args.reply_fps))
        target_base_ns = req_recv_ns - int(float(self.args.lookback_ms) * 1e6)
        qpos_frames: List[np.ndarray] = []
        modes: List[str] = []
        for frame_index in range(need_frames):
            qpos, mode = self._sample_target_qpos(frames, target_base_ns + frame_index * step_ns)
            qpos_frames.append(qpos)
            modes.append(mode)
        return qpos_frames, ",".join(sorted(set(modes)))

    def _mocap_loop(self) -> None:
        assert self.raw_queue is not None
        calibration_frames: List[np.ndarray] = []
        converter: Optional[FreeMoCapXRobotConverter] = None
        try:
            print("FreeMoCap mocap source starting")
            loop_count = 0
            while not self.stop_event.is_set():
                if self.args.source == "recording":
                    mocap_frames = iter_mocap_3d_frames(
                        recording_folder=self.args.recording_folder,
                        calibration_toml=self.args.calibration_toml,
                        max_frames=self.args.max_frames,
                        model_complexity=self.args.model_complexity,
                        tracker=self.args.tracker,
                        static_image_mode=self.args.static_image_mode,
                        parallel_camera_tracking=self.args.parallel_camera_tracking,
                        camera_workers=self.args.camera_workers,
                        resize_width=self.args.resize_width,
                        include_holistic=False,
                        prewarm=not self.args.skip_triangulation_prewarm,
                    )
                elif self.args.source == "skellycam":
                    mocap_frames = iter_skellycam_mocap_3d_frames(
                        calibration_toml=self.args.calibration_toml,
                        camera_ids=self.args.camera_ids,
                        camera_config_json=self.args.camera_config_json,
                        skellycam_home=self.args.skellycam_home,
                        max_frames=self.args.max_frames,
                        model_complexity=self.args.model_complexity,
                        tracker=self.args.tracker,
                        static_image_mode=self.args.static_image_mode,
                        parallel_camera_tracking=self.args.parallel_camera_tracking,
                        camera_workers=self.args.camera_workers,
                        resize_width=self.args.resize_width,
                        include_holistic=False,
                        prewarm=not self.args.skip_triangulation_prewarm,
                        max_camera_skew_ms=self.args.max_camera_skew_ms,
                    )
                else:
                    raise ValueError(f"Unsupported source: {self.args.source}")

                for mocap_frame in mocap_frames:
                    if self.stop_event.is_set():
                        break
                    if not self._logged_first_mocap_frame:
                        print(
                            f"FreeMoCap first 3D frame received from {self.args.source}: seq={mocap_frame.seq}, "
                            f"valid_2d={mocap_frame.valid_2d_point_ratio * 100.0:.1f}%"
                        )
                        self._logged_first_mocap_frame = True
                    points_m = np.asarray(mocap_frame.points_3d, dtype=np.float32).reshape(33, 3) * 0.001
                    if converter is None:
                        if FreeMoCapXRobotConverter._frame_has_torso(points_m):
                            calibration_frames.append(points_m)
                        if len(calibration_frames) >= max(1, int(self.args.calibration_frames)):
                            converter = FreeMoCapXRobotConverter.from_calibration_frames(calibration_frames)
                            print(f"FreeMoCap calibration ready with {len(calibration_frames)} frames")
                            self._logged_calibration_ready = True
                        continue

                    body_pose_dict = converter.to_body_pose_dict(points_m)
                    _put_latest(
                        self.raw_queue,
                        {
                            "seq": int(mocap_frame.seq + loop_count * 1000000),
                            "recv_ns": int(mocap_frame.timestamp_ns),
                            "body_pose_dict": body_pose_dict,
                        },
                    )
                    with self.stats_lock:
                        self.stats["mocap_frames"] += 1
                        self.stats["last_valid_2d_ratio"] = mocap_frame.valid_2d_point_ratio
                        self.stats["latest_seq"] = int(mocap_frame.seq + loop_count * 1000000)
                if self.args.source != "recording" or not self.args.loop_source or self.stop_event.is_set():
                    break
                loop_count += 1
                print(f"FreeMoCap mocap source looping recording, pass={loop_count}")
            print("FreeMoCap mocap source finished")
        except Exception as exc:
            print(f"FreeMoCap mocap source failed: {exc}")
            self.stop_event.set()

    def _result_loop(self) -> None:
        assert self.result_queue is not None
        while not self.stop_event.is_set():
            try:
                payload = self.result_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if payload_type == "retarget_result":
                qpos = np.asarray(payload["qpos"], dtype=np.float32).reshape(-1)
                self._append_retarget_frame(int(payload["recv_ns"]), qpos)
                self._send_qpos_to_mujoco_viewer(qpos)
                with self.stats_lock:
                    self.stats["retarget_frames"] += 1
                if not self._logged_first_retarget_frame:
                    print(
                        f"GMR first retargeted frame ready: seq={payload.get('seq')}, "
                        f"root_pos={qpos[0:3].tolist()}"
                    )
                    self._logged_first_retarget_frame = True
            elif payload_type == "retarget_error":
                with self.stats_lock:
                    self.stats["retarget_errors"] += 1
                print(f"Warning: GMR retarget failed: {payload.get('error')}")
            elif payload_type == "worker_error":
                print(str(payload.get("error", "GMR worker error")))
                self.stop_event.set()

    def _drain_requests_blocking(self) -> tuple[Optional[Dict[str, Any]], Optional[int], int]:
        import zmq

        assert self.req_sock is not None
        poller = zmq.Poller()
        poller.register(self.req_sock, zmq.POLLIN)
        events = dict(poller.poll(100))
        if self.req_sock not in events:
            return None, None, 0

        latest_req: Optional[Dict[str, Any]] = None
        req_recv_ns: Optional[int] = None
        merged_reqs = 0
        any_start = False
        while True:
            try:
                raw = self.req_sock.recv_string(flags=zmq.NOBLOCK)
                req_recv_ns = time.monotonic_ns()
            except zmq.Again:
                break
            try:
                req = json.loads(raw)
            except Exception:
                print("Warning: bad request JSON")
                continue
            if not isinstance(req, dict):
                continue
            any_start = any_start or bool(req.get("start", False))
            latest_req = req
            merged_reqs += 1

        if latest_req is not None:
            latest_req["start"] = any_start
        return latest_req, req_recv_ns, merged_reqs

    def _request_loop(self) -> None:
        import zmq

        assert self.rep_sock is not None
        while not self.stop_event.is_set():
            req, req_recv_ns, merged_reqs = self._drain_requests_blocking()
            if req is None or req_recv_ns is None:
                continue
            start_flag = bool(req.get("start", False))
            if start_flag:
                self.auto_start_acknowledged = True
            out_frames, mode = self._build_reply_frames(req, req_recv_ns)
            retarget_age_ms = None
            with self.retarget_buffer_lock:
                if self.latest_retarget_recv_ns is not None:
                    retarget_age_ms = int((time.monotonic_ns() - self.latest_retarget_recv_ns) / 1e6)
            payload = {
                "start": start_flag,
                "no_interp_applied": "interpolate" not in mode,
                "chunk_size": len(out_frames),
                "mode": mode,
                "merged_reqs": int(merged_reqs),
                "retarget_age_ms": retarget_age_ms,
                "t_rep_ms": int(time.time() * 1000),
                "frames": [self._serialize_qpos_frame(qpos) for qpos in out_frames],
            }
            try:
                self.rep_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
                with self.stats_lock:
                    self.stats["requests"] += int(merged_reqs)
                    self.stats["replies"] += 1
                    self.stats["retarget_age_ms"] = retarget_age_ms
            except zmq.Again:
                print("Warning: reply queue full; dropped reply")
            except Exception as exc:
                print(f"Warning: reply send failed: {exc}")

    def _control_loop(self) -> None:
        import zmq

        assert self.ctrl_sock is not None
        period_s = 1.0 / float(self.args.ctrl_fps)
        pulse_period_s = 1.0
        pulse_width_s = 0.15
        start_time = time.monotonic()
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - start_time
            start_pulse = (
                bool(self.args.auto_start)
                and not self.auto_start_acknowledged
                and (elapsed % pulse_period_s) < pulse_width_s
            )
            payload = {
                "t_ms": int(time.time() * 1000),
                "controller_buttons": self._default_controller_buttons(start=start_pulse),
            }
            try:
                self.ctrl_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"Warning: control send failed: {exc}")
            self.stop_event.wait(timeout=period_s)

    def _mujoco_status_loop(self) -> None:
        if self.mujoco_status_queue is None:
            return
        while not self.stop_event.is_set():
            try:
                payload = self.mujoco_status_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if payload_type == "mujoco_viewer_ready":
                print("MuJoCo viewer ready")
            elif payload_type == "mujoco_viewer_frame":
                with self.stats_lock:
                    self.stats["mujoco_viewer_frames"] += 1
            elif payload_type == "mujoco_viewer_error":
                with self.stats_lock:
                    self.stats["mujoco_viewer_errors"] += 1
                print(f"Warning: MuJoCo viewer failed: {payload.get('error')}")
            elif payload_type == "mujoco_viewer_closed":
                print("MuJoCo viewer closed")

    def _stats_loop(self) -> None:
        if float(self.args.log_interval_s) <= 0.0:
            return
        last = time.monotonic()
        last_mocap = 0
        last_retarget = 0
        while not self.stop_event.is_set():
            if self.stop_event.wait(timeout=float(self.args.log_interval_s)):
                break
            now = time.monotonic()
            with self.stats_lock:
                stats = dict(self.stats)
            dt = max(now - last, 1e-6)
            mocap_fps = (int(stats["mocap_frames"]) - last_mocap) / dt
            gmr_fps = (int(stats["retarget_frames"]) - last_retarget) / dt
            last = now
            last_mocap = int(stats["mocap_frames"])
            last_retarget = int(stats["retarget_frames"])
            valid = stats.get("last_valid_2d_ratio")
            valid_msg = "None" if valid is None else f"{float(valid) * 100.0:.1f}%"
            print(
                "[BridgeStats] "
                f"mocap_fps={mocap_fps:.2f}, gmr_fps={gmr_fps:.2f}, "
                f"mocap_frames={stats['mocap_frames']}, retarget_frames={stats['retarget_frames']}, "
                f"requests={stats['requests']}, replies={stats['replies']}, "
                f"errors={stats['retarget_errors']}, retarget_age_ms={stats['retarget_age_ms']}, "
                f"viewer_frames={stats['mujoco_viewer_frames']}, viewer_errors={stats['mujoco_viewer_errors']}, "
                f"valid_2d={valid_msg}, latest_seq={stats['latest_seq']}"
            )

    def setup(self) -> None:
        import zmq

        mp_context = mp.get_context("spawn")
        self.raw_queue = mp_context.Queue(maxsize=1)
        self.result_queue = mp_context.Queue(maxsize=8)
        self.retarget_process = mp_context.Process(
            target=_retarget_worker_main,
            args=(
                self.raw_queue,
                self.result_queue,
                {
                    "repo_root": str(REPO_ROOT),
                    "actual_human_height": float(self.args.actual_human_height),
                    "gmr_max_iter": int(self.args.gmr_max_iter),
                    "mock_gmr": bool(self.args.mock_gmr),
                },
            ),
            name="freemocap-gmr-retarget",
            daemon=True,
        )
        self.retarget_process.start()

        try:
            worker_msg = self.result_queue.get(timeout=float(self.args.worker_start_timeout_s))
        except queue.Empty as exc:
            raise RuntimeError("GMR retarget worker did not become ready in time.") from exc
        if not isinstance(worker_msg, dict) or worker_msg.get("type") != "worker_ready":
            raise RuntimeError(f"GMR retarget worker failed to start: {worker_msg}")

        if bool(self.args.mujoco_viewer):
            self.mujoco_qpos_queue = mp_context.Queue(maxsize=1)
            self.mujoco_status_queue = mp_context.Queue(maxsize=8)
            self.mujoco_process = mp_context.Process(
                target=_mujoco_viewer_worker_main,
                args=(
                    self.mujoco_qpos_queue,
                    self.mujoco_status_queue,
                    {
                        "repo_root": str(REPO_ROOT),
                        "robot": str(self.args.mujoco_robot),
                        "fps": float(self.args.mujoco_fps),
                    },
                ),
                name="freemocap-mujoco-viewer",
                daemon=True,
            )
            self.mujoco_process.start()

        self.context = zmq.Context.instance()
        self.req_sock = self.context.socket(zmq.PULL)
        self.req_sock.setsockopt(zmq.LINGER, 0)
        self.req_sock.setsockopt(zmq.RCVHWM, 500)
        self.req_sock.bind(self.args.req_bind_addr)

        self.rep_sock = self.context.socket(zmq.PUSH)
        self.rep_sock.setsockopt(zmq.LINGER, 0)
        self.rep_sock.setsockopt(zmq.SNDHWM, 500)
        self.rep_sock.bind(self.args.rep_bind_addr)

        self.ctrl_sock = self.context.socket(zmq.PUSH)
        self.ctrl_sock.setsockopt(zmq.LINGER, 0)
        self.ctrl_sock.setsockopt(zmq.SNDHWM, 500)
        self.ctrl_sock.bind(self.args.ctrl_bind_addr)

        print("FreeMoCap -> GMR bridge initialized")
        print(f"  req_bind_addr: {self.args.req_bind_addr}")
        print(f"  rep_bind_addr: {self.args.rep_bind_addr}")
        print(f"  ctrl_bind_addr: {self.args.ctrl_bind_addr}")
        print(f"  tracker: {self.args.tracker}")
        print(f"  model_complexity: {self.args.model_complexity}")
        print(f"  parallel_camera_tracking: {self.args.parallel_camera_tracking}")
        print(f"  actual_human_height: {self.args.actual_human_height}")
        print(f"  gmr_max_iter: {self.args.gmr_max_iter}")
        print(f"  mock_gmr: {self.args.mock_gmr}")
        print(f"  mujoco_viewer: {self.args.mujoco_viewer}")
        print(f"  mujoco_robot: {self.args.mujoco_robot}")
        print(f"  mujoco_fps: {self.args.mujoco_fps}")
        print(f"  dataset_joint_names: {len(DATASET_JOINT_NAMES_29)} joints")

    def run(self) -> None:
        threads = [
            threading.Thread(target=self._mocap_loop, name="freemocap-mocap", daemon=True),
            threading.Thread(target=self._result_loop, name="freemocap-retarget-results", daemon=True),
            threading.Thread(target=self._request_loop, name="freemocap-zmq-requests", daemon=True),
            threading.Thread(target=self._control_loop, name="freemocap-zmq-control", daemon=True),
            threading.Thread(target=self._mujoco_status_loop, name="freemocap-mujoco-status", daemon=True),
            threading.Thread(target=self._stats_loop, name="freemocap-stats", daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Stopping bridge...")
            self.stop_event.set()
        finally:
            self.stop_event.set()
            for thread in threads:
                thread.join(timeout=2.0)
            self.close()

    def close(self) -> None:
        if self.raw_queue is not None:
            try:
                self.raw_queue.put_nowait(None)
            except Exception:
                pass
        if self.mujoco_qpos_queue is not None:
            try:
                _put_latest(self.mujoco_qpos_queue, None)
            except Exception:
                pass
        if self.retarget_process is not None:
            self.retarget_process.join(timeout=2.0)
            if self.retarget_process.is_alive():
                self.retarget_process.terminate()
                self.retarget_process.join(timeout=1.0)
        if self.mujoco_process is not None:
            self.mujoco_process.join(timeout=2.0)
            if self.mujoco_process.is_alive():
                self.mujoco_process.terminate()
                self.mujoco_process.join(timeout=1.0)
        for sock in (self.req_sock, self.rep_sock, self.ctrl_sock):
            if sock is not None:
                sock.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FreeMoCap pose-only mocap to GMR/sim2real ZMQ bridge.")
    parser.add_argument("--source", choices=["recording", "skellycam"], default="recording")
    parser.add_argument("--recording-folder", type=Path, default=None)
    parser.add_argument("--calibration-toml", type=Path, default=None)
    parser.add_argument("--camera-ids", type=str, default=None)
    parser.add_argument("--camera-config-json", type=Path, default=None)
    parser.add_argument("--skellycam-home", type=Path, default=None)
    parser.add_argument("--max-camera-skew-ms", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--loop-source", action="store_true")
    parser.add_argument("--tracker", choices=["pose", "holistic"], default="pose")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--static-image-mode", action="store_true")
    parser.add_argument("--parallel-camera-tracking", action="store_true")
    parser.add_argument("--camera-workers", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--skip-triangulation-prewarm", action="store_true")
    parser.add_argument("--calibration-frames", type=int, default=30)
    parser.add_argument("--actual-human-height", type=float, default=1.6)
    parser.add_argument("--gmr-max-iter", type=int, default=5)
    parser.add_argument("--mock-gmr", action="store_true")
    parser.add_argument("--mujoco-viewer", action="store_true")
    parser.add_argument("--mujoco-fps", type=float, default=30.0)
    parser.add_argument("--mujoco-robot", type=str, default="unitree_g1")
    parser.add_argument("--req-bind-addr", type=str, default="tcp://*:28701")
    parser.add_argument("--rep-bind-addr", type=str, default="tcp://*:28702")
    parser.add_argument("--ctrl-bind-addr", type=str, default="tcp://*:28703")
    parser.add_argument("--auto-start", action="store_true", default=True)
    parser.add_argument("--no-auto-start", action="store_false", dest="auto_start")
    parser.add_argument("--ctrl-fps", type=float, default=50.0)
    parser.add_argument("--reply-fps", type=float, default=50.0)
    parser.add_argument("--max-reply-frames", type=int, default=5)
    parser.add_argument("--lookback-ms", type=float, default=25.0)
    parser.add_argument("--retarget-buffer-window-s", type=float, default=0.5)
    parser.add_argument("--stale-timeout-ms", type=float, default=300.0)
    parser.add_argument("--log-interval-s", type=float, default=1.0)
    parser.add_argument("--worker-start-timeout-s", type=float, default=15.0)
    args = parser.parse_args()

    if args.source == "recording" and args.recording_folder is None:
        raise ValueError("--recording-folder is required when --source recording")
    if args.source == "skellycam" and args.calibration_toml is None:
        raise ValueError("--calibration-toml is required when --source skellycam")
    if args.source == "skellycam" and args.loop_source:
        raise ValueError("--loop-source only applies to --source recording")
    if args.max_camera_skew_ms <= 0:
        raise ValueError("--max-camera-skew-ms must be positive")
    if args.calibration_frames <= 0:
        raise ValueError("--calibration-frames must be positive")
    if args.ctrl_fps <= 0:
        raise ValueError("--ctrl-fps must be positive")
    if args.reply_fps <= 0:
        raise ValueError("--reply-fps must be positive")
    if args.mujoco_fps <= 0:
        raise ValueError("--mujoco-fps must be positive")
    if args.max_reply_frames <= 0:
        raise ValueError("--max-reply-frames must be positive")
    if args.retarget_buffer_window_s <= 0:
        raise ValueError("--retarget-buffer-window-s must be positive")
    if args.stale_timeout_ms <= 0:
        raise ValueError("--stale-timeout-ms must be positive")
    return args


def main() -> None:
    args = parse_args()
    bridge = FreeMoCapToGMRBridge(args)
    bridge.setup()
    bridge.run()


if __name__ == "__main__":
    main()
