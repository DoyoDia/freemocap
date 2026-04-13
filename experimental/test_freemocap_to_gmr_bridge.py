from __future__ import annotations

import sys
import threading
import queue
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import freemocap_to_gmr_bridge as bridge


def _synthetic_mediapipe_points() -> np.ndarray:
    points = np.zeros((33, 3), dtype=float)
    points[bridge.MEDIAPIPE["left_hip"]] = [-0.15, 0.0, 0.0]
    points[bridge.MEDIAPIPE["right_hip"]] = [0.15, 0.0, 0.0]
    points[bridge.MEDIAPIPE["left_shoulder"]] = [-0.18, 0.0, 0.5]
    points[bridge.MEDIAPIPE["right_shoulder"]] = [0.18, 0.0, 0.5]
    points[bridge.MEDIAPIPE["left_knee"]] = [-0.15, 0.0, -0.45]
    points[bridge.MEDIAPIPE["right_knee"]] = [0.15, 0.0, -0.45]
    points[bridge.MEDIAPIPE["left_ankle"]] = [-0.15, 0.0, -0.9]
    points[bridge.MEDIAPIPE["right_ankle"]] = [0.15, 0.0, -0.9]
    points[bridge.MEDIAPIPE["left_foot_index"]] = [-0.15, 0.15, -0.95]
    points[bridge.MEDIAPIPE["right_foot_index"]] = [0.15, 0.15, -0.95]
    points[bridge.MEDIAPIPE["left_elbow"]] = [-0.45, 0.0, 0.3]
    points[bridge.MEDIAPIPE["right_elbow"]] = [0.45, 0.0, 0.3]
    points[bridge.MEDIAPIPE["left_wrist"]] = [-0.65, 0.0, 0.1]
    points[bridge.MEDIAPIPE["right_wrist"]] = [0.65, 0.0, 0.1]
    points[bridge.MEDIAPIPE["left_index"]] = [-0.7, 0.0, 0.05]
    points[bridge.MEDIAPIPE["right_index"]] = [0.7, 0.0, 0.05]
    points[bridge.MEDIAPIPE["left_ear"]] = [-0.08, 0.0, 0.75]
    points[bridge.MEDIAPIPE["right_ear"]] = [0.08, 0.0, 0.75]
    points[bridge.MEDIAPIPE["nose"]] = [0.0, 0.08, 0.73]
    return points


def test_freemocap_to_xrobot_converter_shapes_and_quaternions() -> None:
    points = _synthetic_mediapipe_points()
    converter = bridge.FreeMoCapXRobotConverter.from_calibration_frames([points])

    pose_dict = converter.to_body_pose_dict(points)

    assert set(pose_dict) == set(bridge.XR_BODY_JOINT_NAMES)
    for pos, quat in pose_dict.values():
        assert len(pos) == 3
        assert len(quat) == 4
        assert np.isfinite(pos).all()
        assert np.isfinite(quat).all()
        assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-4)


def test_qpos_serialization_matches_sim2real_frame_shape() -> None:
    frame = bridge.FreeMoCapToGMRBridge._serialize_qpos_frame(bridge.DEFAULT_QPOS_G1)

    assert len(frame["root_pos"]) == 3
    assert len(frame["root_quat"]) == 4
    assert len(frame["dof_pos"]) == 29
    assert np.isclose(np.linalg.norm(frame["root_quat"]), 1.0, atol=1e-4)


def test_reply_payload_is_json_serializable() -> None:
    payload = {
        "start": True,
        "frames": [bridge.FreeMoCapToGMRBridge._serialize_qpos_frame(bridge.DEFAULT_QPOS_G1)],
    }

    import json

    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded["start"] is True
    assert len(loaded["frames"][0]["dof_pos"]) == len(bridge.DATASET_JOINT_NAMES_29)


def test_mock_qpos_from_body_pose_shape() -> None:
    points = _synthetic_mediapipe_points()
    converter = bridge.FreeMoCapXRobotConverter.from_calibration_frames([points])
    pose_dict = converter.to_body_pose_dict(points)

    qpos = bridge._mock_qpos_from_body_pose(pose_dict)
    assert qpos.shape == (36,)
    assert np.isclose(np.linalg.norm(qpos[3:7]), 1.0, atol=1e-4)


def test_mock_gmr_worker_emits_retarget_result() -> None:
    points = _synthetic_mediapipe_points()
    converter = bridge.FreeMoCapXRobotConverter.from_calibration_frames([points])
    pose_dict = converter.to_body_pose_dict(points)

    raw_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=bridge._retarget_worker_main,
        args=(raw_queue, result_queue, {"actual_human_height": 1.6, "gmr_max_iter": 5, "mock_gmr": True}),
        daemon=True,
    )
    worker.start()
    try:
        ready = result_queue.get(timeout=5.0)
        assert ready["type"] == "worker_ready"
        raw_queue.put({"seq": 1, "recv_ns": 123, "body_pose_dict": pose_dict})
        result = result_queue.get(timeout=5.0)
        assert result["type"] == "retarget_result"
        assert np.asarray(result["qpos"], dtype=np.float32).shape == (36,)
    finally:
        raw_queue.put(None)
        worker.join(timeout=2.0)
