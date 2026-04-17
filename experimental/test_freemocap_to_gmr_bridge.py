from __future__ import annotations

import sys
import threading
import queue
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import freemocap_to_gmr_bridge as bridge
import gmr_runtime


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


def test_put_latest_drops_stale_queue_items() -> None:
    latest_queue: queue.Queue = queue.Queue(maxsize=1)

    bridge._put_latest(latest_queue, "old")
    bridge._put_latest(latest_queue, "new")

    assert latest_queue.get_nowait() == "new"
    assert latest_queue.empty()


def test_gmr_runtime_patch_state_detects_missing_and_patched(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gmr_runtime, "_scipy_supports_scalar_first", lambda: False)
    motion_retarget = tmp_path / "external" / "GMR" / "general_motion_retargeting" / "motion_retarget.py"
    neck_retarget = tmp_path / "external" / "GMR" / "general_motion_retargeting" / "neck_retarget.py"
    motion_retarget.parent.mkdir(parents=True)
    motion_retarget.write_text("R.from_quat(quat, scalar_first=True)\n", encoding="utf-8")
    neck_retarget.write_text("R.from_quat(quat, scalar_first=True)\n", encoding="utf-8")

    assert gmr_runtime.gmr_scipy_patch_state(tmp_path) == "missing"

    motion_retarget.write_text("def _rotation_from_quat_wxyz(quat):\n    return quat\n", encoding="utf-8")
    neck_retarget.write_text("def _rotation_from_quat_wxyz(quat):\n    return quat\n", encoding="utf-8")

    assert gmr_runtime.gmr_scipy_patch_state(tmp_path) == "patched"


def test_validate_gmr_runtime_import_error_mentions_install_command(monkeypatch, tmp_path: Path) -> None:
    gmr_root = tmp_path / "external" / "GMR"
    unitree_xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    unitree_xml.parent.mkdir(parents=True)
    unitree_xml.write_text("<mujoco/>", encoding="utf-8")

    import pytest

    original_import_module = gmr_runtime.importlib.import_module

    def _fake_import_module(name: str):
        if name == "general_motion_retargeting":
            raise ImportError("boom")
        return original_import_module(name)

    monkeypatch.setattr(gmr_runtime.importlib, "import_module", _fake_import_module)
    with pytest.raises(RuntimeError, match="python -m pip install -e external/GMR"):
        gmr_runtime.validate_gmr_runtime(tmp_path, require_mujoco=False, require_patch=False)


def test_validate_gmr_runtime_can_skip_gmr_import(monkeypatch, tmp_path: Path) -> None:
    gmr_root = tmp_path / "external" / "GMR"
    unitree_xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    unitree_xml.parent.mkdir(parents=True)
    unitree_xml.write_text("<mujoco/>", encoding="utf-8")

    def _raise_if_imported(name: str):
        if name == "general_motion_retargeting":
            raise AssertionError("GUI preflight should not import GMR")
        raise ImportError(name)

    monkeypatch.setattr(gmr_runtime.importlib, "import_module", _raise_if_imported)

    status = gmr_runtime.validate_gmr_runtime(
        tmp_path,
        require_import=False,
        require_mujoco=False,
        require_patch=False,
    )

    assert status.unitree_g1_xml == unitree_xml


def test_gmr_runtime_can_load_unitree_g1_mujoco_model() -> None:
    import pytest

    try:
        status = gmr_runtime.validate_gmr_runtime(bridge.REPO_ROOT, require_mujoco=True)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(status.unitree_g1_xml))
    data = mj.MjData(model)
    data.qpos[:36] = bridge.DEFAULT_QPOS_G1
    mj.mj_forward(model, data)

    assert model.nq >= 36
