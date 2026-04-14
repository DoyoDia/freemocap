from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest


def _import_calibration_objects():
    root_logger = logging.getLogger()
    pytest_handlers = list(root_logger.handlers)
    env_names = ("USERPROFILE", "YOLO_CONFIG_DIR", "MPLCONFIGDIR")
    previous_env = {name: os.environ.get(name) for name in env_names}
    workspace_root = Path(__file__).resolve().parents[1]
    test_home_path = workspace_root / ".tmp_freemocap_test_home"
    shutil.rmtree(test_home_path, ignore_errors=True)
    (test_home_path / "freemocap_data").mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(test_home_path)
    os.environ["YOLO_CONFIG_DIR"] = str(test_home_path / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(test_home_path / "matplotlib")
    for handler in pytest_handlers:
        root_logger.removeHandler(handler)
    try:
        from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.freemocap_anipose import (
            AniposeCharucoBoard,
            CameraGroup,
        )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
        for handler in pytest_handlers:
            root_logger.addHandler(handler)
        shutil.rmtree(test_home_path, ignore_errors=True)

    return CameraGroup, AniposeCharucoBoard


CameraGroup, AniposeCharucoBoard = _import_calibration_objects()


class _FakeCamera:
    def __init__(self, camera_matrix: np.ndarray, distortions: np.ndarray):
        self._camera_matrix = camera_matrix
        self._distortions = distortions

    def get_camera_matrix(self) -> np.ndarray:
        return self._camera_matrix

    def get_distortions(self) -> np.ndarray:
        return self._distortions


def _charuco_row(frame_number: int, corner_count: int) -> dict:
    return {
        "framenum": (0, frame_number),
        "corners": np.zeros((corner_count, 2), dtype=np.float32),
    }


def test_validate_charuco_rows_rejects_camera_with_no_usable_frames() -> None:
    all_rows = [
        [_charuco_row(frame_number=1, corner_count=8)],
        [],
    ]

    with pytest.raises(ValueError, match=r"Cameras without usable frames: \[1\]"):
        CameraGroup._validate_charuco_rows_for_calibration(None, all_rows)


def test_validate_charuco_rows_rejects_no_shared_usable_frames() -> None:
    all_rows = [
        [_charuco_row(frame_number=1, corner_count=8)],
        [_charuco_row(frame_number=2, corner_count=8)],
    ]

    with pytest.raises(ValueError, match="Shared usable frame count: 0"):
        CameraGroup._validate_charuco_rows_for_calibration(None, all_rows)


def test_validate_charuco_rows_allows_shared_usable_frames() -> None:
    all_rows = [
        [_charuco_row(frame_number=1, corner_count=8)],
        [_charuco_row(frame_number=1, corner_count=8)],
    ]

    CameraGroup._validate_charuco_rows_for_calibration(None, all_rows)


def test_charuco_pose_estimation_works_without_removed_opencv_api(monkeypatch) -> None:
    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        monkeypatch.delattr(cv2.aruco, "estimatePoseCharucoBoard")

    board = AniposeCharucoBoard(
        squaresX=7,
        squaresY=5,
        square_length=39.0,
        marker_length=31.2,
        marker_bits=4,
        dict_size=250,
    )
    camera_matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortions = np.zeros(5, dtype=np.float64)
    camera = _FakeCamera(camera_matrix=camera_matrix, distortions=distortions)
    ids = np.arange(8, dtype=np.int32).reshape(-1, 1)
    object_points = board.objPoints[ids.reshape(-1)].astype(np.float64)
    expected_rvec = np.array([[0.1], [0.2], [0.05]], dtype=np.float64)
    expected_tvec = np.array([[10.0], [20.0], [1000.0]], dtype=np.float64)
    corners, _ = cv2.projectPoints(object_points, expected_rvec, expected_tvec, camera_matrix, distortions)

    rvec, tvec = board.estimate_pose_points(camera=camera, corners=corners, ids=ids)

    assert rvec is not None
    assert tvec is not None
    assert rvec.shape == (3, 1)
    assert tvec.shape == (3, 1)
