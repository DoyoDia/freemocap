from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest


def _import_camera_group():
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

    return CameraGroup


CameraGroup = _import_camera_group()


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
