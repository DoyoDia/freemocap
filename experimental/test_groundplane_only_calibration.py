from __future__ import annotations

import sys
import logging
import os
import shutil
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _import_groundplane_module():
    root_logger = logging.getLogger()
    pytest_handlers = list(root_logger.handlers)
    env_names = ("USERPROFILE", "YOLO_CONFIG_DIR", "MPLCONFIGDIR")
    previous_env = {name: os.environ.get(name) for name in env_names}
    workspace_root = Path(__file__).resolve().parents[1]
    test_home_path = workspace_root / ".tmp_freemocap_test_home_groundplane"
    shutil.rmtree(test_home_path, ignore_errors=True)
    (test_home_path / "freemocap_data").mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(test_home_path)
    os.environ["YOLO_CONFIG_DIR"] = str(test_home_path / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(test_home_path / "matplotlib")
    for handler in pytest_handlers:
        root_logger.removeHandler(handler)
    try:
        import groundplane_only_calibration as groundplane
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

    return groundplane


groundplane = _import_groundplane_module()


class _FakeGroundplaneSuccess:
    success = True
    error = None


class _FakeCameraGroup:
    def __init__(self, camera_count: int = 2) -> None:
        self.cameras = [object() for _ in range(camera_count)]
        self.metadata = {}
        self.dumped_paths: list[Path] = []

    def get_rows_videos(self, videos, board, verbose=True):
        corners = np.zeros((8, 1, 2), dtype=np.float32)
        return [
            [{"framenum": (0, 0), "corners": corners}],
            [{"framenum": (0, 0), "corners": corners}],
        ]

    def _validate_charuco_rows_for_calibration(self, all_rows):
        assert len(all_rows) == 2

    def dump(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake calibration", encoding="utf-8")
        self.dumped_paths.append(path)


class _FakeCalibrator:
    def __init__(
        self,
        *,
        charuco_board_object,
        charuco_square_size,
        calibration_videos_folder_path,
        progress_callback,
    ) -> None:
        self._anipose_charuco_board = object()
        self._recording_folder_path = Path(calibration_videos_folder_path).parent

    def set_charuco_board_as_groundplane(self, cam_group):
        return cam_group, _FakeGroundplaneSuccess()

    def get_real_world_matrices(self, cam_group):
        return [], []


def test_default_groundplane_toml_path_uses_sibling_suffix() -> None:
    path = Path("calibration.toml")

    assert groundplane.default_groundplane_toml_path(path) == Path("calibration_groundplane.toml")


def test_apply_groundplane_writes_output_toml(monkeypatch, tmp_path) -> None:
    calibration_toml = tmp_path / "camera_calibration.toml"
    calibration_toml.write_text("fake input", encoding="utf-8")
    videos_folder = tmp_path / "synchronized_videos"
    videos_folder.mkdir()
    video_paths = [videos_folder / "Camera_000.mp4", videos_folder / "Camera_001.mp4"]
    for path in video_paths:
        path.write_bytes(b"fake")
    fake_cam_group = _FakeCameraGroup(camera_count=2)

    monkeypatch.setattr(
        groundplane.freemocap_anipose.CameraGroup,
        "load",
        staticmethod(lambda path: fake_cam_group),
    )
    monkeypatch.setattr(groundplane, "AniposeCameraCalibrator", _FakeCalibrator)
    monkeypatch.setattr(groundplane, "get_video_paths", lambda path_to_video_folder: video_paths)

    result = groundplane.apply_groundplane_to_calibration_toml(
        calibration_toml=calibration_toml,
        calibration_videos_folder=videos_folder,
        charuco_square_size=39.0,
        charuco_board_name="7x5 Charuco",
    )

    assert result.output_toml_path.exists()
    assert result.output_toml_path.name == "camera_calibration_groundplane.toml"
    assert fake_cam_group.metadata["groundplane_calibration"] is True
    assert fake_cam_group.metadata["groundplane_source_toml"] == str(calibration_toml.resolve())
