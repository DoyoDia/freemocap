"""Apply a new Charuco groundplane to an existing camera calibration TOML."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_writable_freemocap_home() -> None:
    runtime_home = REPO_ROOT / ".venv" / ".groundplane_only_home"
    yolo_config_dir = runtime_home / "ultralytics"
    (yolo_config_dir / "Ultralytics").mkdir(parents=True, exist_ok=True)
    (runtime_home / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    os.environ["MPLCONFIGDIR"] = str(runtime_home / "matplotlib")
    try:
        (Path.home() / "freemocap_data").mkdir(parents=True, exist_ok=True)
        return
    except OSError:
        runtime_home.mkdir(parents=True, exist_ok=True)
        os.environ["USERPROFILE"] = str(runtime_home)


_ensure_writable_freemocap_home()

from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration import (
    freemocap_anipose,
)
from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.anipose_camera_calibrator import (
    AniposeCameraCalibrator,
)
from freemocap.core_processes.capture_volume_calibration.charuco_stuff.charuco_board_definition import (
    CHARUCO_BOARDS,
)
from freemocap.system.paths_and_filenames.path_getters import get_last_successful_calibration_toml_path
from freemocap.utilities.get_video_paths import get_video_paths


@dataclass
class GroundplaneOnlyCalibrationResult:
    output_toml_path: Path
    charuco_3d_path: Path


def default_groundplane_toml_path(calibration_toml: Path) -> Path:
    calibration_toml = Path(calibration_toml)
    return calibration_toml.with_name(f"{calibration_toml.stem}_groundplane.toml")


def apply_groundplane_to_calibration_toml(
    *,
    calibration_toml: Path,
    calibration_videos_folder: Path,
    charuco_square_size: float,
    charuco_board_name: str,
    output_toml: Optional[Path] = None,
    update_last_successful: bool = False,
    progress_callback: Callable[[str], None] = lambda _: None,
) -> GroundplaneOnlyCalibrationResult:
    calibration_toml = Path(calibration_toml).expanduser().resolve()
    calibration_videos_folder = Path(calibration_videos_folder).expanduser().resolve()
    output_toml = (
        Path(output_toml).expanduser().resolve()
        if output_toml is not None
        else default_groundplane_toml_path(calibration_toml).resolve()
    )

    if not calibration_toml.exists():
        raise FileNotFoundError(f"Calibration TOML not found: {calibration_toml}")
    if not calibration_videos_folder.exists():
        raise FileNotFoundError(f"Calibration videos folder not found: {calibration_videos_folder}")
    if charuco_board_name not in CHARUCO_BOARDS:
        raise ValueError(f"Unknown Charuco board: {charuco_board_name}")

    video_paths = get_video_paths(path_to_video_folder=calibration_videos_folder)
    if not video_paths:
        raise ValueError(f"No calibration videos found in: {calibration_videos_folder}")

    progress_callback(f"Loading existing calibration TOML: {calibration_toml}")
    cam_group = freemocap_anipose.CameraGroup.load(str(calibration_toml))
    if len(cam_group.cameras) != len(video_paths):
        raise ValueError(
            f"Existing TOML has {len(cam_group.cameras)} cameras, but groundplane recording has "
            f"{len(video_paths)} videos. Use the same cameras in the same order."
        )

    charuco_board_definition = CHARUCO_BOARDS[charuco_board_name]()
    calibrator = AniposeCameraCalibrator(
        charuco_board_object=charuco_board_definition,
        charuco_square_size=float(charuco_square_size),
        calibration_videos_folder_path=calibration_videos_folder,
        progress_callback=progress_callback,
    )
    calibrator._anipose_camera_group_object = cam_group
    calibrator._list_of_video_paths = video_paths

    cam_group.metadata["groundplane_source_toml"] = str(calibration_toml)
    cam_group.metadata["groundplane_source_videos"] = str(calibration_videos_folder)
    cam_group.metadata["groundplane_charuco_square_size"] = float(charuco_square_size)
    cam_group.metadata["groundplane_charuco_board_object"] = str(charuco_board_definition)
    cam_group.metadata["groundplane_date_time_calibrated"] = str(np.datetime64("now"))

    progress_callback("Detecting Charuco corners for groundplane-only calibration")
    videos = [[str(path)] for path in video_paths]
    all_rows = cam_group.get_rows_videos(videos, calibrator._anipose_charuco_board, verbose=True)
    cam_group._validate_charuco_rows_for_calibration(all_rows)

    progress_callback("Applying Charuco board as the new groundplane")
    cam_group, groundplane_success = calibrator.set_charuco_board_as_groundplane(cam_group)
    if not groundplane_success.success:
        raise ValueError(groundplane_success.error or "Groundplane calibration failed")

    cam_group.metadata["groundplane_calibration"] = True
    calibrator.get_real_world_matrices(cam_group)

    output_toml.parent.mkdir(parents=True, exist_ok=True)
    cam_group.dump(output_toml)
    progress_callback(f"Groundplane-adjusted calibration saved: {output_toml}")

    if update_last_successful:
        last_successful_path = Path(get_last_successful_calibration_toml_path())
        last_successful_path.parent.mkdir(parents=True, exist_ok=True)
        cam_group.dump(last_successful_path)
        progress_callback(f"Last successful calibration updated: {last_successful_path}")

    return GroundplaneOnlyCalibrationResult(
        output_toml_path=output_toml,
        charuco_3d_path=calibrator._recording_folder_path / "output_data" / "charuco_3d_xyz.npy",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply only Charuco groundplane alignment to an existing TOML.")
    parser.add_argument("--calibration-toml", type=Path, required=True)
    parser.add_argument("--calibration-videos-folder", type=Path, required=True)
    parser.add_argument("--charuco-square-size", type=float, default=39.0)
    parser.add_argument("--charuco-board", type=str, default="7x5 Charuco", choices=sorted(CHARUCO_BOARDS))
    parser.add_argument("--output-toml", type=Path, default=None)
    parser.add_argument("--update-last-successful", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = apply_groundplane_to_calibration_toml(
        calibration_toml=args.calibration_toml,
        calibration_videos_folder=args.calibration_videos_folder,
        charuco_square_size=args.charuco_square_size,
        charuco_board_name=args.charuco_board,
        output_toml=args.output_toml,
        update_last_successful=args.update_last_successful,
        progress_callback=print,
    )
    print(f"Output TOML: {result.output_toml_path}")
    print(f"Charuco 3D debug data: {result.charuco_3d_path}")


if __name__ == "__main__":
    main()
