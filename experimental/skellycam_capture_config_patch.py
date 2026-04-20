from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import cv2

logger = logging.getLogger(__name__)

_PATCH_INSTALLED = False


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _fourcc_to_string(value: Any) -> str:
    try:
        value = int(value)
        return "".join(chr((value >> 8 * index) & 0xFF) for index in range(4))
    except Exception:
        return str(value)


def apply_configuration_fourcc_first(cv2_vid_cap: cv2.VideoCapture, config: Any) -> None:
    logger.info(
        f"Applying patched configuration to Camera {config.camera_id}:"
        f"Fourcc: {config.fourcc}, "
        f"Resolution width: {config.resolution_width}, "
        f"Resolution height: {config.resolution_height}, "
        f"Framerate: {config.framerate}, "
        f"Exposure: {config.exposure}"
    )
    try:
        if not cv2_vid_cap.isOpened():
            logger.error(
                f"Failed to apply configuration to Camera {config.camera_id} - camera is not open"
            )
            return
    except Exception:
        logger.error(f"Failed when trying to check if Camera {config.camera_id} is open")
        return

    try:
        # Some UVC cameras only expose HD modes after the stream is switched to
        # MJPG. Skellycam's upstream order sets FourCC last, which can silently
        # leave a camera at the default 640x480 mode.
        cv2_vid_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(config.fourcc)))
        cv2_vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution_width)
        cv2_vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution_height)
        cv2_vid_cap.set(cv2.CAP_PROP_FPS, config.framerate)
        cv2_vid_cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure)

        actual_width = _coerce_int(cv2_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = _coerce_int(cv2_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cv2_vid_cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = _fourcc_to_string(cv2_vid_cap.get(cv2.CAP_PROP_FOURCC))
        requested_width = _coerce_int(config.resolution_width)
        requested_height = _coerce_int(config.resolution_height)
        if (
            actual_width is not None
            and actual_height is not None
            and requested_width is not None
            and requested_height is not None
            and [actual_width, actual_height] != [requested_width, requested_height]
        ):
            logger.warning(
                f"Camera {config.camera_id} reported {actual_width}x{actual_height} after requesting "
                f"{requested_width}x{requested_height}. Actual frame shape will be checked downstream."
            )
        logger.info(
            f"Camera {config.camera_id} reported capture properties after patch: "
            f"{actual_width}x{actual_height}, fps={actual_fps}, fourcc={actual_fourcc}"
        )
    except Exception as exc:
        logger.error(f"Problem applying patched configuration for camera: {config.camera_id}")
        traceback.print_exc()
        raise exc


def install_skellycam_capture_config_patch() -> bool:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return False

    from skellycam.opencv.config import apply_config

    apply_config.apply_configuration = apply_configuration_fourcc_first
    try:
        from skellycam.opencv.camera import internal_camera_thread

        internal_camera_thread.apply_configuration = apply_configuration_fourcc_first
    except Exception:
        logger.debug("Could not patch already-imported internal_camera_thread", exc_info=True)

    _PATCH_INSTALLED = True
    logger.info("Installed skellycam FourCC-first camera configuration patch")
    return True

