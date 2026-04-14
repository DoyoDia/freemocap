from __future__ import annotations

import logging
import queue
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATCH_INSTALLED = False


def drain_latest_frame(frame_queue: Any, max_drain_frames: int = 500) -> Optional[Any]:
    latest_frame = None
    for _ in range(max(1, int(max_drain_frames))):
        try:
            latest_frame = frame_queue.get(block=False)
        except queue.Empty:
            break
    return latest_frame


def install_latest_frame_preview_patch(max_drain_frames: int = 500, process_class: Any = None) -> bool:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return False

    if process_class is None:
        from skellycam.opencv.group.strategies.cam_group_queue_process import CamGroupQueueProcess

        process_class = CamGroupQueueProcess

    def get_current_frame_by_camera_id(self, camera_id):
        try:
            if camera_id not in self._queues:
                return None

            latest_frame = drain_latest_frame(
                self._get_queue_by_camera_id(camera_id),
                max_drain_frames=max_drain_frames,
            )
            return latest_frame
        except Exception as exc:
            logger.exception(f"Problem when grabbing latest frame from: Camera {camera_id} - {exc}")
            return None

    process_class.get_current_frame_by_camera_id = get_current_frame_by_camera_id
    _PATCH_INSTALLED = True
    logger.info("Installed skellycam latest-frame preview patch")
    return True
