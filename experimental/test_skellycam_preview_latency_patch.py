from __future__ import annotations

import queue

import skellycam_preview_latency_patch as latency_patch


def test_drain_latest_frame_returns_last_frame_and_drops_old_frames() -> None:
    frame_queue: queue.Queue = queue.Queue()
    frame_queue.put("old")
    frame_queue.put("new")

    assert latency_patch.drain_latest_frame(frame_queue) == "new"
    assert frame_queue.empty()


def test_drain_latest_frame_respects_max_drain_frames() -> None:
    frame_queue: queue.Queue = queue.Queue()
    frame_queue.put("old")
    frame_queue.put("middle")
    frame_queue.put("new")

    assert latency_patch.drain_latest_frame(frame_queue, max_drain_frames=2) == "middle"
    assert frame_queue.get_nowait() == "new"


def test_installed_patch_preserves_empty_and_unknown_camera_behavior(monkeypatch) -> None:
    monkeypatch.setattr(latency_patch, "_PATCH_INSTALLED", False)

    class FakeProcess:
        def __init__(self) -> None:
            self._queues = {"0": queue.Queue()}

        def _get_queue_by_camera_id(self, camera_id):
            return self._queues[camera_id]

    assert latency_patch.install_latest_frame_preview_patch(process_class=FakeProcess)
    process = FakeProcess()

    assert process.get_current_frame_by_camera_id("missing") is None
    assert process.get_current_frame_by_camera_id("0") is None


def test_installed_patch_returns_latest_frame(monkeypatch) -> None:
    monkeypatch.setattr(latency_patch, "_PATCH_INSTALLED", False)

    class FakeProcess:
        def __init__(self) -> None:
            self._queues = {"0": queue.Queue()}
            self._queues["0"].put("old")
            self._queues["0"].put("new")

        def _get_queue_by_camera_id(self, camera_id):
            return self._queues[camera_id]

    assert latency_patch.install_latest_frame_preview_patch(process_class=FakeProcess)
    process = FakeProcess()

    assert process.get_current_frame_by_camera_id("0") == "new"
    assert process._queues["0"].empty()
