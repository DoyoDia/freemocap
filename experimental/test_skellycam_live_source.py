from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import skellycam_live_source as live_source


class FakeCameraConfig:
    def __init__(self, camera_id: str = "0", use_this_camera: bool = True, **kwargs):
        self.camera_id = str(camera_id)
        self.use_this_camera = bool(use_this_camera)
        self.extra = kwargs

    def copy(self, update=None):
        data = {"camera_id": self.camera_id, "use_this_camera": self.use_this_camera, **self.extra}
        if update:
            data.update(update)
        return FakeCameraConfig(**data)


def test_build_camera_config_dictionary_respects_explicit_order() -> None:
    camera_ids, configs = live_source._build_camera_config_dictionary(
        camera_ids=["1", "0"],
        camera_config_json=None,
        camera_config_class=FakeCameraConfig,
        expected_camera_count=2,
    )

    assert camera_ids == ["1", "0"]
    assert list(configs) == ["1", "0"]
    assert all(config.use_this_camera for config in configs.values())


@dataclass
class FakeFramePayload:
    image: np.ndarray
    timestamp_ns: int
    success: bool = True


def test_frames_are_synchronized_uses_timestamp_skew() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    assert live_source.frames_are_synchronized(
        [
            FakeFramePayload(image=frame, timestamp_ns=1_000_000_000),
            FakeFramePayload(image=frame, timestamp_ns=1_020_000_000),
        ],
        max_camera_skew_ms=50.0,
    )
    assert not live_source.frames_are_synchronized(
        [
            FakeFramePayload(image=frame, timestamp_ns=1_000_000_000),
            FakeFramePayload(image=frame, timestamp_ns=1_100_000_000),
        ],
        max_camera_skew_ms=50.0,
    )


def test_live_source_fake_frame_payload_yields_mocap_frame(monkeypatch) -> None:
    calibration_toml = Path(__file__).resolve()

    class FakeCalibration:
        cameras = [object(), object()]

        def triangulate(self, points, progress=False):
            assert points.shape == (2, 33, 2)
            return np.arange(99, dtype=np.float32).reshape(33, 3)

    class FakeAniposeCameraGroup:
        @staticmethod
        def load(path):
            assert Path(path) == calibration_toml
            return FakeCalibration()

    class FakeTracker:
        def process_points(self, image, image_size):
            points = np.zeros((33, 3), dtype=np.float32)
            points[:, 0] = np.arange(33, dtype=np.float32)
            points[:, 1] = np.arange(33, dtype=np.float32) + 1.0
            return points

        def cleanup(self):
            pass

    class FakeSkellyCameraGroup:
        def __init__(self, camera_ids_list, camera_config_dictionary):
            self.camera_ids = list(camera_ids_list)
            self.camera_config_dictionary = camera_config_dictionary
            self._groups = [
                {
                    "0": FakeFramePayload(
                        image=np.zeros((4, 6, 3), dtype=np.uint8),
                        timestamp_ns=1_000_000_000,
                    ),
                    "1": FakeFramePayload(
                        image=np.zeros((4, 6, 3), dtype=np.uint8),
                        timestamp_ns=1_010_000_000,
                    ),
                }
            ]

        def start(self):
            pass

        def latest_frames(self):
            if self._groups:
                return self._groups.pop(0)
            return {camera_id: None for camera_id in self.camera_ids}

        def close(self):
            pass

    monkeypatch.setattr(live_source, "_load_camera_group_class", lambda: FakeAniposeCameraGroup)
    monkeypatch.setattr(
        live_source,
        "make_trackers",
        lambda num_cameras, model_complexity, static_image_mode, tracker_backend: [
            FakeTracker()
            for _ in range(num_cameras)
        ],
    )

    frames = list(
        live_source.iter_skellycam_mocap_3d_frames(
            calibration_toml=calibration_toml,
            camera_ids="0,1",
            max_frames=1,
            parallel_camera_tracking=False,
            prewarm=False,
            skellycam_importer=lambda skellycam_home: (FakeSkellyCameraGroup, FakeCameraConfig),
        )
    )

    assert len(frames) == 1
    assert frames[0].seq == 0
    assert frames[0].timestamp_ns == 1_010_000_000
    assert frames[0].points_3d.shape == (33, 3)
    assert frames[0].valid_2d_point_ratio == 1.0
