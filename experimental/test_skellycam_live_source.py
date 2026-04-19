from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np

import skellycam_mocap_diagnostics as diagnostics
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

        def triangulate_ransac(self, points, progress=False):
            return self.triangulate(points, progress=progress)

        def reprojection_error(self, points_3d, points_2d, mean=False):
            if mean:
                return np.full(points_2d.shape[1], 2.0, dtype=np.float32)
            return np.zeros((points_2d.shape[0], points_2d.shape[1], 2), dtype=np.float32)

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


def test_rotate_points_2d_90_clockwise() -> None:
    points = np.asarray([[0.0, 0.0], [10.0, 20.0], [np.nan, 5.0]], dtype=np.float32)
    rotated = live_source.rotate_points_2d(points, image_size=(100, 50), rotation_degrees=90)

    assert np.allclose(rotated[0], [49.0, 0.0])
    assert np.allclose(rotated[1], [29.0, 10.0])
    assert np.isnan(rotated[2]).any()


def test_apply_camera_rotation_variants_updates_sizes() -> None:
    points = np.zeros((2, 3, 2), dtype=np.float32)
    rotated_points, rotated_sizes = live_source.apply_camera_rotation_variants(
        points_2d=points,
        image_sizes=[(1920, 1080), (640, 480)],
        rotation_degrees_by_camera=[0, 270],
    )

    assert rotated_points.shape == points.shape
    assert rotated_sizes == [(1920, 1080), (480, 640)]


def test_replace_latest_queue_item_keeps_only_latest() -> None:
    latest_queue: Queue[int] = Queue(maxsize=1)

    diagnostics.replace_latest_queue_item(latest_queue, 1)
    diagnostics.replace_latest_queue_item(latest_queue, 2)

    assert latest_queue.qsize() == 1
    assert latest_queue.get_nowait() == 2


def test_plausibility_metrics_and_likely_cause() -> None:
    points_m = np.full((33, 3), np.nan, dtype=np.float32)
    points_m[diagnostics.MEDIAPIPE["nose"]] = [0.0, 1.7, 0.0]
    points_m[diagnostics.MEDIAPIPE["left_ear"]] = [-0.05, 1.65, 0.0]
    points_m[diagnostics.MEDIAPIPE["right_ear"]] = [0.05, 1.65, 0.0]
    points_m[diagnostics.MEDIAPIPE["left_shoulder"]] = [-0.2, 1.45, 0.0]
    points_m[diagnostics.MEDIAPIPE["right_shoulder"]] = [0.2, 1.45, 0.0]
    points_m[diagnostics.MEDIAPIPE["left_hip"]] = [-0.15, 1.0, 0.0]
    points_m[diagnostics.MEDIAPIPE["right_hip"]] = [0.15, 1.0, 0.0]
    points_m[diagnostics.MEDIAPIPE["left_knee"]] = [-0.15, 0.55, 0.0]
    points_m[diagnostics.MEDIAPIPE["right_knee"]] = [0.15, 0.55, 0.0]
    points_m[diagnostics.MEDIAPIPE["left_ankle"]] = [-0.15, 0.12, 0.0]
    points_m[diagnostics.MEDIAPIPE["right_ankle"]] = [0.15, 0.12, 0.0]
    points_m[diagnostics.MEDIAPIPE["left_heel"]] = [-0.15, 0.04, -0.04]
    points_m[diagnostics.MEDIAPIPE["right_heel"]] = [0.15, 0.04, -0.04]
    points_m[diagnostics.MEDIAPIPE["left_foot_index"]] = [-0.15, 0.04, 0.16]
    points_m[diagnostics.MEDIAPIPE["right_foot_index"]] = [0.15, 0.04, 0.16]

    plausibility = diagnostics._plausibility_metrics(points_m)

    assert plausibility["height_plausible"] is True
    assert plausibility["segment_plausible"] is True
    assert plausibility["plausible_metric_ratio"] is not None
    assert plausibility["plausible_metric_ratio"] > 0.8

    likely_cause = diagnostics.classify_likely_cause(
        {
            "valid_2d_point_ratio_mean": 0.98,
            "reprojection_error_px_mean_mean": 28.0,
            "plausible_metric_ratio_mean": 0.9,
            "camera_skew_ms_mean": 10.0,
            "camera_skew_ms_max": 12.0,
        },
        max_camera_skew_ms=50.0,
    )

    assert likely_cause == "calibration_or_camera_order_or_rotation_mismatch"
