from __future__ import annotations

from dataclasses import dataclass

import skellycam_capture_config_patch as patch


class FakeCv2:
    CAP_PROP_EXPOSURE = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_FOURCC = 5

    @staticmethod
    def VideoWriter_fourcc(*letters):
        return "".join(letters)


class FakeCapture:
    def __init__(self):
        self.calls = []
        self.props = {
            FakeCv2.CAP_PROP_FRAME_WIDTH: 1920,
            FakeCv2.CAP_PROP_FRAME_HEIGHT: 1080,
            FakeCv2.CAP_PROP_FPS: 30,
            FakeCv2.CAP_PROP_FOURCC: 0,
        }

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)


@dataclass
class FakeConfig:
    camera_id: str = "2"
    exposure: int = -5
    resolution_width: int = 1920
    resolution_height: int = 1080
    framerate: int = 30
    fourcc: str = "MJPG"


def test_patched_configuration_sets_fourcc_before_resolution(monkeypatch) -> None:
    fake_capture = FakeCapture()
    monkeypatch.setattr(patch, "cv2", FakeCv2)

    patch.apply_configuration_fourcc_first(fake_capture, FakeConfig())

    assert fake_capture.calls[:3] == [
        (FakeCv2.CAP_PROP_FOURCC, "MJPG"),
        (FakeCv2.CAP_PROP_FRAME_WIDTH, 1920),
        (FakeCv2.CAP_PROP_FRAME_HEIGHT, 1080),
    ]
