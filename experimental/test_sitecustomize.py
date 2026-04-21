from __future__ import annotations

import importlib


def test_sitecustomize_skips_bridge_entrypoint(monkeypatch) -> None:
    module = importlib.import_module("sitecustomize")

    monkeypatch.setenv(module.SKELLYCAM_CAPTURE_PATCH_STARTUP_ENV, "1")
    monkeypatch.setattr(module.sys, "argv", [r"C:\repo\experimental\freemocap_to_gmr_bridge.py"])

    assert module._should_install_skellycam_capture_patch() is False


def test_sitecustomize_allows_non_bridge_startup_when_enabled(monkeypatch) -> None:
    module = importlib.import_module("sitecustomize")

    monkeypatch.setenv(module.SKELLYCAM_CAPTURE_PATCH_STARTUP_ENV, "1")
    monkeypatch.setattr(module.sys, "argv", [r"C:\repo\experimental\skellycam_mocap_diagnostics.py"])

    assert module._should_install_skellycam_capture_patch() is True
