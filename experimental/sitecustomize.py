from __future__ import annotations

import os
import sys


SKELLYCAM_CAPTURE_PATCH_STARTUP_ENV = "FREEMOCAP_SKELLYCAM_CAPTURE_PATCH_AT_STARTUP"


def _is_bridge_entrypoint() -> bool:
    return any(str(arg).replace("\\", "/").endswith("freemocap_to_gmr_bridge.py") for arg in sys.argv)


def _should_install_skellycam_capture_patch() -> bool:
    if os.environ.get(SKELLYCAM_CAPTURE_PATCH_STARTUP_ENV) != "1":
        return False
    # The realtime bridge must import GMR/MuJoCo before skellycam/OpenGL. Its
    # camera workers enable this hook later, after the GMR processes are spawned.
    return not _is_bridge_entrypoint()


try:
    if _should_install_skellycam_capture_patch():
        from skellycam_capture_config_patch import install_skellycam_capture_config_patch

        install_skellycam_capture_config_patch()
except Exception:
    # The parent process may import sitecustomize before SKELLYCAM_HOME is set.
    # Experimental entrypoints install the same patch explicitly after runtime
    # paths are configured; spawned camera processes inherit SKELLYCAM_HOME and
    # install this successfully at interpreter startup.
    pass
