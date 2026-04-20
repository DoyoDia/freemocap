from __future__ import annotations

try:
    from skellycam_capture_config_patch import install_skellycam_capture_config_patch

    install_skellycam_capture_config_patch()
except Exception:
    # The parent process may import sitecustomize before SKELLYCAM_HOME is set.
    # Experimental entrypoints install the same patch explicitly after runtime
    # paths are configured; spawned camera processes inherit SKELLYCAM_HOME and
    # install this successfully at interpreter startup.
    pass

