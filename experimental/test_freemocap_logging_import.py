from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_freemocap_with_existing_root_handler_does_not_cycle() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import logging
logging.getLogger().addHandler(logging.NullHandler())
import freemocap
assert hasattr(freemocap, "logger")
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
