from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


PatchState = Literal["not_needed", "patched", "missing"]


@dataclass(frozen=True)
class GMRRuntimeStatus:
    gmr_root: Path
    smplx_root: Optional[Path]
    unitree_g1_xml: Path
    patch_state: PatchState


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_gmr_paths(repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    gmr_root = root / "external" / "GMR"
    _prepend_sys_path(gmr_root)
    _prepend_sys_path(root / "external" / "smplx")
    return gmr_root


def _scipy_supports_scalar_first() -> bool:
    try:
        from scipy.spatial.transform import Rotation as R

        R.from_quat([1.0, 0.0, 0.0, 0.0], scalar_first=True)
        R.identity().as_quat(scalar_first=True)
    except TypeError:
        return False
    except Exception:
        return False
    return True


def gmr_scipy_patch_state(repo_root: Path | None = None) -> PatchState:
    if _scipy_supports_scalar_first():
        return "not_needed"

    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    gmr_root = root / "external" / "GMR"
    required_files = [
        gmr_root / "general_motion_retargeting" / "motion_retarget.py",
        gmr_root / "general_motion_retargeting" / "neck_retarget.py",
    ]
    for file_path in required_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            return "missing"
        if "_rotation_from_quat_wxyz" not in text:
            return "missing"
    return "patched"


def validate_gmr_runtime(
    repo_root: Path | None = None,
    *,
    require_mujoco: bool = False,
    require_patch: bool = True,
) -> GMRRuntimeStatus:
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    gmr_root = ensure_gmr_paths(root)
    if not gmr_root.exists():
        raise RuntimeError(
            f"GMR submodule is missing at {gmr_root}. "
            "Run: git submodule update --init --recursive external/GMR"
        )

    try:
        importlib.import_module("general_motion_retargeting")
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import general_motion_retargeting from {gmr_root}: {exc}. "
            "Install the package into the active environment with "
            "'python -m pip install -e external/GMR'."
        ) from exc

    if require_mujoco:
        try:
            importlib.import_module("mujoco")
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import mujoco: {exc}. Install it into the active environment before "
                "starting the MuJoCo viewer."
            ) from exc

    unitree_g1_xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    if not unitree_g1_xml.exists():
        raise RuntimeError(f"Unitree G1 MuJoCo XML is missing: {unitree_g1_xml}")

    patch_state = gmr_scipy_patch_state(root)
    if require_patch and patch_state == "missing":
        patch_path = root / "experimental" / "patches" / "gmr_scipy_compat.patch"
        raise RuntimeError(
            "The local SciPy build does not support scalar_first=True, and the GMR SciPy "
            f"compatibility patch is not applied. Apply {patch_path} inside external/GMR, for example: "
            'git -C external/GMR apply ../../experimental/patches/gmr_scipy_compat.patch'
        )

    smplx_root = root / "external" / "smplx"
    return GMRRuntimeStatus(
        gmr_root=gmr_root,
        smplx_root=smplx_root if smplx_root.exists() else None,
        unitree_g1_xml=unitree_g1_xml,
        patch_state=patch_state,
    )
