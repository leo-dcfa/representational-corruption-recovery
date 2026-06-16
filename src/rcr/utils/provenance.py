"""Per-run provenance: package versions + git SHA + hardware (SPEC §2.5, §6).

Every artifact must trace to an environment. ``provenance()`` returns a JSON-
serializable dict that gets written alongside every run / dataset / figure.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from rcr.config import REPO_ROOT

# Packages whose versions we always record (load-bearing for reproducibility).
_TRACKED = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "numpy",
    "scipy",
    "scikit-learn",
    "sentence-transformers",
    "datasketch",
    "openai",
]


def _git_sha(repo: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty(repo: Path = REPO_ROOT) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        idx = torch.cuda.current_device()
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(idx),
            "capability": list(torch.cuda.get_device_capability(idx)),
            "torch_cuda": torch.version.cuda,
        }
    except ImportError:
        return {"cuda_available": False, "note": "torch not importable"}


def provenance() -> dict:
    """Full provenance record for the current environment."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }
