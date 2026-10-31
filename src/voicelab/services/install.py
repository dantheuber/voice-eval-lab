"""Whether a profile's programs and model files are on disk, checked before any server stops.

Two sources, because neither is enough alone. A server's command names its program and most of its
models, and works for any profile someone writes. The setup manifest names what no command does,
such as Phonon's voices, and is what `scripts/setup.sh` itself goes by.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from voicelab.core.profiles import LocalService

_MANIFEST = Path("scripts/setup/manifest.tsv")
_INSTALLED_UNDER = ".runtime/"


@dataclass(frozen=True)
class _Row:
    profiles: tuple[str, ...]
    required: bool
    path: str
    size: int


def _rows(root: Path) -> Iterator[_Row]:
    try:
        lines = (root / _MANIFEST).read_text().splitlines()
    except OSError:  # a profiles directory outside a checkout has no manifest
        return
    for line in lines:
        if line and not line.startswith("#"):
            profiles, need, path, size = line.split("\t")[:4]
            yield _Row(tuple(profiles.split(",")), need == "required", path, int(size))


def _program_found(program: str, cwd: Path) -> bool:
    if "/" not in program:
        return shutil.which(program) is not None
    path = cwd / program
    return path.is_file() and os.access(path, os.X_OK)


def missing_from_commands(services: Iterable[LocalService], cwd: Path) -> list[str]:
    """Programs and installed files these servers' commands name that aren't there."""
    missing = []
    for service in services:
        program, *arguments = service.command
        if not _program_found(program, cwd):
            missing.append(program)
        paths = [a for a in arguments if a.startswith(_INSTALLED_UNDER)]
        missing.extend(path for path in paths if not (cwd / path).exists())
    return missing


def missing_from_manifest(profile_id: str, root: Path) -> list[str]:
    """Required files setup puts in place for this profile that aren't there, or aren't whole."""
    missing = []
    for row in _rows(root):
        if row.required and profile_id in row.profiles:
            path = root / row.path
            if not path.exists():
                missing.append(row.path)
            elif path.stat().st_size != row.size:
                missing.append(f"{row.path} (incomplete)")
    return missing


def setup_command(profile_id: str, root: Path) -> str | None:
    """The command that installs this profile. None if setup doesn't know it."""
    if any(profile_id in row.profiles for row in _rows(root)):
        return f"scripts/setup.sh --profile {profile_id}"
    return None
