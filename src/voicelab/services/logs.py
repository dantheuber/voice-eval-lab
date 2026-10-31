"""Keeps server and eval logs from filling the disk."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

LOG_MAX_BYTES = 32 * 1024 * 1024
LOG_BACKUPS = 3


def rotate_log(path: Path, *, max_bytes: int = LOG_MAX_BYTES, backups: int = LOG_BACKUPS) -> bool:
    """Move an oversized log to `path.1`, shifting older copies, and empty it.

    This copies and truncates instead of renaming, because a running server holds the file open
    and would keep writing to the renamed one. Servers write with O_APPEND, so after the truncate
    they carry on from the new end. Lines written during the copy are lost, which is fine for a log.
    """
    try:
        if path.stat().st_size <= max_bytes:
            return False
    except OSError:
        return False
    for index in range(backups - 1, 0, -1):
        older = Path(f"{path}.{index}")
        if older.exists():
            os.replace(older, f"{path}.{index + 1}")
    shutil.copyfile(path, f"{path}.1")
    os.truncate(path, 0)
    return True
