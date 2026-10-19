"""Disk usage of job folders, as shown by the CLI, the TUI and the web UI"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Units used by :func:`format_size` (powers of 1024)
_UNITS = ["B", "KB", "MB", "GB", "TB"]


def _walk_usage(path: Path) -> int:
    """Disk usage computed in Python (fallback when ``du`` is not available)"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            # st_blocks counts 512-byte blocks, like du; Windows has no blocks
            total += stat.st_size if sys.platform == "win32" else stat.st_blocks * 512
    return total


def folder_size(path: Path, timeout: float = 30) -> int:
    """Disk usage of a folder, in bytes

    Uses ``du -sk`` (the same on Linux and macOS), falling back to a Python
    walk counting blocks the same way. Returns 0 if the folder does not exist.
    """
    path = Path(path)
    if not path.exists():
        return 0
    if shutil.which("du"):
        try:
            output = subprocess.run(
                ["du", "-sk", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            ).stdout
            return int(output.split()[0]) * 1024
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            pass
    return _walk_usage(path)


def format_size(size: float) -> str:
    """Human-readable size (``512B``, ``1.5MB``)"""
    for unit in _UNITS[:-1]:
        if size < 1024:
            return f"{size}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}{_UNITS[-1]}"
