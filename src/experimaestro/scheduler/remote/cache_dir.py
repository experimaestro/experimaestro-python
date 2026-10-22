"""Local cache directories of the remote monitors

Each SSH monitoring session copies remote files (job logs, service data) in
its own temporary directory, removed when the session ends. A monitor that
does not end normally (killed, crashed) leaves its directory behind: the next
monitors remove the directories whose monitor is gone.

Only the directories created by a monitor are removed: they hold a marker
file (:data:`MARKER`) naming the host and process of the monitor.
"""

import json
import logging
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("xpm.remote.client")

PREFIX = "xpm_remote_"
#: Marker of a cache directory, with its owner (JSON)
MARKER = ".experimaestro-remote-cache"


def create_cache_dir() -> Path:
    """Create the cache directory of a monitoring session (after removing the
    ones left by the monitors of this host that are gone)"""
    remove_stale_cache_dirs()
    path = Path(tempfile.mkdtemp(prefix=PREFIX))
    (path / MARKER).write_text(
        json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "created": time.time()}
        )
    )
    return path


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user's process
        return True
    return True


def _stale(path: Path) -> bool:
    """Whether a directory is the cache of a monitor of this host that is gone"""
    if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.getuid():
        return False
    try:
        owner = json.loads((path / MARKER).read_text())
        pid, host = int(owner["pid"]), owner["host"]
    except (OSError, ValueError, KeyError, TypeError):
        # Not a monitor cache (or not readable): left alone
        return False
    return host == socket.gethostname() and pid != os.getpid() and not _running(pid)


def remove_stale_cache_dirs(directory: Path | None = None) -> list[Path]:
    """Remove the cache directories of the monitors of this host that are
    gone; returns them"""
    directory = Path(tempfile.gettempdir()) if directory is None else directory
    removed = []
    for path in directory.glob(f"{PREFIX}*"):
        try:
            if _stale(path):
                shutil.rmtree(path)
                removed.append(path)
        except OSError as e:
            logger.warning("Could not remove the stale cache %s: %s", path, e)
    if removed:
        logger.info("Removed %d stale remote cache directories", len(removed))
    return removed
