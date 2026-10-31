"""Named crash points, used by the filesystem protocol tests

Each write point of the inter-process protocol (see
``docs/source/development/filesystem-protocol.md``) calls
``crashpoint("<actor>.<point>")``. It does nothing unless the point is
activated, through the ``XPM_CRASHPOINTS`` environment variable (inherited by
the job processes) or :func:`activate` (in-process):

``XPM_CRASHPOINTS="job.lock_released=pause,experiment.job_submitted=kill"``

Actions:

- ``kill``: the process sends itself SIGKILL (as an OOM kill or ``kill -9``);
- ``raise``: raises :class:`CrashPointReached` (a ``BaseException``, so that
  ``except Exception`` blocks do not swallow it) — for in-process tests;
- ``pause``: writes ``<name>.reached`` in ``XPM_CRASHPOINTS_DIR`` and waits
  until ``<name>.resume`` exists there, so that a test can act while the
  process is stopped at that point (concurrent actors).
"""

import logging
import os
import signal
import time
from pathlib import Path

logger = logging.getLogger("xpm.crashpoints")

ENV_CRASHPOINTS = "XPM_CRASHPOINTS"
ENV_CRASHPOINTS_DIR = "XPM_CRASHPOINTS_DIR"


class CrashPointReached(BaseException):
    """Raised by a crash point with the ``raise`` action"""


def _parse(value: str) -> dict[str, str]:
    points = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, action = item.partition("=")
        points[name.strip()] = action.strip() or "kill"
    return points


#: Active crash points (name -> action), empty in normal operation. Read
#: from the environment at the first call, since the job scripts set their
#: environment after importing experimaestro.
_ACTIVE: dict[str, str] | None = None


def activate(points: dict[str, str] | None):
    """(Re)defines the active crash points of this process

    ``None`` reads them again from the environment at the next call.
    """
    global _ACTIVE
    _ACTIVE = None if points is None else dict(points)


def crashpoint(name: str):
    """Crash, raise or pause here if the crash point ``name`` is active"""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _parse(os.environ.get(ENV_CRASHPOINTS, ""))
    if not _ACTIVE:
        return
    action = _ACTIVE.get(name)
    if action is None:
        return

    logger.warning("Crash point %s reached (%s)", name, action)
    match action:
        case "kill":
            os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(60)  # SIGKILL is asynchronous
        case "raise":
            raise CrashPointReached(name)
        case "pause":
            directory = Path(os.environ[ENV_CRASHPOINTS_DIR])
            (directory / f"{name}.reached").write_text(str(os.getpid()))
            resume = directory / f"{name}.resume"
            while not resume.exists():
                time.sleep(0.02)
        case _:
            raise ValueError(f"Unknown crash point action {action} for {name}")
