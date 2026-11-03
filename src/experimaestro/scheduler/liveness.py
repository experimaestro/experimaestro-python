"""Liveness of experiments and jobs, decided from their locks

See ``docs/source/development/filesystem-protocol.md`` (I5): whether an
experiment or a job is alive is decided from the lock its process holds (the
kernel releases it when the process dies), never from a status field that a
crashed process never updates. Destructive operations take the lock of what
they destroy, without waiting, and refuse if it is held.
"""

import json
import logging
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Iterator

import filelock

from experimaestro.locking import create_file_lock

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseJob

logger = logging.getLogger("xpm.liveness")


class JobInUseError(RuntimeError):
    """A job directory cannot be removed: its job runs or is being started"""


def lock_is_held(path: Path) -> bool:
    """Whether a process holds the lock ``path``

    The lock is taken (without waiting) and released at once if it is free.
    """
    if not path.exists():
        return False
    lock = create_file_lock(path, timeout=0)
    try:
        lock.acquire()
    except filelock.Timeout:
        return True
    lock.release()
    return False


def experiment_is_live(experiment_dir: Path) -> bool:
    """Whether a run of the experiment (``experiments/<xp>``) is live"""
    return lock_is_held(experiment_dir / "lock")


def _run_sort_key(run_dir: Path) -> tuple[str, int]:
    """Run ids are ``YYYYmmdd_HHMMSS[.k]``"""
    base, _, suffix = run_dir.name.partition(".")
    return (base, int(suffix) if suffix.isdigit() else 0)


def latest_run_dir(experiment_dir: Path) -> Path | None:
    """The most recent run directory of an experiment

    Run directories are created under the experiment lock, so while the lock
    is held, the most recent run is the live one (``current`` is re-pointed
    later).
    """
    runs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
    ]
    return max(runs, key=_run_sort_key, default=None)


def run_is_live(run_dir: Path) -> bool:
    """Whether ``run_dir`` (``experiments/<xp>/<run_id>``) is a live run"""
    experiment_dir = run_dir.parent
    if not experiment_is_live(experiment_dir):
        return False
    latest = latest_run_dir(experiment_dir)
    return latest is not None and latest.name == run_dir.name


def _job_process_alive(job: "BaseJob") -> bool:
    """Whether the process in the job's pid file exists

    A pid file that cannot be read or checked counts as alive.
    """
    from experimaestro.connectors import Process
    from experimaestro.connectors.local import LocalConnector

    if not job.pidfile.exists():
        return False
    try:
        spec = json.loads(job.pidfile.read_text())
        return Process.fromDefinition(LocalConnector.instance(), spec) is not None
    except Exception as e:
        logger.warning("Cannot check the process of %s: %s", job.path, e)
        return True


@contextmanager
def job_locked_for_removal(job: "BaseJob") -> Iterator[None]:
    """Holds the job lock while the job directory is changed or removed

    Raises:
        JobInUseError: if the job lock is held (the job runs or is being
            started) or if the job's process still exists
    """
    lock = create_file_lock(job.lockpath, timeout=0)
    try:
        lock.acquire()
    except filelock.Timeout:
        raise JobInUseError(f"job {job.identifier} is running") from None
    try:
        if _job_process_alive(job):
            raise JobInUseError(f"the process of job {job.identifier} is running")
        yield
    finally:
        lock.release()


#: Suffix of a job directory renamed by :func:`remove_job_directory` before
#: being removed (``.<job id>.removing-<random>``, next to the job directory)
REMOVING_MARKER = ".removing-"


def is_removal_leftover(path: Path) -> bool:
    """Whether ``path`` is a job directory being (or not fully) removed"""
    return path.name.startswith(".") and REMOVING_MARKER in path.name


def remove_job_directory(job: "BaseJob") -> None:
    """Removes the directory of a job that does not run

    The job lock lives in the job directory and is held while the directory is
    changed: removing the directory while the lock file is open fails on NFS
    (the open file is renamed ``.nfsXXXX`` and the directory is never empty).
    The directory is thus renamed while the lock is held — the job path is
    then free, a job starting afterwards creates a new directory — and removed
    once the lock is released.

    Raises:
        JobInUseError: if the job runs or is being started
        OSError: if the job directory cannot be renamed
    """
    if not job.path.exists():
        return
    path = job.path
    with job_locked_for_removal(job):
        removed = path.with_name(f".{path.name}{REMOVING_MARKER}{uuid.uuid4().hex[:8]}")
        path.rename(removed)
    remove_leftover(removed)


def remove_leftover(path: Path) -> bool:
    """Removes a job directory renamed by :func:`remove_job_directory`

    Returns:
        whether the directory was removed (a failure is logged: the directory
        is removed by :func:`remove_removal_leftovers` later)
    """
    try:
        shutil.rmtree(path)
        return True
    except OSError as e:
        logger.warning("Could not fully remove %s (removed later): %s", path, e)
        return False


def remove_removal_leftovers(jobs_dir: Path) -> int:
    """Removes the job directories that :func:`remove_job_directory` renamed
    but could not remove (``jobs/<task id>/.<job id>.removing-*``)

    Returns:
        the number of directories removed
    """
    count = 0
    if not jobs_dir.is_dir():
        return 0
    for task_dir in jobs_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for path in task_dir.iterdir():
            if is_removal_leftover(path) and path.is_dir():
                count += remove_leftover(path)
    return count
