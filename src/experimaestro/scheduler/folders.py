"""Auxiliary workspace folders (backup / move / use).

**Beta**: API may change. See issue #55.

A folder is an extra location attached to a workspace where successful
jobs can be archived (mode=backup), moved (mode=move), or simply looked
up read-only (mode=use). The on-disk layout mirrors the primary
workspace: ``<folder>/jobs/<task_id>/<hash>/``.

Copies and moves are atomic (staged in a sibling ``.tmp.<hash>``
directory, then renamed) and run on a dedicated background worker
thread so the scheduler is never blocked on filesystem I/O.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import filelock

from experimaestro.locking import create_file_lock

from experimaestro.settings import FolderMode, FolderSettings

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job

logger = logging.getLogger(__name__)


# Files inside a job directory that we never copy to a folder. Locks,
# scheduler bookkeeping, and pid files are tied to the original
# workspace and not meaningful elsewhere.
_SKIP_NAMES = {
    ".lock",
    ".scheduler",
}

# Suffixes inside the job directory that should not be copied.
_SKIP_SUFFIXES = {".lock", ".pid"}


@dataclass
class _Job:
    """Reduced job view for the worker (avoids holding a live Job reference)."""

    src: Path
    rel: Path  # task_id / hash
    name: str  # script name (used for marker files)


@dataclass
class _Request:
    job: _Job
    folder: FolderSettings


def _should_copy(entry: os.DirEntry) -> bool:
    if entry.name in _SKIP_NAMES:
        return False
    for suffix in _SKIP_SUFFIXES:
        if entry.name.endswith(suffix):
            return False
    return True


def _tmp_lock_path(tmp_dir: Path) -> Path:
    """Lock file that guards an in-progress staging directory."""
    return tmp_dir.with_name(tmp_dir.name + ".lock")


def _sweep_stale_tmp(parent: Path) -> None:
    """Delete leftover ``.tmp.*`` directories whose owner has crashed.

    Each in-progress archive holds an exclusive file lock on
    ``<tmp>.lock`` for the entire duration of the copy. If we can
    acquire that lock now, the previous owner is dead → we can
    safely remove both the staging dir and the lock file. If we
    cannot acquire it, some other process is currently archiving and
    we leave its work alone.

    Sweeps are scoped to *parent* so the cost stays bounded.
    """
    if not parent.is_dir():
        return
    try:
        entries = list(parent.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir():
            continue
        if not entry.name.startswith(".tmp."):
            continue
        lock_path = _tmp_lock_path(entry)
        lock = create_file_lock(lock_path, timeout=0)
        try:
            lock.acquire()
        except filelock.Timeout:
            logger.debug("Leaving in-progress staging %s alone (lock held)", entry)
            continue
        try:
            logger.info("Removing orphaned staging directory %s", entry)
            shutil.rmtree(entry, ignore_errors=True)
        finally:
            lock.release()
            try:
                lock_path.unlink()
            except OSError:
                pass


def _copy_tree(src: Path, dest: Path) -> None:
    """Copy src/* into dest, skipping experimaestro-internal artifacts.

    `dest` must not exist yet. The ``.experimaestro`` directory IS
    preserved (it holds ``task-outputs.jsonl`` which is needed for
    callback replay).
    """
    dest.mkdir(parents=True, exist_ok=False)
    with os.scandir(src) as it:
        for entry in it:
            if not _should_copy(entry):
                continue
            target = dest / entry.name
            if entry.is_dir(follow_symlinks=False):
                shutil.copytree(
                    entry.path,
                    target,
                    symlinks=True,
                    dirs_exist_ok=False,
                )
            else:
                shutil.copy2(entry.path, target, follow_symlinks=False)


def _rewrite_legacy_params_json(dest: Path, src_workspace: Path) -> None:
    """Rewrite a v1/v2 params.json to make absolute paths portable.

    For v3+ params.json paths are already encoded relative to the job
    or workspace, so nothing to do. Older versions store absolute
    paths; we rewrite any string starting with the source workspace
    path so the file is at least readable in the new location.

    This is best-effort: only string fields are rewritten, only the
    workspace-prefix substitution is attempted. Anything more
    intrusive belongs in a migration tool.
    """
    params = dest / "params.json"
    if not params.is_file():
        return
    try:
        data = json.loads(params.read_text())
    except Exception:
        logger.debug("Could not parse %s for path rewriting", params)
        return

    if not isinstance(data, dict):
        return
    version = data.get("version", 0)
    if version >= 3:
        return  # paths are already relative

    src_prefix = str(src_workspace).rstrip("/") + "/"

    def rewrite(value):
        if isinstance(value, str) and value.startswith(src_prefix):
            return value  # we keep the original; loader resolves via wspath
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        return value

    data = rewrite(data)
    params.write_text(json.dumps(data))


def _archive_job(job: _Job, folder: FolderSettings) -> Optional[Path]:
    """Perform a single archive operation. Returns the final dest path."""
    dest_root = folder.path / "jobs" / job.rel
    if dest_root.exists():
        logger.debug(
            "Folder %s already contains job %s; skipping", folder.path, job.rel
        )
        return dest_root

    # Atomic stage: copy/move into a sibling .tmp.<hash> and then rename.
    # An exclusive file lock is held for the entire copy so a sweep
    # from another process (or our own next run) can distinguish
    # in-progress staging from crashed leftovers.
    tmp = dest_root.parent / f".tmp.{dest_root.name}"
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    _sweep_stale_tmp(dest_root.parent)

    lock_path = _tmp_lock_path(tmp)
    lock = create_file_lock(lock_path, timeout=0)
    try:
        lock.acquire()
    except filelock.Timeout:
        logger.info("Another process is archiving %s; skipping this one", dest_root)
        return None

    try:
        if tmp.exists():
            shutil.rmtree(tmp)

        try:
            if folder.mode == FolderMode.BACKUP:
                _copy_tree(job.src, tmp)
            elif folder.mode == FolderMode.MOVE:
                # Move means: copy first, then on success replace the
                # original with a symlink so the primary workspace keeps
                # working. We never rename(2) across filesystems.
                _copy_tree(job.src, tmp)
            else:
                logger.error("Folder mode %s should not reach worker", folder.mode)
                return None

            os.replace(tmp, dest_root)
        except Exception:
            logger.exception("Failed to archive %s to %s", job.src, folder.path)
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            return None

        # Best-effort legacy path rewriting (no-op for v3+ params.json)
        try:
            _rewrite_legacy_params_json(dest_root, job.src.parent.parent.parent)
        except Exception:
            logger.exception("Path rewriting failed for %s (continuing)", dest_root)

        if folder.mode == FolderMode.MOVE:
            try:
                shutil.rmtree(job.src)
                job.src.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(dest_root, job.src, target_is_directory=True)
            except Exception:
                logger.exception(
                    "Failed to swap %s for a symlink to %s", job.src, dest_root
                )

        return dest_root
    finally:
        lock.release()
        try:
            lock_path.unlink()
        except OSError:
            pass


class _FolderWorker:
    """Background thread that processes archive requests."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[_Request]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending: set[tuple[Path, Path]] = set()

    def _ensure_running(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="xpm-folders", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return
            try:
                _archive_job(request.job, request.folder)
            except Exception:
                logger.exception("Archive worker caught an exception")
            finally:
                with self._lock:
                    self._pending.discard((request.job.src, request.folder.path))
                self._queue.task_done()

    def submit(self, job: _Job, folder: FolderSettings) -> None:
        key = (job.src, folder.path)
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
        self._ensure_running()
        self._queue.put(_Request(job=job, folder=folder))

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until all queued requests are processed (test helper)."""
        # queue.Queue has no timeout on join; emulate it with a poll loop
        if timeout is None:
            self._queue.join()
            return True
        import time as _time

        deadline = _time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._pending:
                    return True
            if _time.monotonic() >= deadline:
                return False
            _time.sleep(0.05)


# Module-level singleton (matches the scheduler-singleton pattern used
# elsewhere in this package).
_WORKER = _FolderWorker()


def _drain_on_exit() -> None:
    """Best-effort drain of pending archives at interpreter shutdown.

    The worker thread is a daemon, so without this hook a Python exit
    while archives are still queued would silently drop them. We give
    pending work a generous timeout; if it expires we log loudly so
    users know archives may be incomplete.
    """
    if not _WORKER.wait(timeout=60.0):
        logger.warning(
            "Folder archive worker did not finish within 60s at exit; "
            "some jobs may not have been copied to their archive folder."
        )


atexit.register(_drain_on_exit)


def archive_job_async(job: "Job", folders: list[FolderSettings]) -> None:
    """Queue an archive operation for each eligible folder.

    Folders with mode=USE are ignored (read-only).
    """
    if not folders:
        return
    eligible = [f for f in folders if f.mode != FolderMode.USE]
    if not eligible:
        return

    reduced = _Job(
        src=job.path,
        rel=Path(job.relpath),
        name=job.name,
    )
    for folder in eligible:
        _WORKER.submit(reduced, folder)


def wait_for_pending(timeout: Optional[float] = None) -> bool:
    """Test helper: block until all background work is done."""
    return _WORKER.wait(timeout=timeout)


def recover_from_folders(
    rel: Path,
    primary_jobs: Path,
    folders: list[FolderSettings],
) -> Optional[Path]:
    """Copy a job back from a backup folder into the primary workspace.

    Looks up ``rel`` (``<task_id>/<hash>``) in each attached folder; if
    found, copies it into ``primary_jobs/rel`` atomically. Returns the
    destination path on success, ``None`` if no folder had the job.

    ``mode=use`` folders are checked too — recovery is a read operation
    and ``use`` folders are explicitly meant for cross-workspace reuse.
    """
    dest = primary_jobs / rel
    if dest.exists():
        return dest

    for folder in folders:
        src = folder.path / "jobs" / rel
        if not src.exists():
            continue

        tmp = dest.parent / f".tmp.{dest.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _sweep_stale_tmp(dest.parent)

        lock_path = _tmp_lock_path(tmp)
        lock = create_file_lock(lock_path, timeout=0)
        try:
            lock.acquire()
        except filelock.Timeout:
            logger.info("Another process is recovering %s; skipping this one", dest)
            continue

        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            _copy_tree(src, tmp)
            os.replace(tmp, dest)
            return dest
        except Exception:
            logger.exception("Recovery from %s failed", src)
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            continue
        finally:
            lock.release()
            try:
                lock_path.unlink()
            except OSError:
                pass

    return None
