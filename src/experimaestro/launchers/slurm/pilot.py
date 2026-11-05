"""Pilot allocations: several jobs run one after another in one SLURM job (beta)

Instead of one ``sbatch`` per job, a pilot SLURM job runs a worker (``python
-m experimaestro.launchers.slurm.pilot``) that executes queued jobs until its
time limit is near or it has been idle for a while. Everything goes through
files in a pool directory of the workspace (shared file system), so that
pilots, schedulers and monitors only need to see the same directory::

    <workspace>/pilots/<pool key>/
        queue/<entry>.json        job to run (written by the scheduler)
        entries/<entry>.claim     taken by a pilot: which one, until when (worker),
                                  or cancelled before it started (scheduler)
        entries/<entry>.exit      how it ended (worker, or scheduler if cancelled)
        entries/<entry>.cancel    request to stop it (scheduler)
        pilots/<pilot>.json       the SLURM job of the pilot (scheduler)
        pilots/<pilot>.worker     worker state: end time, current job (worker)
        pilots/<pilot>.stop       request to stop taking jobs (CLI)
        pilots/<pilot>.stopped    the worker does not take jobs anymore (worker)
        holders/<host>-<pid>.json lease of a scheduler with jobs in the pool

A job is taken by exclusively creating its claim file (with a hard link, which
is reliable on NFS, unlike renames), so that a job is run by one pilot only and
cannot be both cancelled and started.

Pilots of a pool share the same SLURM options (except the time limit), so any
of them can run any job of the pool, whichever experiment submitted it. An idle
pilot stops after ``idle_timeout``, or ``linger`` seconds after no scheduler
holds a lease on the pool (e.g. when the experiments have ended).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experimaestro.connectors import (
    Connector,
    Process,
    ProcessState,
    Redirect,
    RedirectType,
)
from experimaestro.utils.atomic import atomic_create, atomic_write_json

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import JobState

    from .base import SlurmJobState, SlurmLauncher

logger = logging.getLogger("xpm.slurm.pilot")

#: Environment variable giving a job run by a pilot its deadline (UNIX time)
DEADLINE_ENV = "XPM_JOB_DEADLINE"

#: Entries that ended more than this many seconds ago are removed
ENTRY_RETENTION = 24 * 3600


def parse_slurm_duration(value: str | int | float) -> int:
    """Parses a SLURM ``--time`` value into seconds

    A number is a number of minutes (as for SLURM); strings follow the SLURM
    formats: ``minutes``, ``minutes:seconds``, ``hours:minutes:seconds``,
    ``days-hours``, ``days-hours:minutes`` and ``days-hours:minutes:seconds``.
    """
    if isinstance(value, (int, float)):
        return int(value * 60)

    text = value.strip()
    days = 0
    if "-" in text:
        days_text, text = text.split("-", 1)
        days = int(days_text)
        parts = [int(p) for p in text.split(":")]
        hours, minutes, seconds = (parts + [0, 0])[:3]
    else:
        parts = [int(p) for p in text.split(":")]
        match len(parts):
            case 1:
                hours, minutes, seconds = 0, parts[0], 0
            case 2:
                hours, (minutes, seconds) = 0, parts
            case 3:
                hours, minutes, seconds = parts
            case _:
                raise ValueError(f"Invalid SLURM time: {value}")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


@dataclass
class PilotOptions:
    """Runs the jobs of a SLURM launcher in pilot allocations (beta)

    Jobs whose duration (the ``time`` SLURM option) fits are queued, and run one
    after another by pilot SLURM jobs requesting the same resources for
    ``time``; the other jobs are submitted with ``sbatch`` as usual.
    """

    time: str | int = "24:00:00"
    """Time limit of a pilot (SLURM format, or seconds if an integer)"""

    idle_timeout: float = 300
    """A pilot without any job to run stops after this many seconds"""

    max_pilots: int | None = None
    """Maximum number of pilots of the pool at a given time (pending or
    running); jobs then wait for a pilot to be free"""

    margin: float = 60
    """A job is run by a pilot only if it has this many seconds left after the
    job time limit"""

    kill_wait: float = 30
    """Seconds between SIGTERM and SIGKILL when a job is stopped"""

    poll_interval: float = 2
    """Seconds between two checks of the queue (by workers and schedulers)"""

    linger: float = 60
    """An idle pilot stops when no scheduler has had jobs in the pool for this
    many seconds (e.g. when the experiments have ended)"""

    @property
    def seconds(self) -> int:
        """Time limit of a pilot, in seconds"""
        if isinstance(self.time, int):
            return self.time
        return parse_slurm_duration(self.time)


@dataclass
class EntryStatus:
    """Status of a queued job, read from the pool files"""

    state: ProcessState
    code: int | None = None
    #: Information written by the worker when the job ended
    exit: dict[str, Any] = field(default_factory=dict)
    #: Information written by the worker when the job was claimed
    claim: dict[str, Any] = field(default_factory=dict)
    #: SLURM state of the pilot when the job was lost with its pilot
    lost_with: str | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class PoolDirectory:
    """Paths of a pool directory (shared by the scheduler and the workers)"""

    def __init__(self, path: Path):
        self.path = path
        self.queue = path / "queue"
        self.entries = path / "entries"
        self.pilots = path / "pilots"
        self.holders = path / "holders"

    def mkdirs(self):
        for path in (self.queue, self.entries, self.pilots, self.holders):
            path.mkdir(parents=True, exist_ok=True)

    def pending(self) -> list[Path]:
        """Entries not taken by a pilot yet, oldest first"""
        try:
            paths = sorted(self.queue.glob("*.json"))
        except OSError:
            return []
        return [p for p in paths if not self.claim_path(p.stem).exists()]

    def held(self, linger: float) -> bool:
        """Whether a scheduler has renewed its lease in the last ``linger``
        seconds"""
        limit = time.time() - linger
        try:
            paths = list(self.holders.glob("*.json"))
        except OSError:
            return False
        for path in paths:
            if (info := _read_json(path)) is not None and info["time"] > limit:
                return True
        return False

    def claim_path(self, name: str) -> Path:
        return self.entries / f"{name}.claim"

    def exit_path(self, name: str) -> Path:
        return self.entries / f"{name}.exit"

    def cancel_path(self, name: str) -> Path:
        return self.entries / f"{name}.cancel"


# --- Scheduler side


class PilotPool:
    """Pilots sharing the same SLURM options, seen from a scheduler

    The pool queues jobs and submits pilots when no pilot (pending or running
    with enough time left) can take the queued jobs.
    """

    POOLS: dict[Path, "PilotPool"] = {}
    POOLS_LOCK = threading.Lock()

    def __init__(self, launcher: "SlurmLauncher", directory: Path):
        assert launcher.pilot is not None
        self.launcher = launcher
        self.pilot = launcher.pilot
        self.directory = PoolDirectory(directory)
        self.directory.mkdirs()
        self._lock = threading.Lock()
        self._last_check = 0.0

    @staticmethod
    def key(launcher: "SlurmLauncher") -> str:
        """Pool key: the SLURM options (except time) and the pilot options"""
        assert launcher.pilot is not None
        options = {
            k: v
            for k, v in asdict(launcher.options).items()
            if k != "time" and v is not None
        }
        data = {
            "binpath": str(launcher.binpath),
            "options": options,
            "pilot": asdict(launcher.pilot),
        }
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    @staticmethod
    def get(launcher: "SlurmLauncher", workspace: Path) -> "PilotPool":
        path = workspace / "pilots" / PilotPool.key(launcher)
        with PilotPool.POOLS_LOCK:
            pool = PilotPool.POOLS.get(path)
            if pool is None:
                pool = PilotPool(launcher, path)
                PilotPool.POOLS[path] = pool
            return pool

    def pilot_launcher(self) -> "SlurmLauncher":
        """The launcher used to submit pilots (and watch them)"""
        from .base import SlurmOptions

        return self.launcher.config(
            time=SlurmOptions.format_time(self.pilot.seconds), pilot=None
        )

    # --- Queue

    def submit(
        self,
        command: list[str],
        *,
        cwd: Path,
        stdout: Path | None,
        stderr: Path | None,
        duration: int,
    ) -> "PilotProcess":
        """Queues a job and returns its process"""
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        entry = {
            "command": [str(c) for c in command],
            "cwd": str(cwd),
            "stdout": str(stdout) if stdout else None,
            "stderr": str(stderr) if stderr else None,
            "time": duration,
            "submitted": time.time(),
        }
        atomic_write_json(self.directory.queue / f"{name}.json", entry, fsync=True)
        logger.info("Queued job %s in pilot pool %s", name, self.directory.path)
        process = PilotProcess(self, name)
        self.ensure_capacity(force=True)
        return process

    def status(self, name: str) -> EntryStatus:
        """Returns the status of an entry"""
        d = self.directory
        if (exit_info := _read_json(d.exit_path(name))) is not None:
            code = exit_info.get("code")
            return EntryStatus(
                ProcessState.DONE if code == 0 else ProcessState.ERROR,
                code=code if code is not None else 1,
                exit=exit_info,
                claim=_read_json(d.claim_path(name)) or {},
            )

        claim = _read_json(d.claim_path(name))
        if claim is None:
            if (d.queue / f"{name}.json").exists():
                return EntryStatus(ProcessState.SCHEDULED)
            # Unknown (e.g. removed): the job markers will tell
            return EntryStatus(ProcessState.ERROR, code=None)

        if claim.get("cancelled"):
            # Cancelled before it started (the exit file follows)
            return EntryStatus(
                ProcessState.ERROR, code=1, exit={"cancelled": True}, claim=claim
            )

        # Running: check that the pilot is still alive
        pilot_state = self._slurm_state(claim.get("slurm_id"))
        if pilot_state is not None and pilot_state.finished():
            # The worker writes the exit file before ending: check again
            if _read_json(d.exit_path(name)) is not None:
                return self.status(name)
            return EntryStatus(
                ProcessState.ERROR,
                code=1,
                claim=claim,
                lost_with=pilot_state.slurm_state,
            )
        return EntryStatus(ProcessState.RUNNING, claim=claim)

    def cancel(self, name: str):
        """Cancels a queued or running job"""
        d = self.directory
        now = time.time()
        if atomic_create(
            d.claim_path(name), json.dumps({"cancelled": True, "time": now})
        ):
            # Not started: no pilot can take it anymore
            atomic_write_json(
                d.exit_path(name), {"code": None, "cancelled": True, "end": now}
            )
            logger.info("Removed job %s from the pilot queue", name)
        else:
            # Taken by a pilot, which will stop it
            atomic_write_json(d.cancel_path(name), {"time": now})
            logger.info("Asked the pilot to stop job %s", name)

    # --- Pilots

    def _slurm_state(self, slurm_id: str | None) -> "SlurmJobState | None":
        """SLURM state of a pilot job (None if unknown)"""
        from .base import SlurmProcessWatcher

        if slurm_id is None:
            return None
        with SlurmProcessWatcher.get(self.launcher) as watcher:
            return watcher.getjob(slurm_id, wait_for_update=False)

    def _pilots(self) -> list[dict[str, Any]]:
        """Pilots that are pending or running and take jobs"""
        d = self.directory
        pilots = []
        try:
            paths = list(d.pilots.glob("*.json"))
        except OSError:
            return []

        for path in paths:
            pilot_id = path.stem
            if (d.pilots / f"{pilot_id}.stopped").exists():
                continue
            info = _read_json(path)
            if info is None:
                continue
            state = self._slurm_state(info.get("slurm_id"))
            if state is not None and state.finished():
                continue
            worker = _read_json(d.pilots / f"{pilot_id}.worker")
            if worker is not None and worker["end"] < time.time():
                # Past its time limit (SLURM may not know the job anymore)
                continue
            info["id"] = pilot_id
            info["worker"] = worker
            pilots.append(info)
        return pilots

    def ensure_capacity(self, *, force: bool = False):
        """Submits pilots for the queued jobs that no pilot can take

        Called at most every ``poll_interval`` seconds (unless forced) by the
        processes waiting for their job.
        """
        if not force and time.time() - self._last_check < self.pilot.poll_interval:
            return
        if not self._lock.acquire(blocking=force):
            return
        try:
            self._last_check = time.time()
            self._renew_lease()
            self._ensure_capacity()
            self._cleanup()
        except Exception:
            logger.exception("Error while checking the pilots")
        finally:
            self._lock.release()

    def _renew_lease(self):
        """Tells the pilots that this scheduler still has jobs in the pool"""
        path = self.directory.holders / f"{socket.gethostname()}-{os.getpid()}.json"
        atomic_write_json(path, {"time": time.time()})

    def _ensure_capacity(self):
        if not self.directory.pending():
            return

        # The pilots are read before the queue: a worker claims a job before
        # saying it is busy with it, so a job seen as queued is never also
        # counted as keeping a pilot busy (which would submit a useless pilot)
        pilots = self._pilots()
        now = time.time()
        queued = []
        for path in self.directory.pending():
            if (entry := _read_json(path)) is not None:
                queued.append(entry)
        if not queued:
            return

        # Remaining time of the pilots that can take a job now (or when they
        # start, for pending ones)
        available = []
        for pilot in pilots:
            worker = pilot["worker"]
            if worker is None:
                available.append(self.pilot.seconds)
            elif worker.get("busy") is None:
                available.append(worker["end"] - now)
        available.sort()

        # Greedy matching of the longest jobs first
        missing = 0
        for entry in sorted(queued, key=lambda e: -e["time"]):
            needed = entry["time"] + self.pilot.margin
            for ix, remaining in enumerate(available):
                if remaining >= needed:
                    del available[ix]
                    break
            else:
                missing += 1

        if self.pilot.max_pilots is not None:
            missing = min(missing, self.pilot.max_pilots - len(pilots))
        for _ in range(missing):
            self.submit_pilot()

    def _cleanup(self):
        """Removes the files of entries that ended long ago, and old leases"""
        d = self.directory
        limit = time.time() - ENTRY_RETENTION
        for path in d.entries.glob("*.exit"):
            try:
                if path.stat().st_mtime > limit:
                    continue
                name = path.stem
                for other in (
                    d.queue / f"{name}.json",
                    d.claim_path(name),
                    d.cancel_path(name),
                    path,
                ):
                    other.unlink(missing_ok=True)
            except OSError:
                pass
        for path in d.holders.glob("*.json"):
            try:
                if path.stat().st_mtime < limit:
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    def submit_pilot(self) -> str:
        """Submits a new pilot SLURM job and returns its ID"""
        from shlex import quote

        from .base import BatchSlurmProcess, SlurmProcessBuilder

        d = self.directory
        pilot_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        launcher = self.pilot_launcher()
        script = d.pilots / f"{pilot_id}.sh"
        python = os.environ.get("XPM_PILOT_PYTHON", sys.executable)
        with script.open("wt") as out:
            out.write("#!/bin/sh\n\n")
            out.write(f"#SBATCH --chdir={quote(str(d.path))}\n")
            out.write(f"#SBATCH --output={quote(str(d.pilots / f'{pilot_id}.out'))}\n")
            out.write(f"#SBATCH --error={quote(str(d.pilots / f'{pilot_id}.err'))}\n")
            out.write("#SBATCH --job-name=xpm-pilot\n")
            for arg in launcher.options.args():
                out.write(f"#SBATCH {arg}\n")
            command = [
                python,
                "-m",
                "experimaestro.launchers.slurm.pilot",
                str(d.path),
                pilot_id,
                f"--time={self.pilot.seconds}",
                f"--idle-timeout={self.pilot.idle_timeout}",
                f"--margin={self.pilot.margin}",
                f"--kill-wait={self.pilot.kill_wait}",
                f"--poll-interval={self.pilot.poll_interval}",
                f"--linger={self.pilot.linger}",
            ]
            out.write("\nexec " + " ".join(quote(c) for c in command) + "\n")
        script.chmod(0o755)

        builder = SlurmProcessBuilder(launcher)
        builder.command = [str(script)]
        process = builder.start(True)
        assert isinstance(process, BatchSlurmProcess)
        atomic_write_json(
            d.pilots / f"{pilot_id}.json",
            {"slurm_id": process.jobid, "submitted": time.time()},
            fsync=True,
        )
        logger.info(
            "Submitted pilot %s (SLURM job %s) for %s",
            pilot_id,
            process.jobid,
            d.path,
        )
        return pilot_id

    def spec(self) -> dict[str, Any]:
        """Serializable description of the pool (to rebuild it)"""
        return {
            "pool": str(self.directory.path),
            "options": [list(kv) for kv in self.launcher.key],
            "slurm": {
                k: v for k, v in asdict(self.launcher.options).items() if v is not None
            },
            "pilot": asdict(self.pilot),
        }

    @staticmethod
    def fromspec(connector: Connector, spec: dict[str, Any]) -> "PilotPool":
        from .base import SlurmLauncher, SlurmOptions

        path = Path(spec["pool"])
        with PilotPool.POOLS_LOCK:
            if (pool := PilotPool.POOLS.get(path)) is not None:
                return pool
            options = {k: v for k, v in spec.get("options", ())}
            launcher = SlurmLauncher(
                connector=connector,
                options=SlurmOptions(**spec["slurm"]),
                pilot=PilotOptions(**spec["pilot"]),
                **options,
            )
            pool = PilotPool(launcher, path)
            PilotPool.POOLS[path] = pool
            return pool


class PilotProcess(Process):
    """A job queued in a pilot pool"""

    # Everything is in the (shared) pool directory
    host_dependent = False

    def __init__(self, pool: PilotPool, name: str):
        self.pool = pool
        self.name = name
        self._status: EntryStatus | None = None

    def __repr__(self):
        return f"slurm-pilot:{self.name}"

    @classmethod
    def describe(cls, definition: dict[str, Any]) -> str | None:
        """Where the job runs: its pilot, or waiting for one"""
        from datetime import datetime

        d = PoolDirectory(Path(definition["pool"]))
        name = definition["pid"]
        pool = d.path.name
        claim = _read_json(d.claim_path(name))
        if claim is None:
            if (d.queue / f"{name}.json").exists():
                return f"waiting for a pilot (pool {pool})"
            return None
        if claim.get("cancelled"):
            return f"cancelled before it started (pool {pool})"

        where = f"pilot SLURM job {claim['slurm_id']} on {claim['host']}"
        if d.exit_path(name).exists():
            return f"ran in {where} (pool {pool})"
        deadline = datetime.fromtimestamp(claim["deadline"])
        return (
            f"running in {where}, time limit at {deadline:%Y-%m-%d %H:%M} (pool {pool})"
        )

    def tospec(self) -> dict[str, Any]:
        return {"type": "slurm-pilot", "pid": self.name, **self.pool.spec()}

    @classmethod
    def fromspec(cls, connector: Connector, spec: dict[str, Any]):
        pool = PilotPool.fromspec(connector, spec)
        process = PilotProcess(pool, spec["pid"])
        if process._poll().state.finished:
            return None
        return process

    def _poll(self, manage: bool = False) -> EntryStatus:
        """Reads the status of the job

        :param manage: submit pilots if needed (only done by the processes
            waiting for their job, not by monitors)
        """
        if manage:
            self.pool.ensure_capacity()
        self._status = self.pool.status(self.name)
        return self._status

    def wait(self) -> int:
        from .base import SlurmProcessWatcher

        with SlurmProcessWatcher.get(self.pool.launcher):
            while not (status := self._poll(True)).state.finished:
                time.sleep(self.pool.pilot.poll_interval)
        return status.code if status.code is not None else 1

    async def aio_wait(self) -> int:
        code = await self.aio_code()
        return code if code is not None else 1

    async def aio_code(self) -> int | None:
        from .base import SlurmProcessWatcher

        # Keeps the SLURM watcher alive while waiting
        with SlurmProcessWatcher.get(self.pool.launcher):
            while not (
                status := await asyncio.to_thread(self._poll, True)
            ).state.finished:
                await asyncio.sleep(self.pool.pilot.poll_interval)
        logger.debug("Pilot job %s ended: %s", self.name, status)
        return status.code

    async def aio_state(self, timeout: float | None = None) -> ProcessState:
        return (await asyncio.to_thread(self._poll)).state

    def kill(self):
        self.pool.cancel(self.name)

    def get_job_state(self, code: int) -> "JobState":
        from experimaestro.scheduler.jobs import (
            JobFailureStatus,
            JobState,
            JobStateError,
        )

        if code == 0:
            return JobState.DONE

        status = self._status or self.pool.status(self.name)
        if status.exit.get("timeout"):
            return JobStateError(
                JobFailureStatus.TIMEOUT, message="Job time limit reached (pilot)"
            )
        if status.exit.get("cancelled"):
            return JobStateError(
                JobFailureStatus.CANCELLED, message="Cancelled before it started"
            )
        if status.exit.get("pilot_ended") or status.lost_with is not None:
            pilot_state = status.lost_with or status.exit.get("pilot_state")
            slurm_id = status.claim.get("slurm_id")
            if pilot_state == "TIMEOUT":
                return JobStateError(
                    JobFailureStatus.TIMEOUT,
                    message=f"Pilot SLURM job {slurm_id} time limit reached",
                )
            return JobStateError(
                JobFailureStatus.FAILED,
                message=f"Pilot SLURM job {slurm_id} ended"
                + (f" ({pilot_state})" if pilot_state else ""),
            )
        return JobState.ERROR


def pilot_duration(launcher: "SlurmLauncher") -> int | None:
    """Duration of a job (from the ``time`` option), or None if not set"""
    if launcher.options.time is None:
        return None
    return parse_slurm_duration(launcher.options.time)


def start_in_pilot(
    launcher: "SlurmLauncher",
    command: list[str],
    *,
    workspace: Path | None,
    stdout: Redirect,
    stderr: Redirect,
) -> PilotProcess | None:
    """Queues a job script in a pilot pool, if the job fits in a pilot

    Returns None if the job must be submitted with ``sbatch``.
    """
    assert launcher.pilot is not None
    duration = pilot_duration(launcher)
    if workspace is None or duration is None:
        logger.info("No time limit for the job: not using a pilot")
        return None
    if duration + launcher.pilot.margin > launcher.pilot.seconds:
        logger.info(
            "Job time limit (%ss) too long for a pilot (%ss): using sbatch",
            duration,
            launcher.pilot.seconds,
        )
        return None

    def path(redirect: Redirect) -> Path | None:
        if redirect.type == RedirectType.FILE:
            return Path(redirect.path)
        return None

    pool = PilotPool.get(launcher, workspace)
    # The job script runs in its directory (as with the #SBATCH --chdir)
    return pool.submit(
        command,
        cwd=Path(command[-1]).parent,
        stdout=path(stdout),
        stderr=path(stderr),
        duration=duration,
    )


# --- Listing (CLI)


@dataclass
class PilotInfo:
    """A pilot, as seen from the pool files"""

    pool: str
    id: str
    slurm_id: str | None
    #: pending (in the SLURM queue), running, idle, stopping (asked to stop),
    #: stopped, or ended (time limit passed without the worker saying so)
    status: str
    #: Time limit of the pilot (UNIX time), once started
    end: float | None = None
    #: Directory of the job being run
    job: str | None = None


@dataclass
class QueuedJob:
    """A job waiting for a pilot"""

    pool: str
    job: str
    time: int


def list_pilots(workspace: Path) -> tuple[list[PilotInfo], list[QueuedJob]]:
    """Pilots and waiting jobs of a workspace (read from the pool files only:
    a pilot killed before it started is shown as pending)"""
    pilots: list[PilotInfo] = []
    queued: list[QueuedJob] = []
    now = time.time()
    root = workspace / "pilots"
    if not root.is_dir():
        return pilots, queued

    for pool_path in sorted(root.iterdir()):
        d = PoolDirectory(pool_path)
        pool = pool_path.name
        for path in d.pending():
            if (entry := _read_json(path)) is not None:
                queued.append(QueuedJob(pool, str(Path(entry["cwd"])), entry["time"]))

        for path in sorted(d.pilots.glob("*.json")):
            pilot_id = path.stem
            info = _read_json(path) or {}
            worker = _read_json(d.pilots / f"{pilot_id}.worker")
            job = None
            if (d.pilots / f"{pilot_id}.stopped").exists():
                status = "stopped"
            elif worker is None:
                status = "pending"
            elif worker["end"] < now:
                status = "ended"
            else:
                busy = worker.get("busy")
                if busy is not None and (entry := _read_json(d.queue / f"{busy}.json")):
                    job = entry["cwd"]
                if (d.pilots / f"{pilot_id}.stop").exists():
                    status = "stopping"
                else:
                    status = "running" if busy else "idle"
            pilots.append(
                PilotInfo(
                    pool,
                    pilot_id,
                    info.get("slurm_id"),
                    status,
                    end=worker["end"] if worker else None,
                    job=job,
                )
            )
    return pilots, queued


def stop_pilots(workspace: Path) -> list[PilotInfo]:
    """Asks the pending and running pilots to stop taking jobs (a running job
    is not stopped); returns the pilots asked"""
    pilots, _ = list_pilots(workspace)
    asked = []
    for pilot in pilots:
        if pilot.status in ("pending", "running", "idle"):
            path = workspace / "pilots" / pilot.pool / "pilots" / f"{pilot.id}.stop"
            atomic_write_json(path, {"time": time.time()})
            asked.append(pilot)
    return asked


# --- Worker side (runs in the pilot SLURM job)


class Worker:
    """Runs the queued jobs of a pool, one after another"""

    def __init__(
        self,
        directory: Path,
        pilot_id: str,
        *,
        time_limit: float,
        idle_timeout: float,
        margin: float,
        kill_wait: float,
        poll_interval: float,
        linger: float,
    ):
        self.directory = PoolDirectory(directory)
        self.pilot_id = pilot_id
        self.start = time.time()
        self.end = self.start + time_limit
        self.idle_timeout = idle_timeout
        self.margin = margin
        self.kill_wait = kill_wait
        self.poll_interval = poll_interval
        self.linger = linger
        self.slurm_id = os.environ.get("SLURM_JOB_ID")
        self.process: subprocess.Popen | None = None
        self.terminated = False

    @property
    def worker_path(self) -> Path:
        return self.directory.pilots / f"{self.pilot_id}.worker"

    def write_state(self, busy: str | None):
        atomic_write_json(
            self.worker_path,
            {
                "start": self.start,
                "end": self.end,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "slurm_id": self.slurm_id,
                "busy": busy,
            },
        )

    def claim(self) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        """Takes the oldest queued job that can finish in time

        Returns the entry name, the entry and the claim
        """
        now = time.time()
        remaining = self.end - now - self.margin
        for path in self.directory.pending():
            entry = _read_json(path)
            if entry is None or entry["time"] > remaining:
                continue
            claim = {
                "pilot": self.pilot_id,
                "slurm_id": self.slurm_id,
                "host": socket.gethostname(),
                "start": now,
                "deadline": min(now + entry["time"], self.end),
            }
            name = path.stem
            if atomic_create(self.directory.claim_path(name), json.dumps(claim)):
                return name, entry, claim
            # Taken by another pilot (or cancelled)
        return None

    def on_sigterm(self, signum, frame):
        """SLURM terminates the pilot: stops the current job"""
        logger.warning("Pilot terminated (signal %d)", signum)
        self.terminated = True
        if self.process is not None:
            self._signal(signal.SIGTERM)

    def _signal(self, signum: int):
        assert self.process is not None
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass

    def run_entry(self, name: str, entry: dict[str, Any], claim: dict[str, Any]):
        d = self.directory
        start, deadline = claim["start"], claim["deadline"]
        self.write_state(name)
        logger.info("Running job %s (deadline in %ds)", name, deadline - start)

        env = dict(os.environ)
        env[DEADLINE_ENV] = str(deadline)

        def output(key: str):
            return open(entry[key], "w") if entry.get(key) else subprocess.DEVNULL

        stdout, stderr = output("stdout"), output("stderr")
        try:
            self.process = subprocess.Popen(
                entry["command"],
                cwd=entry["cwd"],
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
        finally:
            for fp in (stdout, stderr):
                if fp is not subprocess.DEVNULL:
                    fp.close()

        timeout = cancelled = False
        term_time: float | None = None
        while (code := self.process.poll()) is None:
            now = time.time()
            if term_time is None:
                if d.cancel_path(name).exists():
                    cancelled = True
                elif now >= deadline:
                    timeout = True
                if cancelled or timeout or self.terminated:
                    logger.info("Stopping job %s", name)
                    self._signal(signal.SIGTERM)
                    term_time = now
            elif now - term_time > self.kill_wait:
                self._signal(signal.SIGKILL)
            time.sleep(min(0.2, self.poll_interval))

        logger.info("Job %s ended with code %s", name, code)
        info: dict[str, Any] = {"code": code, "start": start, "end": time.time()}
        if timeout:
            info["timeout"] = True
        if cancelled:
            info["cancelled_running"] = True
        if self.terminated and not (timeout or cancelled):
            info["pilot_ended"] = True
        atomic_write_json(d.exit_path(name), info, fsync=True)
        self.process = None

    def run(self):
        signal.signal(signal.SIGTERM, self.on_sigterm)
        self.directory.mkdirs()
        self.write_state(None)
        logger.info(
            "Pilot %s started (SLURM job %s, %ds)",
            self.pilot_id,
            self.slurm_id,
            self.end - self.start,
        )
        last_activity = time.time()
        stop_path = self.directory.pilots / f"{self.pilot_id}.stop"
        try:
            while not self.terminated and not stop_path.exists():
                if (claimed := self.claim()) is not None:
                    self.run_entry(*claimed)
                    self.write_state(None)
                    last_activity = time.time()
                    continue

                now = time.time()
                if now - last_activity > self.idle_timeout:
                    logger.info("Pilot idle for %ds: stopping", now - last_activity)
                    break
                if not self.directory.held(self.linger):
                    logger.info("No scheduler uses the pool anymore: stopping")
                    break
                if self.end - now <= self.margin:
                    logger.info("Not enough time left: stopping")
                    break
                time.sleep(self.poll_interval)
        finally:
            atomic_write_json(
                self.directory.pilots / f"{self.pilot_id}.stopped",
                {"time": time.time()},
                fsync=True,
            )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="experimaestro SLURM pilot worker")
    parser.add_argument("directory", type=Path)
    parser.add_argument("pilot_id")
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--idle-timeout", type=float, default=300)
    parser.add_argument("--margin", type=float, default=60)
    parser.add_argument("--kill-wait", type=float, default=30)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--linger", type=float, default=60)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(asctime)s [%(name)s] %(message)s",
    )
    Worker(
        args.directory,
        args.pilot_id,
        time_limit=args.time,
        idle_timeout=args.idle_timeout,
        margin=args.margin,
        kill_wait=args.kill_wait,
        poll_interval=args.poll_interval,
        linger=args.linger,
    ).run()


if __name__ == "__main__":
    main()
