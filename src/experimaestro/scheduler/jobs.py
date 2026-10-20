import asyncio
from collections import ChainMap
from datetime import datetime
from functools import cached_property
import itertools
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Set

import concurrent

from experimaestro.core.objects import Config, ConfigWalkContext, WatchedOutput
from experimaestro.notifications import LevelInformation

# from experimaestro.scheduler.base import Scheduler
from experimaestro.scheduler.dependencies import Dependency, Resource
from experimaestro.scheduler.workspace import RunMode, Workspace
from experimaestro.scheduler.transient import TransientMode
from experimaestro.scheduler.interfaces import (
    BaseJob,
    JobState,
    JobStateUnscheduled,
    JobStateWaiting,
    JobStateReady,
    JobStateScheduled,
    JobStateRunning,
    JobStateDone,
    JobStateError,
    JobFailureStatus,
)
from experimaestro.locking import Lock
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.launchers import Launcher
    from experimaestro.scheduler.experiment import experiment


# Re-export JobState for backward compatibility
__all__ = [
    "JobState",
    "JobStateUnscheduled",
    "JobStateWaiting",
    "JobStateReady",
    "JobStateScheduled",
    "JobStateRunning",
    "JobStateDone",
    "JobStateError",
    "JobFailureStatus",
    "Job",
    "TransientMode",
]


class JobLock(Lock):
    def __init__(self, job):
        super().__init__()
        self.job = job

    async def _aio_acquire(self):
        if self.job.scheduler_state != JobState.DONE:
            raise RuntimeError(f"Job {self.job.identifier} not done")

    async def _aio_release(self):
        pass


class JobDependency(Dependency):
    origin: "Job"

    def __init__(self, job, *, soft: bool = False):
        super().__init__(job)
        self.soft = soft  # Soft dependencies don't block execution

    async def ensure_started(self):
        """Ensure the dependency job is started.

        If the dependency is a transient job that was skipped (state is UNSCHEDULED),
        this will start it so it actually runs.
        """
        origin_job = self.origin
        if (
            origin_job.transient.is_transient
            # Several dependents may need the same transient job: only the
            # first one starts it (notstarted() stays true while WAITING)
            and not origin_job._needed_transient
            and origin_job.scheduler_state.notstarted()
        ):
            # Transient job was skipped but now is needed - start it
            from experimaestro.utils import logger

            logger.info(
                "Starting transient job %s (needed by dependent job)",
                origin_job.identifier,
            )
            # Mark as needed so aio_submit won't skip it again
            origin_job._needed_transient = True
            # Mark as WAITING and start the job via aio_submit
            # Use aio_submit (not aio_start) to properly handle all job lifecycle
            origin_job.set_scheduler_state(JobState.WAITING)
            origin_job.begin_run()
            if origin_job.scheduler is not None:
                # Create a new future for the job so aio_lock can wait on it
                origin_job._future = asyncio.ensure_future(
                    origin_job.scheduler.aio_submit(origin_job)
                )

    async def aio_lock(self, timeout: float = 0):
        """Acquire lock on job dependency by waiting for job to complete

        Args:
            timeout: Must be 0 (wait indefinitely) for job dependencies

        Raises:
            ValueError: If timeout is not 0
            RuntimeError: If the job has not been submitted or if it failed
        """
        # Soft dependencies don't block - return immediately
        if self.soft:
            # All good, return the lock as is
            return JobLock(self.origin)

        if timeout != 0:
            raise ValueError(
                "Job dependencies only support timeout=0 (wait indefinitely)"
            )

        # Ensure the dependency job is started (handles transient jobs)
        await self.ensure_started()

        # Wait for the job to finish
        if self.origin._future is None:
            raise RuntimeError(f"Job {self.origin} has no future - not submitted")
        await asyncio.wrap_future(self.origin._future)

        # Check if the job succeeded
        if self.origin.scheduler_state != JobState.DONE:
            raise RuntimeError(
                f"Dependency job {self.origin} failed with state {self.origin.scheduler_state} for {self.target}"
            )

        # Job succeeded, acquire and return the lock
        lock = JobLock(self.origin)
        await lock.aio_acquire()
        return lock


class Job(BaseJob, Resource):
    """A job is a resource that is produced by the execution of some code"""

    # Set by the scheduler
    _future: Optional["concurrent.futures.Future"]

    def __init__(
        self,
        config: Config,
        *,
        workspace: Workspace = None,
        launcher: "Launcher" = None,
        run_mode: RunMode = RunMode.NORMAL,
        max_retries: Optional[int] = None,
        transient: TransientMode = TransientMode.NONE,
        backup: Optional[bool] = None,
    ):
        from experimaestro.scheduler.base import Scheduler

        super().__init__()

        # Per-job backup override for the folders feature (beta).
        # None  -> use workspace defaults (archive to every BACKUP/MOVE folder)
        # True  -> force archival even if workspace has no folders configured
        # False -> opt out of all folder archival for this job
        self.backup: bool = True if backup is None else backup

        self.workspace = workspace or Workspace.CURRENT
        self.launcher = launcher or self.workspace.launcher if self.workspace else None

        if run_mode == RunMode.NORMAL:
            assert self.workspace is not None, "No experiment has been defined"
            assert self.launcher is not None, (
                "No launcher, and no default defined for the workspace %s" % workspace
            )

        self.type = config.__xpmtype__
        self.name = str(self.type.identifier).rsplit(".", 1)[-1]

        self.scheduler: Optional["Scheduler"] = None
        self.experiments: List["experiment"] = []  # Experiments this job belongs to
        self.config = config

        # Dependencies
        self.dependencies: Set[Dependency] = set()  # as target

        # Check if this is a resumable task
        from experimaestro.core.objects import ResumableTask

        self.resumable = isinstance(config, ResumableTask)

        # Retry configuration for resumable tasks
        # Use workspace setting if max_retries is not specified
        if max_retries is None and self.workspace:
            max_retries = self.workspace.workspace_settings.max_retries
        self.max_retries = max_retries if max_retries is not None else 3

        # Scheduler lifecycle state (independent from execution state)
        self._scheduler_state: JobState = JobState.UNSCHEDULED

        # Transient mode for intermediary tasks
        self.transient = transient if transient is not None else TransientMode.NONE
        # Flag set when a transient job's mode is merged to non-transient,
        # indicating the job should run even though it was originally transient
        self._needed_transient = False

        # Watched outputs (stored for deferred registration with scheduler)
        self.watched_outputs: List["WatchedOutput"] = list(
            config.__xpm__.watched_outputs
        )

        #: Resolved once the current run of the job is over *and* its hooks
        #: have run (see :meth:`begin_run`); experiments wait on it to exit
        self.settled: asyncio.Future | None = None

        # on_completed callbacks, with the experiment they were registered in
        self._on_completed: list[tuple["experiment | None", Callable[[], None]]] = []
        # Set once the callbacks have been taken: later ones are called directly
        self._on_completed_fired = False
        self._on_completed_lock = threading.Lock()

    def on_completed(
        self, callback: Callable[[], None], xp: "experiment | None" = None
    ):
        """Call ``callback`` once the job has completed successfully

        If it already has, ``callback`` is called right away.

        :param xp: the experiment whose callback executor runs the callback
            (defaults to the current one)
        """
        from experimaestro.scheduler.experiment import experiment

        with self._on_completed_lock:
            if not self._on_completed_fired:
                self._on_completed.append((xp or experiment.CURRENT, callback))
                return
        callback()

    def take_on_completed(self) -> list[tuple["experiment | None", Callable[[], None]]]:
        """Take the on_completed callbacks (called once the job is done)"""
        with self._on_completed_lock:
            self._on_completed_fired = True
            callbacks, self._on_completed = self._on_completed, []
        return callbacks

    def begin_run(self):
        """Start a new run: a fresh :attr:`settled` future, tracked by every
        experiment of the job

        Called on the scheduler loop each time the job is (re)submitted.
        """
        self.settled = asyncio.get_running_loop().create_future()
        for xp in self.experiments:
            xp.track(self.settled)

    def settle(self, settled: asyncio.Future | None = None):
        """Mark a run as settled (no-op if it already is)

        :param settled: the :attr:`settled` future of the run (defaults to the
            current one); a new run may have started since that one ended
        """
        settled = settled or self.settled
        if settled is not None and not settled.done():
            settled.set_result(self.scheduler_state)

    def watch_output(self, watched: "WatchedOutput"):
        """Add a watched output to this job.

        :param watched: A description of the watched output
        """
        self.watched_outputs.append(watched)

    async def aio_drain_task_outputs(self):
        """Wait until the dynamic outputs of the job have been processed

        Called by: Scheduler.aio_final_state(), once the process is over

        Reads the outputs file to its end (the watcher may lag behind) and
        waits for the callbacks of every experiment watching it.
        """
        expected = self.task_outputs_count() if self.state == JobState.DONE else None
        for xp in self.experiments:
            if reader := xp.task_outputs_reader(self):
                await reader.drain(expected)

    def task_outputs_count(self) -> int | None:
        """Number of dynamic outputs the task wrote, as recorded in ``.done``

        Returns None if unknown (no or empty ``.done``, as written by older
        versions or by :meth:`mark_done`).
        """
        try:
            content = self.donepath.read_text()
        except OSError:
            return None
        if not content.strip():
            return None
        try:
            return json.loads(content)["task_outputs"]
        except (ValueError, KeyError, TypeError):
            logger.warning("Unexpected content in %s", self.donepath)
            return None

    def __str__(self):
        return "Job[{}/{}]".format(self.task_id, self.identifier[:8])

    def wait(self) -> JobState:
        assert self._future, "Cannot wait a not submitted job"
        return self._future.result()

    @property
    def scheduler_state(self) -> JobState:
        """Scheduler lifecycle state (independent from execution state)"""
        return self._scheduler_state

    def set_scheduler_state(
        self,
        new_state: JobState,
    ):
        """Set the scheduler lifecycle state.

        Updates experiment statistics and notifies listeners.
        This does NOT update the execution state (_state) — that is updated
        only by load_from_disk() and apply_event().

        :param new_state: The new scheduler state
        """
        old_state = self._scheduler_state

        # Nothing changed
        if old_state == new_state:
            return

        self._scheduler_state = new_state

        # Helper to determine if a state should be "counted" in unfinishedJobs
        # A job is counted from UNSCHEDULED (registered) until finished or TRANSIENT.
        # UNSCHEDULED is counted because add_job increments unfinishedJobs at
        # registration time, before aio_submit processes the job asynchronously.
        def is_counted(state):
            return not state.finished() and state != JobState.TRANSIENT

        # Update experiment statistics based on state transition
        for xp in self.experiments:
            # Handle transitions in/out of "counted" state
            if is_counted(new_state) and not is_counted(old_state):
                # Job is now being tracked (new submission or resubmit)
                xp.unfinishedJobs += 1
                logger.debug(
                    "Job %s submitted, unfinished jobs for %s: %d",
                    self.identifier[:8],
                    xp.name,
                    xp.unfinishedJobs,
                )
            elif not is_counted(new_state) and is_counted(old_state):
                # Job is no longer being tracked (finished)
                xp.unfinishedJobs -= 1
                logger.debug(
                    "Job %s finished, unfinished jobs for %s: %d",
                    self.identifier[:8],
                    xp.name,
                    xp.unfinishedJobs,
                )

            # Handle error state
            if new_state.is_error() and not old_state.is_error():
                xp.failedJobs[self.identifier] = self

            # Handle recovery from error (e.g., resubmit)
            if old_state.is_error() and not new_state.is_error():
                xp.failedJobs.pop(self.identifier, None)

        # Clear process and progress when job finishes
        if new_state.finished():
            # Always clear process reference when finished
            self._process = None
            # Clear progress only when done (keep for failed jobs for debugging)
            if new_state == JobState.DONE:
                self._progress = []

        # Notify listeners via scheduler's thread-safe mechanism
        if self.scheduler:
            self.scheduler.notify_job_state(self)

    def find_done_in_folders(self) -> Optional[str]:
        """Check if job has finished in any attached workspace folder.

        Uses shared find_in_folders detection. Returns the workspace/folder
        display name if found done, else None.
        """
        if not self.workspace or not self.workspace.folders:
            return None

        from experimaestro.scheduler.folders import (
            find_in_folders,
            get_folder_workspace_name,
        )
        from experimaestro.scheduler.interfaces import JobState

        match = find_in_folders(Path(self.relpath), self.workspace.folders)
        if match is not None:
            folder, src = match
            marker_state = JobState.from_path(src, self.name)
            if marker_state == JobState.DONE:
                return get_folder_workspace_name(folder)

        return None

    @cached_property
    def python_path(self) -> Iterator[str]:
        """Returns an iterator over python path"""
        return itertools.chain(self.workspace.python_path)

    @cached_property
    def environ(self):
        """Returns the job environment

        It is made of (by order of priority):

        1. The job environment
        1. The launcher environment
        1. The workspace environment

        """
        return ChainMap(
            {},
            self.launcher.environ if self.launcher else {},
            self.workspace.env if self.workspace else {},
        )

    def set_progress(self, level: int, value: float, desc: Optional[str]):
        if value < 0:
            logger.warning(f"Progress value out of bounds ({value})")
            value = 0
        elif value > 1:
            logger.warning(f"Progress value out of bounds ({value})")
            value = 1

        # Adjust the length of the array
        self._progress = self._progress[: (level + 1)]
        while len(self._progress) <= level:
            self._progress.append(LevelInformation(len(self._progress), None, 0.0))

        if desc:
            self._progress[-1].desc = desc
        self._progress[-1].progress = value

    @property
    def ready(self):
        return self.scheduler_state == JobState.READY

    @property
    def jobpath(self) -> Path:
        """Deprecated, use `path`"""
        return self.workspace.jobspath / self.relpath

    @property
    def path(self) -> Path:
        return self.workspace.jobspath / self.relpath

    @property
    def experimaestro_path(self) -> Path:
        return (self.path / ".experimaestro").resolve()

    @cached_property
    def task_outputs_path(self) -> Path:
        return self.experimaestro_path / "task-outputs.jsonl"

    @property
    def relpath(self):
        identifier = self.config.__xpm__.identifier
        base = Path(str(self.type.identifier))
        return base / identifier.all.hex()

    @property
    def relmainpath(self):
        identifier = self.config.__xpm__.identifier
        base = Path(str(self.type.identifier))
        return base / identifier.main.hex()

    @property
    def hashidentifier(self):
        return self.config.__xpm__.identifier

    @property
    def identifier(self):
        return self.config.__xpm__.identifier.all.hex()

    @property
    def task_id(self) -> str:
        """Task class identifier (for BaseJob interface)"""
        return str(self.type.identifier)

    @property
    def locator(self) -> str:
        """Full task locator (for BaseJob interface)"""
        return self.identifier

    def prepare(self, overwrite=False):
        """Prepare all files before starting a task

        :param overwrite: if True, overwrite files even if the task has been run
        """
        pass

    async def aio_run(self) -> "Process":
        """Actually run the code

        Returns:
            A Process instance representing the running job
        """
        raise NotImplementedError(f"Method aio_run not implemented in {self.__class__}")

    @property
    def pidpath(self):
        """This file contains the file PID"""
        return self.jobpath / ("%s.pid" % self.name)

    @property
    def lockpath(self):
        """This file is used as a lock for running the job"""
        return self.workspace.jobspath / self.relmainpath / ("%s.lock" % self.name)

    @property
    def donepath(self) -> Path:
        """When a job has been successful, this file is written"""
        return self.jobpath / ("%s.done" % self.name)

    @property
    def failedpath(self):
        """When a job has been unsuccessful, this file is written with an error
        code inside"""
        return self.jobpath / ("%s.failed" % self.name)

    def is_output_valid(self) -> Optional[bool]:
        """Whether the job output is present, independently of marker files.

        See :meth:`experimaestro.Config.__xpm_output_valid__`. None — the
        default — means the marker files are the source of truth.
        """
        return self.config.__xpm_output_valid__()

    def clear_done_marker(self) -> None:
        """Drop a ``.done`` marker that no longer reflects reality."""
        logger.info(
            "Job %s is marked done but its output is missing - invalidating",
            self.identifier[:8],
        )
        self.donepath.unlink(missing_ok=True)
        self.set_state(JobState.UNSCHEDULED)
        self.set_scheduler_state(JobState.UNSCHEDULED)

    def mark_done(self) -> None:
        """Mark the job as done without running it, its output being present."""
        logger.info(
            "Job %s output is already present - not running it",
            self.identifier[:8],
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self.donepath.touch()
        self.set_state(JobState.DONE)
        self.set_scheduler_state(JobState.DONE)

    @property
    def stdout(self) -> Path:
        return self.jobpath / ("%s.out" % self.name)

    @property
    def stderr(self) -> Path:
        return self.jobpath / ("%s.err" % self.name)

    def rotate_logs(self) -> None:
        """Rotate log files before restarting a task.

        Renames non-empty stdout and stderr files with a timestamp suffix
        (e.g., job.20231215143022.out) to preserve logs from previous runs.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for log_path in [self.stdout, self.stderr]:
            if log_path.exists() and log_path.stat().st_size > 0:
                # Extract extension (.out or .err)
                ext = log_path.suffix
                # Create new name with timestamp before extension
                new_name = f"{log_path.stem}.{timestamp}{ext}"
                new_path = log_path.parent / new_name
                logger.info("Rotating log file %s -> %s", log_path.name, new_name)
                log_path.rename(new_path)

    def process_state_dict(self) -> dict | None:
        """Get process state as dictionary."""
        if self._process is not None:
            return self._process.tospec()
        return None

    @property
    def basepath(self) -> Path:
        return self.jobpath / self.name

    def finalState(self) -> "concurrent.futures.Future[JobState]":
        assert self._future is not None
        return self._future


class JobContext(ConfigWalkContext):
    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    @property
    def name(self):
        return self.job.name

    @property
    def path(self):
        return self.job.path

    @property
    def task(self):
        return self.job.config

    def partial_path(self, partial, config) -> Path:
        """Returns the partial directory path for a given partial instance.

        The partial path structure is:
        WORKSPACE/partials/TASK_ID/SUBPARAM_NAME/PARTIAL_ID/

        Args:
            partial: The Partial instance defining which groups to exclude
            config: The configuration to compute the partial identifier for

        Returns:
            The partial directory path.
        """
        # Compute partial identifier
        partial_id = config.__xpm__.get_partial_identifier(partial)

        # Build partial directory path
        task_id = str(config.__xpmtype__.identifier)
        return (
            self.job.workspace.partialspath
            / task_id
            / partial.name
            / partial_id.all.hex()
        )


class JobError(Exception):
    def __init__(self, code):
        super().__init__(f"Job exited with code {code}")
