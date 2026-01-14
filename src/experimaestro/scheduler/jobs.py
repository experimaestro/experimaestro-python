import asyncio
from collections import ChainMap
from datetime import datetime
from functools import cached_property
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set

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
        if self.job.state != JobState.DONE:
            raise RuntimeError(f"Job {self.job.identifier} not done")

    async def _aio_release(self):
        pass


class JobDependency(Dependency):
    origin: "Job"

    def __init__(self, job):
        super().__init__(job)

    async def ensure_started(self):
        """Ensure the dependency job is started.

        If the dependency is a transient job that was skipped (state is UNSCHEDULED),
        this will start it so it actually runs.
        """
        origin_job = self.origin
        if (
            origin_job.transient.is_transient
            and origin_job.state == JobState.UNSCHEDULED
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
            origin_job.set_state(JobState.WAITING)
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
        if self.origin.state != JobState.DONE:
            raise RuntimeError(
                f"Dependency job {self.origin.identifier} failed with state {self.origin.state} for {self.target.identifier}"
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
    ):
        from experimaestro.scheduler.base import Scheduler

        super().__init__()

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
        self.state: JobState = JobState.UNSCHEDULED

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
        self.retry_count = 0

        # Transient mode for intermediary tasks
        self.transient = transient if transient is not None else TransientMode.NONE
        # Flag set when a transient job's mode is merged to non-transient,
        # indicating the job should run even though it was originally transient
        self._needed_transient = False

        # Watched outputs (stored for deferred registration with scheduler)
        self.watched_outputs: List["WatchedOutput"] = list(
            config.__xpm__.watched_outputs
        )

        # Process
        self._process = None

        # Meta-information
        self.starttime: Optional[datetime] = None
        self.submittime: Optional[datetime] = None
        self.endtime: Optional[datetime] = None
        self.exit_code: Optional[int] = None
        self._progress: List[LevelInformation] = []
        self.tags = config.tags()

    def watch_output(self, watched: "WatchedOutput"):
        """Add a watched output to this job.

        :param watched: A description of the watched output
        """
        self.watched_outputs.append(watched)

    def register_watched_outputs(self):
        """Register all watched outputs with the scheduler.

        This should be called after the job is submitted and has a scheduler.
        """
        from experimaestro.scheduler.experiment import experiment

        xp = experiment.current()
        for watched in self.watched_outputs:
            # Set the job reference so the watcher knows where to look
            watched.job = self
            xp.watch_output(watched)

    def done_handler(self):
        """The task has been completed.

        Ensures all remaining task output events are processed by explicitly
        reading the task outputs file. This is necessary because file system
        watchers may have latency, and we need to process all outputs before
        the experiment can exit.
        """
        if not self.watched_outputs:
            return

        for xp in self.experiments:
            xp.taskOutputsWorker.process_job_outputs(self)

    def __str__(self):
        return "Job[{}]".format(self.identifier)

    def wait(self) -> JobState:
        assert self._future, "Cannot wait a not submitted job"
        return self._future.result()

    def set_state(self, new_state: JobState):
        """Set the job state and update experiment statistics

        This method should be called instead of direct state assignment
        to ensure experiment statistics (unfinishedJobs, failedJobs) are
        properly updated.

        :param new_state: The new job state
        """
        old_state = self.state
        self.state = new_state

        # Helper to determine if a state should be "counted" in unfinishedJobs
        # A job is counted when it's been submitted and hasn't finished yet
        def is_counted(state):
            return state != JobState.UNSCHEDULED and not state.finished()

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

        # Notify listeners via scheduler's thread-safe mechanism
        if self.scheduler:
            self.scheduler.notify_job_state(self)

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

    @property
    def progress(self):
        return self._progress

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
        return self.state == JobState.READY

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

    async def aio_process(self) -> Optional["Process"]:
        """Returns the process if it exists"""
        raise NotImplementedError("Not implemented")

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
