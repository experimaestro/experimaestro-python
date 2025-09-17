import asyncio
import time
from collections import ChainMap
import enum
from functools import cached_property
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set

import concurrent

from experimaestro.core.objects import Config, ConfigWalkContext, WatchedOutput
from experimaestro.notifications import LevelInformation, Reporter

# from experimaestro.scheduler.base import Scheduler
from experimaestro.scheduler.dependencies import Dependency, DependencyStatus, Resource
from experimaestro.scheduler.workspace import RunMode, Workspace
from experimaestro.locking import Lock, LockError, Locks
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.launchers import Launcher


class JobState(enum.Enum):
    # Job is not yet scheduled
    UNSCHEDULED = 0

    # Job is waiting for dependencies to be done
    WAITING = 1

    # Job is ready to run
    READY = 2

    # Job is scheduled (e.g. slurm)
    SCHEDULED = 3

    # Job is running
    RUNNING = 4

    # Job is done (finished)
    DONE = 5

    # Job failed (finished)
    ERROR = 6

    def notstarted(self):
        return self.value <= JobState.READY.value

    def running(self):
        return (
            self.value == JobState.RUNNING.value
            or self.value == JobState.SCHEDULED.value
        )

    def finished(self):
        return self.value >= JobState.DONE.value


class JobFailureStatus(enum.Enum):
    #: Job failed
    DEPENDENCY = 0

    #: Job dependency failed
    FAILED = 1

    #: Memory
    MEMORY = 2


class JobLock(Lock):
    def __init__(self, job):
        super().__init__()
        self.job = job

    def _acquire(self):
        return self.job.state == JobState.DONE

    def _release(self):
        return False


class JobDependency(Dependency):
    def __init__(self, job):
        super().__init__(job)

    def status(self) -> DependencyStatus:
        if self.origin.state == JobState.DONE:
            return DependencyStatus.OK
        elif self.origin.state == JobState.ERROR:
            return DependencyStatus.FAIL
        return DependencyStatus.WAIT

    def lock(self):
        return JobLock(self.origin)


class Job(Resource):
    """A job is a resource that is produced by the execution of some code"""

    # Set by the scheduler
    _readyEvent: Optional[asyncio.Event]
    _future: Optional["concurrent.futures.Future"]

    def __init__(
        self,
        config: Config,
        *,
        workspace: Workspace = None,
        launcher: "Launcher" = None,
        run_mode: RunMode = RunMode.NORMAL,
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
        self.config = config
        self.state: JobState = JobState.UNSCHEDULED

        #: If a job has failed, indicates the failure status
        self.failure_status: JobFailureStatus = None

        # Dependencies
        self.dependencies: Set[Dependency] = set()  # as target

        # Watched outputs
        self.watched_outputs = {}
        for watched in config.__xpm__.watched_outputs:
            self.watch_output(watched)

        # Process
        self._process = None
        self.unsatisfied = 0

        # Meta-information
        self.starttime: Optional[float] = None
        self.submittime: Optional[float] = None
        self.endtime: Optional[float] = None
        self._progress: List[LevelInformation] = []
        self.tags = config.tags()

    def watch_output(self, watched: "WatchedOutput"):
        """Monitor task outputs

        :param watched: A description of the watched output
        """
        self.scheduler.xp.watch_output(watched)

    def task_output_update(self, subpath: Path):
        """Notification of an updated task output"""
        if watcher := self.watched_outputs.get(subpath, None):
            watcher.update()

    def done_handler(self):
        """The task has been completed"""
        for watcher in self.watched_outputs.values():
            watcher.update()

    def __str__(self):
        return "Job[{}]".format(self.identifier)

    def wait(self) -> JobState:
        assert self._future, "Cannot wait a not submitted job"
        return self._future.result()

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

        for listener in self.scheduler.listeners:
            listener.job_state(self)

    def add_notification_server(self, server):
        """Adds a notification server"""
        key, baseurl = server.getNotificationSpec()
        dirpath = self.path / Reporter.NOTIFICATION_FOLDER
        dirpath.mkdir(exist_ok=True)
        (dirpath / key).write_text(f"{baseurl}/{self.identifier}")

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

    def prepare(self, overwrite=False):
        """Prepare all files before starting a task

        :param overwrite: if True, overwrite files even if the task has been run
        """
        pass

    async def aio_start(self, sched_dependency_lock, notification_server=None):
        """Start the job with core job starting logic

        This method contains the core logic for starting a job that was previously
        located in Scheduler.aio_start(). It handles job locking, dependency
        acquisition, directory setup, and job execution while using the scheduler's
        coordination lock to prevent race conditions between multiple jobs.

        :param sched_dependency_lock: The scheduler's dependency lock for coordination
            between jobs to prevent race conditions during dependency acquisition
        :param notification_server: Optional notification server from the experiment
            for job progress reporting
        :return: JobState.DONE if job completed successfully, JobState.ERROR if job
            failed during execution, or None if dependencies couldn't be locked
            (signals WAITING state to scheduler)
        :raises Exception: Various exceptions during job execution, dependency locking,
            or process creation
        """
        # We first lock the job before proceeding
        assert self.launcher is not None

        with Locks() as locks:
            logger.debug("[starting] Locking job %s", self)
            async with self.launcher.connector.lock(self.lockpath):
                logger.debug("[starting] Locked job %s", self)

                state = None
                try:
                    logger.debug(
                        "Starting job %s with %d dependencies",
                        self,
                        len(self.dependencies),
                    )

                    # Individual dependency lock acquisition
                    # We use the scheduler-wide lock to avoid cross-jobs race conditions
                    async with sched_dependency_lock:
                        for dependency in self.dependencies:
                            try:
                                locks.append(dependency.lock().acquire())
                            except LockError:
                                logger.warning(
                                    "Could not lock %s, aborting start for job %s",
                                    dependency,
                                    self,
                                )
                                dependency.check()
                                return None  # Signal to scheduler that dependencies couldn't be locked

                    # Dependencies have been locked, we can start the job
                    self.starttime = time.time()

                    # Creates the main directory
                    directory = self.path
                    logger.debug("Making directories job %s...", directory)
                    if not directory.is_dir():
                        directory.mkdir(parents=True, exist_ok=True)

                    # Sets up the notification URL
                    if notification_server is not None:
                        self.add_notification_server(notification_server)

                except Exception:
                    logger.warning("Error while locking job", exc_info=True)
                    return None  # Signal waiting state to scheduler

                try:
                    # Runs the job
                    process = await self.aio_run()
                except Exception:
                    logger.warning("Error while starting job", exc_info=True)
                    return JobState.ERROR

            try:
                if isinstance(process, JobState):
                    state = process
                    logger.debug("Job %s ended (state %s)", self, state)
                else:
                    logger.debug("Waiting for job %s process to end", self)

                    code = await process.aio_code()
                    logger.debug("Got return code %s for %s", code, self)

                    # Check the file if there is no return code
                    if code is None:
                        # Case where we cannot retrieve the code right away
                        if self.donepath.is_file():
                            code = 0
                        else:
                            code = int(self.failedpath.read_text())

                    logger.debug("Job %s ended with code %s", self, code)
                    state = JobState.DONE if code == 0 else JobState.ERROR

            except JobError:
                logger.warning("Error while running job")
                state = JobState.ERROR

            except Exception:
                logger.warning(
                    "Error while running job (in experimaestro)", exc_info=True
                )
                state = JobState.ERROR
        return state

    async def aio_run(self):
        """Actually run the code"""
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

    @property
    def basepath(self) -> Path:
        return self.jobpath / self.name

    def dependencychanged(self, dependency, oldstatus, status):
        """Called when a dependency has changed"""

        def value(s):
            return 1 if s == DependencyStatus.OK else 0

        self.unsatisfied -= value(status) - value(oldstatus)

        logger.debug("Job %s: unsatisfied %d", self, self.unsatisfied)

        if status == DependencyStatus.FAIL:
            # Job completed
            if not self.state.finished():
                self.state = JobState.ERROR
                self.failure_status = JobFailureStatus.DEPENDENCY
                self._readyEvent.set()

        if self.unsatisfied == 0:
            logger.info("Job %s is ready to run", self)
            # We are ready
            self.state = JobState.READY
            self._readyEvent.set()

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


class JobError(Exception):
    def __init__(self, code):
        super().__init__(f"Job exited with code {code}")
