from collections import ChainMap
from functools import cached_property
import itertools
import logging
import os
from pathlib import Path
from shutil import rmtree
import threading
import time
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    TYPE_CHECKING,
)
import enum
import signal
import asyncio
from experimaestro.exceptions import HandledException
from experimaestro.notifications import LevelInformation, Reporter
from typing import Dict
from experimaestro.scheduler.services import Service
from experimaestro.settings import WorkspaceSettings, get_settings


from experimaestro.core.objects import Config, ConfigWalkContext, WatchedOutput
from experimaestro.utils import logger
from experimaestro.locking import Locks, LockError, Lock
from experimaestro.utils.asyncio import asyncThreadcheck
from .workspace import RunMode, Workspace
from .dependencies import Dependency, DependencyStatus, Resource
import concurrent.futures


if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.launchers import Launcher


class FailedExperiment(HandledException):
    """Raised when an experiment failed"""

    pass


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


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

    def service_add(self, service: Service):
        """Notify when a new service is added"""
        pass


class JobError(Exception):
    def __init__(self, code):
        super().__init__(f"Job exited with code {code}")


class SignalHandler:
    def __init__(self):
        self.experiments: Set["experiment"] = set()
        self.original_sigint_handler = None

    def add(self, xp: "experiment"):
        if not self.experiments:
            self.original_sigint_handler = signal.getsignal(signal.SIGINT)

            signal.signal(signal.SIGINT, self)

        self.experiments.add(xp)

    def remove(self, xp):
        self.experiments.remove(xp)
        if not self.experiments:
            signal.signal(signal.SIGINT, self.original_sigint_handler)

    def __call__(self, signum, frame):
        """SIGINT signal handler"""
        logger.warning("Signal received")
        for xp in self.experiments:
            xp.stop()


SIGNAL_HANDLER = SignalHandler()


class SchedulerCentral(threading.Thread):
    loop: asyncio.AbstractEventLoop

    """The event loop thread used by the scheduler"""

    def __init__(self, name: str):
        # Daemon thread so it is non blocking
        super().__init__(name=f"Scheduler EL ({name})", daemon=True)

        self._ready = threading.Event()

    def run(self):
        logger.debug("Starting event loop thread")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Set loop-dependent variables
        self.exitCondition = asyncio.Condition()
        self.dependencyLock = asyncio.Lock()

        # Start the event loop
        self._ready.set()
        self.loop.run_forever()

    @staticmethod
    def create(name: str):
        instance = SchedulerCentral(name)
        instance.start()
        instance._ready.wait()
        return instance


class Scheduler:
    """A job scheduler

    The scheduler is based on asyncio for easy concurrency handling
    """

    def __init__(self, xp: "experiment", name: str):
        # Name of the experiment
        self.name = name
        self.xp = xp

        # Exit mode activated
        self.exitmode = False

        # List of all jobs
        self.jobs: Dict[str, "Job"] = {}

        # List of jobs
        self.waitingjobs: Set[Job] = set()

        # Listeners
        self.listeners: Set[Listener] = set()

    @property
    def loop(self):
        return self.xp.loop

    def addlistener(self, listener: Listener):
        self.listeners.add(listener)

    def removelistener(self, listener: Listener):
        self.listeners.remove(listener)

    def getJobState(self, job: Job) -> "concurrent.futures.Future[JobState]":
        return asyncio.run_coroutine_threadsafe(self.aio_getjobstate(job), self.loop)

    async def aio_getjobstate(self, job: Job):
        return job.state

    def submit(self, job: Job) -> Optional[Job]:
        # Wait for the future containing the submitted job
        logger.debug("Registering the job %s within the scheduler", job)
        otherFuture = asyncio.run_coroutine_threadsafe(
            self.aio_registerJob(job), self.loop
        )
        other = otherFuture.result()
        logger.debug("Job already submitted" if other else "First submission")
        if other:
            return other

        job._future = asyncio.run_coroutine_threadsafe(self.aio_submit(job), self.loop)

    def prepare(self, job: Job):
        """Prepares the job for running"""
        logger.info("Preparing job %s", job.path)
        job.prepare(overwrite=True)

    async def aio_registerJob(self, job: Job):
        """Register a job by adding it to the list, and checks
        whether the job has already been submitted
        """
        logger.debug("Registering job %s", job)

        if self.exitmode:
            logger.warning("Exit mode: not submitting")

        elif job.identifier in self.jobs:
            other = self.jobs[job.identifier]
            assert job.type == other.type
            if other.state == JobState.ERROR:
                logger.info("Re-submitting job")
            else:
                logger.warning("Job %s already submitted", job.identifier)
                return other

        else:
            # Register this job
            self.xp.unfinishedJobs += 1
            self.jobs[job.identifier] = job

        return None

    async def aio_submit(self, job: Job) -> JobState:  # noqa: C901
        """Main scheduler function: submit a job, run it (if needed), and returns
        the status code
        """
        logger.info("Submitting job %s", job)
        job._readyEvent = asyncio.Event()
        job.submittime = time.time()
        job.scheduler = self
        self.waitingjobs.add(job)

        # Check that we don't have a completed job in
        # alternate directories
        for jobspath in experiment.current().alt_jobspaths:
            # FIXME: check if done
            pass

        # Creates a link into the experiment folder
        path = experiment.current().jobspath / job.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.symlink_to(job.path)

        job.state = JobState.WAITING
        for listener in self.listeners:
            try:
                listener.job_submitted(job)
            except Exception:
                logger.exception("Got an error with listener %s", listener)

        # Add dependencies, and add to blocking resources
        if job.dependencies:
            job.unsatisfied = len(job.dependencies)

            for dependency in job.dependencies:
                dependency.target = job
                dependency.loop = self.loop
                dependency.origin.dependents.add(dependency)
                dependency.check()
        else:
            job._readyEvent.set()
            job.state = JobState.READY

        if job.donepath.exists():
            job.state = JobState.DONE

        # Check if we have a running process
        process = await job.aio_process()
        if process is not None:
            # Yep! First we notify the listeners
            job.state = JobState.RUNNING
            for listener in self.listeners:
                try:
                    listener.job_state(job)
                except Exception:
                    logger.exception("Got an error with listener %s", listener)

            # Adds to the listeners
            if self.xp.server is not None:
                job.add_notification_server(self.xp.server)

            # And now, we wait...
            logger.info("Got a process for job %s - waiting to complete", job)
            code = await process.aio_code()
            logger.info("Job %s completed with code %s", job, code)
            job.state = JobState.DONE if code == 0 else JobState.ERROR

        # Check if done
        if job.donepath.exists():
            job.state = JobState.DONE

        # OK, not done; let's start the job for real
        while not job.state.finished():
            # Wait that the job is ready
            await job._readyEvent.wait()
            job._readyEvent.clear()

            if job.state == JobState.READY:
                try:
                    state = await self.aio_start(job)
                except Exception:
                    logger.exception("Got an exception while starting the job")
                    raise

                if state is None:
                    # State is None if this is not the main thread
                    return JobState.ERROR

                job.state = state

        for listener in self.listeners:
            try:
                listener.job_state(job)
            except Exception as e:
                logger.exception("Listener %s did raise an exception", e)

        # Job is finished
        if job.state != JobState.DONE:
            self.xp.failedJobs[job.identifier] = job

        # Process all remaining tasks outputs
        await asyncThreadcheck("End of job processing", job.done_handler)

        # Decrement the number of unfinished jobs and notify
        self.xp.unfinishedJobs -= 1
        async with self.xp.central.exitCondition:
            logging.debug("Updated number of unfinished jobs")
            self.xp.central.exitCondition.notify_all()

        job.endtime = time.time()
        if job in self.waitingjobs:
            self.waitingjobs.remove(job)

        with job.dependents as dependents:
            logger.info("Processing %d dependent jobs", len(dependents))
            for dependency in dependents:
                logger.debug("Checking dependency %s", dependency)
                self.loop.call_soon(dependency.check)

        return job.state

    async def aio_start(self, job: Job) -> Optional[JobState]:
        """Start a job

        Returns None if the dependencies could not be locked after all
        Returns DONE/ERROR depending on the process outcome
        """

        # We first lock the job before proceeding
        assert job.launcher is not None
        assert self.xp.central is not None

        with Locks() as locks:
            logger.debug("[starting] Locking job %s", job)
            async with job.launcher.connector.lock(job.lockpath):
                logger.debug("[starting] Locked job %s", job)

                state = None
                try:
                    logger.debug(
                        "Starting job %s with %d dependencies",
                        job,
                        len(job.dependencies),
                    )

                    async with self.xp.central.dependencyLock:
                        for dependency in job.dependencies:
                            try:
                                locks.append(dependency.lock().acquire())
                            except LockError:
                                logger.warning(
                                    "Could not lock %s, aborting start for job %s",
                                    dependency,
                                    job,
                                )
                                dependency.check()
                                return JobState.WAITING

                    for listener in self.listeners:
                        listener.job_state(job)

                    job.starttime = time.time()

                    # Creates the main directory
                    directory = job.path
                    logger.debug("Making directories job %s...", directory)
                    if not directory.is_dir():
                        directory.mkdir(parents=True, exist_ok=True)

                    # Sets up the notification URL
                    if self.xp.server is not None:
                        job.add_notification_server(self.xp.server)

                except Exception:
                    logger.warning("Error while locking job", exc_info=True)
                    return JobState.WAITING

                try:
                    # Runs the job
                    process = await job.aio_run()
                except Exception:
                    logger.warning("Error while starting job", exc_info=True)
                    return JobState.ERROR

            try:
                if isinstance(process, JobState):
                    state = process
                    logger.debug("Job %s ended (state %s)", job, state)
                else:
                    logger.debug("Waiting for job %s process to end", job)

                    code = await process.aio_code()
                    logger.debug("Got return code %s for %s", code, job)

                    # Check the file if there is no return code
                    if code is None:
                        # Case where we cannot retrieve the code right away
                        if job.donepath.is_file():
                            code = 0
                        else:
                            code = int(job.failedpath.read_text())

                    logger.debug("Job %s ended with code %s", job, code)
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


ServiceClass = TypeVar("ServiceClass", bound=Service)


class experiment:
    """Main experiment object

    It is a context object, i.e. experiments is run with

    ```py
        with experiment(...) as xp:
            ...
    ```
    """

    #: Current experiment
    CURRENT: Optional["experiment"] = None

    @staticmethod
    def current() -> "experiment":
        """Returns the current experiment, but checking first if set

        If there is no current experiment, raises an AssertError
        """
        assert experiment.CURRENT is not None, "No current experiment defined"
        return experiment.CURRENT

    def __init__(
        self,
        env: Union[Path, str, WorkspaceSettings],
        name: str,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        token: Optional[str] = None,
        run_mode: Optional[RunMode] = None,
        launcher=None,
    ):
        """
        :param env: an environment -- or a working directory for a local
            environment

        :param name: the identifier of the experiment

        :param launcher: The launcher (if not provided, inferred from path)

        :param host: The host for the web server (overrides the environment if
            set)
        :param port: the port for the web server (overrides the environment if
            set). Use negative number to avoid running a web server (default when dry run).

        :param run_mode: The run mode for the experiment (normal, generate run
            files, dry run)
        """

        from experimaestro.server import Server
        from experimaestro.scheduler import Listener

        settings = get_settings()
        if not isinstance(env, WorkspaceSettings):
            env = WorkspaceSettings(id=None, path=Path(env))

        # Creates the workspace
        run_mode = run_mode or RunMode.NORMAL
        self.workspace = Workspace(settings, env, launcher=launcher, run_mode=run_mode)

        # Mark the directory has an experimaestro folder
        self.workdir = self.workspace.experimentspath / name
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.xplockpath = self.workdir / "lock"
        self.xplock = None
        self.old_experiment = None
        self.services: Dict[str, Service] = {}
        self._job_listener: Optional[Listener] = None

        # Get configuration settings

        if host is not None:
            settings.server.host = host

        if port is not None:
            settings.server.port = port

        if token is not None:
            settings.server.token = token

        # Create the scheduler
        self.scheduler = Scheduler(self, name)
        self.server = (
            Server(self.scheduler, settings.server)
            if (settings.server.port is not None and settings.server.port >= 0)
            and self.workspace.run_mode == RunMode.NORMAL
            else None
        )

        if os.environ.get("XPM_ENABLEFAULTHANDLER", "0") == "1":
            import faulthandler

            logger.info("Enabling fault handler")
            faulthandler.enable(all_threads=True)

    def submit(self, job: Job):
        return self.scheduler.submit(job)

    def prepare(self, job: Job):
        """Generate the file"""
        return self.scheduler.prepare(job)

    @property
    def run_mode(self):
        return self.workspace.run_mode

    @property
    def loop(self):
        assert self.central is not None
        return self.central.loop

    @property
    def resultspath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "results"

    @property
    def jobspath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "jobs"

    @property
    def alt_jobspaths(self):
        """Return potential other directories"""
        for alt_workdir in self.workspace.alt_workdirs:
            yield alt_workdir / "jobs"

    @property
    def jobsbakpath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "jobs.bak"

    def stop(self):
        """Stop the experiment as soon as possible"""

        async def doStop():
            assert self.central is not None
            async with self.central.exitCondition:
                self.exitMode = True
                logging.debug("Setting exit mode to true")
                self.central.exitCondition.notify_all()

        assert self.central is not None and self.central.loop is not None
        asyncio.run_coroutine_threadsafe(doStop(), self.central.loop)

    def wait(self):
        """Wait until the running processes have finished"""

        async def awaitcompletion():
            assert self.central is not None
            logger.debug("Waiting to exit scheduler...")
            async with self.central.exitCondition:
                while True:
                    if self.exitMode:
                        break

                    # If we have still unfinished jobs or possible new tasks, wait
                    logger.debug(
                        "Checking exit condition: unfinished jobs=%d, task output queue size=%d",
                        self.unfinishedJobs,
                        self.taskOutputQueueSize,
                    )
                    if self.unfinishedJobs == 0 and self.taskOutputQueueSize == 0:
                        break

                    # Wait for more news...
                    await self.central.exitCondition.wait()

                if self.failedJobs:
                    # Show some more information
                    count = 0
                    for job in self.failedJobs.values():
                        if job.failure_status != JobFailureStatus.DEPENDENCY:
                            count += 1
                            logger.error(
                                "Job %s failed, check the log file %s",
                                job.relpath,
                                job.stderr,
                            )
                    raise FailedExperiment(f"{count} failed jobs")

        future = asyncio.run_coroutine_threadsafe(awaitcompletion(), self.loop)
        return future.result()

    def setenv(self, name, value, override=True):
        """Shortcut to set the environment value"""
        if override or name not in self.workspace.env:
            logging.info("Setting environment: %s=%s", name, value)
            self.workspace.env[name] = value

    def token(self, name: str, count: int):
        """Returns a token for this experiment

        The token is the default token of the workspace connector"""
        return self.workspace.connector.createtoken(name, count)

    def __enter__(self):
        from .dynamic_outputs import TaskOutputsWorker

        if self.workspace.run_mode != RunMode.DRY_RUN:
            logger.info("Locking experiment %s", self.xplockpath)
            self.xplock = self.workspace.connector.lock(self.xplockpath, 0).__enter__()
            logger.info("Experiment locked")

        # Move old jobs into "jobs.bak"
        if self.workspace.run_mode == RunMode.NORMAL:
            self.jobsbakpath.mkdir(exist_ok=True)
            for p in self.jobspath.glob("*/*"):
                if p.is_symlink():
                    target = self.jobsbakpath / p.relative_to(self.jobspath)
                    if target.is_symlink():
                        # Remove if duplicate
                        p.unlink()
                    else:
                        # Rename otherwise
                        target.parent.mkdir(parents=True, exist_ok=True)
                        p.rename(target)

        if self.server:
            self.server.start()

        self.workspace.__enter__()
        (self.workspace.path / ".__experimaestro__").touch()

        global SIGNAL_HANDLER
        # Number of unfinished jobs
        self.unfinishedJobs = 0
        self.taskOutputQueueSize = 0

        # List of failed jobs
        self.failedJobs: Dict[str, Job] = {}

        # Exit mode when catching signals
        self.exitMode = False

        self.central = SchedulerCentral.create(self.scheduler.name)
        self.taskOutputsWorker = TaskOutputsWorker(self)
        self.taskOutputsWorker.start()

        SIGNAL_HANDLER.add(self)

        self.old_experiment = experiment.CURRENT
        experiment.CURRENT = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug("Exiting scheduler context")
        # If no exception and normal run mode, remove old "jobs"
        if self.workspace.run_mode == RunMode.NORMAL:
            if exc_type is None and self.jobsbakpath.is_dir():
                rmtree(self.jobsbakpath)

        # Close the different locks
        try:
            if exc_type:
                # import faulthandler
                # faulthandler.dump_traceback()
                logger.error(
                    "Not waiting since an exception was thrown"
                    " (some jobs may be running)"
                )
            else:
                self.wait()
        finally:
            SIGNAL_HANDLER.remove(self)

            # Stop services
            for service in self.services.values():
                logger.info("Closing service %s", service.description())
                service.stop()

            if self.central is not None:
                logger.info("Stopping scheduler event loop")
                self.central.loop.stop()

            if self.taskOutputsWorker is not None:
                logger.info("Stopping tasks outputs worker")
                self.taskOutputsWorker.queue.put(None)

            self.central = None
            self.workspace.__exit__(exc_type, exc_value, traceback)
            if self.xplock:
                self.xplock.__exit__(exc_type, exc_value, traceback)

            # Put back old experiment as current one
            experiment.CURRENT = self.old_experiment
            if self.server:
                logger.info("Stopping web server")
                self.server.stop()

        if self.workspace.run_mode == RunMode.NORMAL:
            # Write the state
            logging.info("Saving the experiment state")
            from experimaestro.scheduler.state import ExperimentState

            ExperimentState.save(
                self.workdir / "state.json", self.scheduler.jobs.values()
            )

    async def update_task_output_count(self, delta: int):
        """Change in the number of task outputs to process"""
        async with self.central.exitCondition:
            self.taskOutputQueueSize += delta
            logging.debug(
                "Updating queue size with %d => %d", delta, self.taskOutputQueueSize
            )
            if self.taskOutputQueueSize == 0:
                self.central.exitCondition.notify_all()

    def watch_output(self, watched: "WatchedOutput"):
        """Watch an output

        :param watched: The watched output specification
        """

        self.taskOutputsWorker.watch_output(watched)

    def add_service(self, service: ServiceClass) -> ServiceClass:
        """Adds a service (e.g. tensorboard viewer) to the experiment

        :param service: A service instance
        :return: The same service instance
        """
        self.services[service.id] = service
        for listener in self.scheduler.listeners:
            listener.service_add(service)
        return service

    def save(self, obj: Any, name: str = "default"):
        """Serializes configurations.

        Saves configuration objects within the experimental directory

        :param obj: The object to save
        :param name: The name of the saving directory (default to `default`)
        """

        if self.workspace.run_mode == RunMode.NORMAL:
            from experimaestro import save

            save_dir = self.workdir / "data" / name
            save_dir.mkdir(exist_ok=True, parents=True)

            save(obj, save_dir)

    def load(self, reference: str, name: str = "default"):
        """Serializes configurations.

        Loads configuration objects from an experimental directory

        :param reference: The name of the experiment
        :param name: The name of the saving directory (default to `default`)
        """
        from experimaestro import load

        path = self.workspace.experimentspath / reference / "data" / name
        return load(path)
