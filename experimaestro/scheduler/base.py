from collections import defaultdict
import os
from pathlib import Path
from shutil import rmtree
import threading
import time
from typing import List, Optional, Set, Union, TYPE_CHECKING
import enum
import signal
import asyncio
from experimaestro.notifications import LevelInformation, Reporter
from typing import Dict

from experimaestro.tokens import ProcessCounterToken

from experimaestro.core.objects import Config, GenerationContext
from experimaestro.utils import logger
from experimaestro.locking import Locks, LockError, Lock
from .environment import Environment
from .workspace import Workspace
from .dependencies import Dependency, DependencyStatus, Resource
import concurrent.futures


if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.launchers import Launcher


class FailedExperiment(RuntimeError):
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
    """A job is a resouce that is produced by the execution of some code"""

    # Set by the scheduler
    _readyEvent: Optional[asyncio.Event]
    _future: Optional["concurrent.futures.Future"]

    def __init__(
        self,
        config: Config,
        *,
        workspace: Workspace = None,
        launcher: "Launcher" = None,
        dryrun: bool = False,
    ):
        super().__init__()

        self.workspace = workspace or Workspace.CURRENT
        self.launcher = launcher or self.workspace.launcher if self.workspace else None

        if not dryrun:
            assert self.workspace is not None, "No experiment has been defined"
            assert self.launcher is not None, (
                "No launcher, and no default defined for the workspace %s" % workspace
            )

        self.type = config.__xpmtype__
        self.name = str(self.type.identifier).rsplit(".", 1)[-1]

        self.scheduler: Optional["Scheduler"] = None
        self.config = config
        self.state: JobState = JobState.UNSCHEDULED

        # Dependencies
        self.dependencies: Set[Dependency] = set()  # as target

        # Process
        self._process = None
        self.unsatisfied = 0

        # Meta-information
        self.starttime: Optional[float] = None
        self.submittime: Optional[float] = None
        self.endtime: Optional[float] = None
        self._progress: List[LevelInformation] = []
        self.tags = config.tags()

    def __str__(self):
        return "Job[{}]".format(self.identifier)

    def wait(self) -> JobState:
        assert self._future, "Cannot wait a not submitted job"
        return self._future.result()

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
    def jobpath(self):
        """Deprecated, use `path`"""
        return self.workspace.jobspath / self.relpath

    @property
    def path(self) -> Path:
        return self.workspace.jobspath / self.relpath

    @property
    def relpath(self):
        identifier = self.config.__xpm__.identifier
        base = Path(str(self.type.identifier))
        if identifier.sub:
            return base / identifier.main.hex() / "xpms" / identifier.sub.hex()
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
        """When a job has been unsuccessful, this file is written with an error code inside"""
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
        value = lambda s: (1 if s == DependencyStatus.OK else 0)
        self.unsatisfied -= value(status) - value(oldstatus)

        logger.debug("Job %s: unsatisfied %d", self, self.unsatisfied)

        if status == DependencyStatus.FAIL:
            # Job completed
            if not self.state.finished():
                self.state = JobState.ERROR
                self._readyEvent.set()

        if self.unsatisfied == 0:
            logger.info("Job %s is ready to run", self)
            # We are ready
            self.state = JobState.READY
            self._readyEvent.set()

    def finalState(self) -> "concurrent.futures.Future[JobState]":
        assert self._future is not None
        return self._future


class JobContext(GenerationContext):
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


class JobError(Exception):
    def __init__(self, code):
        super().__init__(f"Job exited with code {code}")


class SignalHandler:
    def __init__(self):
        self.experiments: Set["experiment"] = set()
        signal.signal(signal.SIGINT, self)

    def add(self, xp: "experiment"):
        self.experiments.add(xp)

    def remove(self, xp):
        self.experiments.remove(xp)

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
        self.jobs: Dict[str, Job] = {}

        # List of jobs
        self.waitingjobs = set()

        # Sub-param jobs tokens
        self.subjobsTokens: Dict[str, ProcessCounterToken] = defaultdict(
            lambda: ProcessCounterToken(1)
        )

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
        otherFuture = asyncio.run_coroutine_threadsafe(
            self.aio_registerJob(job), self.loop
        )
        other = otherFuture.result()
        if other:
            return other

        job._future = asyncio.run_coroutine_threadsafe(self.aio_submit(job), self.loop)

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

    async def aio_submit(self, job: Job) -> JobState:
        """Main scheduler function: submit a job, run it (if needed), and returns
        the status code
        """
        logger.info("Submitting job %s", job)
        job._readyEvent = asyncio.Event()
        job.submittime = time.time()
        job.scheduler = self
        self.waitingjobs.add(job)

        # Creates a link into the experiment folder
        path = experiment.current().jobspath / job.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.symlink_to(job.jobpath)

        # Add process dependency if job has subparameters
        hashidentifier = job.hashidentifier
        if hashidentifier.sub is not None:
            token = self.subjobsTokens[hashidentifier.main.hex()]
            dependency = token.dependency(1)
            job.dependencies.add(dependency)

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
                except Exception as e:
                    logger.exception("Got an error with listener %s", listener)

            # Adds to the listeners
            if self.xp.server is not None:
                job.add_notification_server(self.xp.server)

            # And now, we wait...
            logger.info("Got a process for job %s - waiting to complete", job)
            code = await process.aio_code()
            logger.info("Job %s completed with code %d", job, code)
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
            self.xp.failedJobs += 1

        # Decrement the number of unfinished jobs and notify
        self.xp.unfinishedJobs -= 1
        async with self.xp.central.exitCondition:
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


class experiment:
    """Experiment context"""

    # Current experiment
    CURRENT: Optional["experiment"] = None

    @staticmethod
    def current() -> "experiment":
        """Returns the current experiment, but checking first if set"""
        assert experiment.CURRENT is not None, "No current experiment defined"
        return experiment.CURRENT

    def __init__(
        self,
        env: Union[Path, str, Environment],
        name: str,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        launcher=None,
    ):
        """
        :param env: an environment -- or a working directory for a local environment
        :param port: the port for the web server (overrides environment port if any)
        :param launcher: The launcher (if not provided, inferred from path)
        """

        from experimaestro.server import Server

        if isinstance(env, Environment):
            self.environment = env
        else:
            self.environment = Environment(workdir=env)

        # Creates the workspace
        self.workspace = Workspace(self.environment, launcher=launcher)
        self.workdir = self.workspace.experimentspath / name
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.xplockpath = self.workdir / "lock"
        self.xplock = None
        self.old_experiment = None

        # Create the scheduler
        self.scheduler = Scheduler(self, name)
        self.server = (
            Server(self.scheduler, host=host, port=port) if port is not None else None
        )

        if os.environ.get("XPM_ENABLEFAULTHANDLER", "0") == "1":
            import faulthandler

            logger.info("Enabling fault handler")
            faulthandler.enable(all_threads=True)

    def submit(self, job: Job):
        return self.scheduler.submit(job)

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
    def jobsbakpath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "jobs.bak"

    def stop(self):
        """Stop the experiment as soon as possible"""

        async def doStop():
            assert self.central is not None
            async with self.central.exitCondition:
                self.exitMode = True
                self.central.exitCondition.notify_all()

        assert self.central is not None and self.central.loop is not None
        asyncio.run_coroutine_threadsafe(doStop(), self.central.loop)

    def wait(self):
        """Wait until the running processes have finished"""

        async def awaitcompletion():
            assert self.central is not None
            async with self.central.exitCondition:
                while True:
                    if self.unfinishedJobs == 0 or self.exitMode:
                        break
                    await self.central.exitCondition.wait()

                if self.failedJobs:
                    raise FailedExperiment(f"{self.failedJobs} failed jobs")

        future = asyncio.run_coroutine_threadsafe(awaitcompletion(), self.loop)
        return future.result()

    def setenv(self, name, value):
        """Shortcut to set the environment value"""
        self.environment.setenv(name, value)

    def token(self, name: str, count: int):
        """Returns a token for this experiment depending on the host"""
        return self.workspace.connector.createtoken(name, count)

    def __enter__(self):
        logger.debug("Locking experiment %s", self.xplockpath)
        self.xplock = self.workspace.connector.lock(self.xplockpath, 0).__enter__()

        # Move old jobs into "jobs.bak"
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

        global SIGNAL_HANDLER
        # Number of unfinished jobs
        self.unfinishedJobs = 0

        # Number of failed jobs
        self.failedJobs = 0

        # Exit mode when catching signals
        self.exitMode = False

        self.central = SchedulerCentral.create(self.scheduler.name)

        if not SIGNAL_HANDLER:
            SIGNAL_HANDLER = SignalHandler()

        SIGNAL_HANDLER.add(self)

        self.old_experiment = experiment.CURRENT
        experiment.CURRENT = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If no exception, remove old "jobs"
        if exc_type is None and self.jobsbakpath.is_dir():
            rmtree(self.jobsbakpath)

        # Close the different locks
        try:
            if exc_type:
                # import faulthandler
                # faulthandler.dump_traceback()
                logger.exception(
                    "Not waiting since an exception was thrown (some jobs may be running)"
                )
            else:
                self.wait()
        finally:
            SIGNAL_HANDLER.remove(self)

            if self.central is not None:
                self.central.loop.stop()

            self.central = None
            self.workspace.__exit__(exc_type, exc_value, traceback)
            if self.xplock:
                self.xplock.__exit__(exc_type, exc_value, traceback)

            # Put back old experiment as current one
            experiment.CURRENT = self.old_experiment
            if self.server:
                self.server.stop()
