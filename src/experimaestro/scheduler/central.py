import logging
import threading
import time
from typing import (
    ClassVar,
    Optional,
    Set,
    TypeVar,
    TYPE_CHECKING,
)
import signal
import asyncio
from typing import Dict
from experimaestro.scheduler.services import Service


from experimaestro.utils import logger
from experimaestro.locking import Locks, LockError
from experimaestro.utils.asyncio import asyncThreadcheck
import concurrent.futures


if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.launchers import Launcher


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
    """This object has a unique instance and manages all the jobs
    from all the schedulers"""

    #: The event loop used to run and monitor jobs
    loop: asyncio.AbstractEventLoop

    #: The unique SchedulerCentral instance
    INSTANCE: ClassVar["SchedulerCentral"] | None = None

    def __init__(self):
        # Daemon thread so it is non blocking
        super().__init__(name="Scheduler Central", daemon=True)

        #: Signals that we are ready
        self._ready = threading.Event()

    def run(self):
        logger.debug("Starting event loop thread")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Set loop-dependent variables
        self.dependencyLock = asyncio.Lock()

        # Start the event loop
        self._ready.set()
        self.loop.run_forever()

    @staticmethod
    def create():
        if SchedulerCentral is None:
            # Create a new instance and create it
            SchedulerCentral.INSTANCE = SchedulerCentral()
            SchedulerCentral.INSTANCE.start()

            # Wait until the loop is ready
            SchedulerCentral.INSTANCE._ready.wait()
        return SchedulerCentral.INSTANCE


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
        return None

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
        job.submittime = time.time()
        job.scheduler = self
        self.waitingjobs.add(job)

        path = experiment.current().jobspath / job.relpath
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check that we don't have a completed job in
        # alternate directories
        for jobspath in experiment.current().alt_jobspaths:
            # FIXME: check if done
            pass

        # Creates a link into the experiment folder
        if path.is_symlink():
            path.unlink()
        path.symlink_to(job.path)

        job._readyEvent = asyncio.Event()
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
