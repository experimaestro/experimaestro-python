import logging
import threading
import time
from typing import (
    Optional,
    Set,
    ClassVar,
    TYPE_CHECKING,
)
import asyncio
from typing import Dict

from experimaestro.scheduler import experiment
from experimaestro.scheduler.jobs import Job, JobState, JobError
from experimaestro.scheduler.services import Service


from experimaestro.utils import logger
from experimaestro.utils.asyncio import asyncThreadcheck
import concurrent.futures

if TYPE_CHECKING:
    from experimaestro.server import Server
    from experimaestro.settings import ServerSettings


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

    def service_add(self, service: Service):
        """Notify when a new service is added"""
        pass


class Scheduler(threading.Thread):
    """A job scheduler (singleton)

    The scheduler is based on asyncio for easy concurrency handling.
    This is a singleton - only one scheduler instance exists per process.
    """

    _instance: ClassVar[Optional["Scheduler"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, name: str = "Global"):
        super().__init__(name=f"Scheduler ({name})", daemon=True)
        self._ready = threading.Event()

        # Name of the scheduler
        self.name = name

        # Track experiments (simple dict for now)
        self.experiments: Dict[str, "experiment"] = {}

        # Exit mode activated
        self.exitmode = False

        # List of all jobs
        self.jobs: Dict[str, "Job"] = {}

        # List of jobs
        self.waitingjobs: Set[Job] = set()

        # Listeners
        self.listeners: Set[Listener] = set()

        # Server (managed by scheduler)
        self.server: Optional["Server"] = None

    @staticmethod
    def instance() -> "Scheduler":
        """Get or create the global scheduler instance"""
        if Scheduler._instance is None:
            with Scheduler._lock:
                if Scheduler._instance is None:
                    Scheduler._instance = Scheduler._create()
        return Scheduler._instance

    @staticmethod
    def _create(name: str = "Global"):
        """Internal method to create and start scheduler"""
        instance = Scheduler(name)
        instance.start()
        instance._ready.wait()
        return instance

    @staticmethod
    def create(xp: "experiment" = None, name: str = "Global"):
        """Create or get the scheduler instance

        Args:
            xp: (Deprecated) Experiment reference, ignored
            name: Name for the scheduler (only used on first creation)

        Returns:
            The global scheduler instance
        """
        return Scheduler.instance()

    def register_experiment(self, xp: "experiment"):
        """Register an experiment with the scheduler"""
        # Use experiment name as key for now
        key = xp.workdir.name
        self.experiments[key] = xp

        logger.debug("Registered experiment %s with scheduler", key)

    def unregister_experiment(self, xp: "experiment"):
        """Unregister an experiment from the scheduler"""
        key = xp.workdir.name
        if key in self.experiments:
            del self.experiments[key]
            logger.debug("Unregistered experiment %s from scheduler", key)

    def start_server(self, settings: "ServerSettings" = None):
        """Start the notification server (if not already running)"""
        if self.server is None:
            from experimaestro.server import Server

            self.server = Server.instance(settings)
            self.server.start()
            logger.info("Server started by scheduler")
        else:
            logger.debug("Server already running")

    def stop_server(self):
        """Stop the notification server"""
        if self.server is not None:
            self.server.stop()
            logger.info("Server stopped by scheduler")

    def run(self):
        """Run the event loop forever"""
        logger.debug("Starting event loop thread")
        # Ported from SchedulerCentral
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Set loop-dependent variables
        self.exitCondition = asyncio.Condition()
        self.dependencyLock = asyncio.Lock()
        self._ready.set()
        self.loop.run_forever()

    def start_scheduler(self):
        """Start the scheduler event loop in a thread"""
        if not self.is_alive():
            self.start()
            self._ready.wait()
        else:
            logger.warning("Scheduler already started")

    def addlistener(self, listener: Listener):
        self.listeners.add(listener)

    def removelistener(self, listener: Listener):
        self.listeners.remove(listener)

    def getJobState(self, job: Job) -> "concurrent.futures.Future[JobState]":
        # Check if the job belongs to this scheduler
        if job.identifier not in self.jobs:
            # If job is not in this scheduler, return its current state directly
            future = concurrent.futures.Future()
            future.set_result(job.state)
            return future

        return asyncio.run_coroutine_threadsafe(self.aio_getjobstate(job), self.loop)

    async def aio_getjobstate(self, job: Job):
        return job.state

    def submit(self, job: Job) -> Optional[Job]:
        # Wait for the future containing the submitted job
        logger.debug("Submit job %s to the scheduler", job)
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

            # Add current experiment to the existing job's experiments list
            xp = experiment.current()
            if xp not in other.experiments:
                other.experiments.append(xp)
                xp.unfinishedJobs += 1

            if other.state == JobState.ERROR:
                logger.info("Re-submitting job")
            else:
                logger.warning("Job %s already submitted", job.identifier)
                return other

        else:
            # Register this job
            xp = experiment.current()
            xp.unfinishedJobs += 1
            self.jobs[job.identifier] = job
            # Track which experiments this job belongs to
            if xp not in job.experiments:
                job.experiments.append(xp)

        return None

    def notify_job_submitted(self, job: Job):
        """Notify the listeners that a job has been submitted"""
        for listener in self.listeners:
            try:
                listener.job_submitted(job)
            except Exception:
                logger.exception("Got an error with listener %s", listener)

    def notify_job_state(self, job: Job):
        """Notify the listeners that a job has changed state"""
        for listener in self.listeners:
            try:
                listener.job_state(job)
            except Exception:
                logger.exception("Got an error with listener %s", listener)

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

        self.notify_job_submitted(job)

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
            # Notify the listeners
            self.notify_job_state(job)

            # Adds to the listeners
            if self.server is not None:
                job.add_notification_server(self.server)

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

        self.notify_job_state(job)

        # Job is finished - update all experiments tracking this job
        for xp in job.experiments:
            if job.state != JobState.DONE:
                xp.failedJobs[job.identifier] = job

        # Process all remaining tasks outputs
        await asyncThreadcheck("End of job processing", job.done_handler)

        # Decrement the number of unfinished jobs for all experiments and notify
        for xp in job.experiments:
            xp.unfinishedJobs -= 1

        async with self.exitCondition:
            logging.debug("Updated number of unfinished jobs")
            self.exitCondition.notify_all()

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
        """Start a job with full job starting logic

        This method handles job locking, dependency acquisition, directory setup,
        and job execution while using the scheduler's coordination lock to prevent
        race conditions between multiple jobs.

        :param job: The job to start
        :return: JobState.WAITING if dependencies could not be locked, JobState.DONE
            if job completed successfully, JobState.ERROR if job failed during execution,
            or None (should not occur in normal operation)
        :raises Exception: Various exceptions during job execution, dependency locking,
            or process creation
        """
        from experimaestro.locking import Locks, LockError

        # Assert preconditions
        assert job.launcher is not None

        # We first lock the job before proceeding
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

                    # Individual dependency lock acquisition
                    # We use the scheduler-wide lock to avoid cross-jobs race conditions
                    async with self.dependencyLock:
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

                    # Dependencies have been locked, we can start the job
                    job.starttime = time.time()

                    # Creates the main directory
                    directory = job.path
                    logger.debug("Making directories job %s...", directory)
                    if not directory.is_dir():
                        directory.mkdir(parents=True, exist_ok=True)

                    # Sets up the notification URL
                    if self.server is not None:
                        job.add_notification_server(self.server)

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

        # Notify scheduler listeners of job state after job completes
        self.notify_job_state(job)
        return state
