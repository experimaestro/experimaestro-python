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

            # Set up dependencies
            for dependency in job.dependencies:
                dependency.target = job
                dependency.origin.dependents.add(dependency)

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

    async def aio_submit(self, job: Job) -> JobState:
        """Main scheduler function: submit a job, run it (if needed), and returns
        the status code
        """
        logger.info("Submitting job %s", job)
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

        # Check if already done
        if job.donepath.exists():
            job.state = JobState.DONE

        # Check if we have a running process
        if not job.state.finished():
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

                # Check the file if there is no return code
                if code is None:
                    # Case where we cannot retrieve the code right away
                    if job.donepath.is_file():
                        code = 0
                    else:
                        code = int(job.failedpath.read_text())

                job.state = process.get_job_state(code)

        # If not done or running, start the job
        if not job.state.finished():
            try:
                state = await self.aio_start(job)
                job.state = state
            except Exception:
                logger.exception("Got an exception while starting the job")
                raise

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

        return job.state

    async def aio_start(self, job: Job) -> Optional[JobState]:  # noqa: C901
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
        from experimaestro.scheduler.jobs import JobFailureStatus

        # Assert preconditions
        assert job.launcher is not None

        # Restart loop for resumable tasks that timeout
        while True:
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

                    # Separate static and dynamic dependencies
                    static_deps = [d for d in job.dependencies if not d.is_dynamic()]
                    dynamic_deps = [d for d in job.dependencies if d.is_dynamic()]

                    # First, wait for all static dependencies (jobs) to complete
                    # These don't need the dependency lock as they can't change state
                    # Static dependency locks don't need to be added to locks list
                    logger.debug("Waiting for %d static dependencies", len(static_deps))
                    for dependency in static_deps:
                        logger.debug("Waiting for static dependency %s", dependency)
                        await dependency.aio_lock()

                    # Now handle dynamic dependencies (tokens) with retry logic
                    # CRITICAL: Only one task at a time can acquire dynamic dependencies
                    # to prevent deadlocks (e.g., Task A holds Token1 waiting for Token2,
                    # Task B holds Token2 waiting for Token1)
                    if dynamic_deps:
                        async with self.dependencyLock:
                            logger.debug(
                                "Locking %d dynamic dependencies (tokens)",
                                len(dynamic_deps),
                            )
                            while True:
                                all_locked = True
                                for idx, dependency in enumerate(dynamic_deps):
                                    try:
                                        # Use timeout=0 for first dependency, 0.1s for subsequent
                                        timeout = 0 if idx == 0 else 0.1
                                        # Acquire the lock (this might block on IPC locks)
                                        lock = await dependency.aio_lock(
                                            timeout=timeout
                                        )
                                        locks.append(lock)
                                    except LockError:
                                        logger.debug(
                                            "Could not lock %s, retrying", dependency
                                        )
                                        # Release all locks and restart
                                        for lock in locks.locks:
                                            lock.release()
                                        locks.locks.clear()
                                        # Put failed dependency first
                                        dynamic_deps.remove(dependency)
                                        dynamic_deps.insert(0, dependency)
                                        all_locked = False
                                        break

                                if all_locked:
                                    # All locks acquired successfully
                                    break

                    # Dependencies have been locked, we can start the job
                    job.starttime = time.time()

                    # Creates the main directory
                    directory = job.path
                    logger.debug("Making directories job %s...", directory)

                    # Warn about directory cleanup for non-resumable tasks
                    if directory.is_dir() and not job.resumable:
                        logger.warning(
                            "In a future version, directory will be cleaned up for "
                            "non-resumable tasks. Use ResumableTask if you want to "
                            "preserve the directory contents."
                        )

                    if not directory.is_dir():
                        directory.mkdir(parents=True, exist_ok=True)

                    # Sets up the notification URL
                    if self.server is not None:
                        job.add_notification_server(self.server)

                except RuntimeError as e:
                    # Dependency failed - mark job as failed due to dependency
                    from experimaestro.scheduler.jobs import JobStateError

                    logger.warning("Dependency failed: %s", e)
                    return JobStateError(JobFailureStatus.DEPENDENCY)
                except Exception:
                    logger.warning("Error while locking job", exc_info=True)
                    return JobState.WAITING

                try:
                    # Runs the job
                    process = await job.aio_run()
                except Exception:
                    logger.warning("Error while starting job", exc_info=True)
                    return JobState.ERROR

                # Wait for job to complete while holding locks
                try:
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
                    # Let the process determine the job state (handles launcher-specific details)
                    state = process.get_job_state(code)

                except JobError:
                    logger.warning("Error while running job")
                    state = JobState.ERROR

                except Exception:
                    logger.warning(
                        "Error while running job (in experimaestro)", exc_info=True
                    )
                    state = JobState.ERROR

            # Locks are released here after job completes

            # Check if we should restart a resumable task that timed out
            from experimaestro.scheduler.jobs import JobStateError

            if (
                isinstance(state, JobStateError)
                and state.failure_reason == JobFailureStatus.TIMEOUT
                and job.resumable
            ):
                job.retry_count += 1
                if job.retry_count <= job.max_retries:
                    logger.info(
                        "Resumable task %s timed out - restarting (attempt %d/%d)",
                        job,
                        job.retry_count,
                        job.max_retries,
                    )
                    # Clear cached process so aio_run() will create a new one
                    job._process = None
                    # Delete PID file so the job will be resubmitted
                    if job.pidpath.exists():
                        job.pidpath.unlink()
                    # Continue the loop to restart
                    continue
                else:
                    logger.warning(
                        "Resumable task %s exceeded max retries (%d), marking as failed",
                        job,
                        job.max_retries,
                    )
                    # Fall through to return the error state

            # Job finished (success or non-recoverable error)
            # Notify scheduler listeners of job state after job completes
            self.notify_job_state(job)
            return state
