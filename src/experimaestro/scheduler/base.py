import logging
import threading
import time
from typing import (
    Optional,
    Set,
)
import asyncio
from typing import Dict

from experimaestro.scheduler import experiment
from experimaestro.scheduler.jobs import Job, JobState
from experimaestro.scheduler.services import Service


from experimaestro.utils import logger
from experimaestro.utils.asyncio import asyncThreadcheck
import concurrent.futures


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

    def service_add(self, service: Service):
        """Notify when a new service is added"""
        pass


class Scheduler(threading.Thread):
    """A job scheduler

    The scheduler is based on asyncio for easy concurrency handling
    """

    def __init__(self, xp: "experiment", name: str):
        super().__init__(name=f"Scheduler ({name})", daemon=True)
        self._ready = threading.Event()

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

    @staticmethod
    def create(xp: "experiment", name: str):
        instance = Scheduler(xp, name)
        instance.start()
        instance._ready.wait()
        return instance

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

        self.notify_job_state(job)

        # Job is finished
        if job.state != JobState.DONE:
            self.xp.failedJobs[job.identifier] = job

        # Process all remaining tasks outputs
        await asyncThreadcheck("End of job processing", job.done_handler)

        # Decrement the number of unfinished jobs and notify
        self.xp.unfinishedJobs -= 1
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
        """Start a job (scheduler coordination layer)

        This method serves as a coordination layer that delegates the actual
        job starting logic to the job itself while handling scheduler-specific
        concerns like state notifications and providing coordination context.

        :param job: The job to start
        :return: JobState.WAITING if dependencies could not be locked, JobState.DONE
            if job completed successfully, JobState.ERROR if job failed during execution,
            or None (should not occur in normal operation)
        :raises Exception: Various exceptions during scheduler coordination
        """

        # Assert preconditions
        assert job.launcher is not None

        try:
            # Call job's start method with scheduler context
            state = await job.aio_start(
                sched_dependency_lock=self.dependencyLock,
                notification_server=self.xp.server if self.xp else None,
            )

            if state is None:
                # Dependencies couldn't be locked, return WAITING state
                return JobState.WAITING

            # Notify scheduler listeners of job state after successful start
            self.notify_job_state(job)
            return state

        except Exception:
            logger.warning("Error in scheduler job coordination", exc_info=True)
            return JobState.ERROR
