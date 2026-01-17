import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Set,
    ClassVar,
    TYPE_CHECKING,
)
import asyncio


from experimaestro.exceptions import GracefulTimeout
from experimaestro.scheduler import experiment
from experimaestro.scheduler.jobs import (
    Job,
    JobState,
    JobError,
    JobDependency,
    JobStateError,
    JobFailureStatus,
)
from experimaestro.scheduler.services import Service
from experimaestro.scheduler.interfaces import (
    BaseJob,
    BaseExperiment,
    BaseService,
    JobStateUnscheduled,
    JobStateWaiting,
)
from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.state_status import (
    CarbonMetricsEvent,
    EventReader,
    JobProgressEvent,
    JobStateChangedEvent,
    WatchedDirectory,
    job_entity_id_extractor,
)
from experimaestro.scheduler.state_provider import CarbonMetricsData


import concurrent.futures

if TYPE_CHECKING:
    from experimaestro.webui import WebUIServer
    from experimaestro.settings import ServerSettings
    from experimaestro.scheduler.workspace import Workspace
    from experimaestro.connectors import Process


logger = logging.getLogger("xpm.scheduler")


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

    def service_add(self, service: Service):
        """Notify when a new service is added"""
        pass


class Scheduler(StateProvider, threading.Thread):
    """A job scheduler (singleton) that provides live state

    The scheduler is based on asyncio for easy concurrency handling.
    This is a singleton - only one scheduler instance exists per process.

    Inherits from StateProvider to allow TUI/Web interfaces to access
    live job and experiment state during experiment execution.
    """

    _instance: ClassVar[Optional["Scheduler"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    #: Scheduler is always live
    is_live: bool = True

    def __init__(self, name: str = "Global"):
        StateProvider.__init__(self)  # Initialize state listener management
        threading.Thread.__init__(self, name=f"Scheduler ({name})", daemon=True)
        self._ready = threading.Event()

        # Name of the scheduler
        self.name = name

        # Track experiments (simple dict for now)
        self.experiments: Dict[str, "experiment"] = {}

        # Exit mode activated
        self.exitmode = False

        # List of all jobs
        self.jobs: Dict[str, "Job"] = {}

        # Services: (experiment_id, run_id) -> {service_id -> Service}
        self.services: Dict[tuple[str, str], Dict[str, Service]] = {}

        # Tags map: (experiment_id, run_id) -> {job_id -> {tag_key: tag_value}}
        self._tags_map: dict[tuple[str, str], dict[str, dict[str, str]]] = {}

        # Dependencies map: (experiment_id, run_id) -> {job_id -> [depends_on_job_ids]}
        self._dependencies_map: dict[tuple[str, str], dict[str, list[str]]] = {}

        # List of jobs
        self.waitingjobs: Set[Job] = set()

        # Legacy listeners with thread-safe access
        self._listeners: Set[Listener] = set()
        self._listeners_lock = threading.Lock()

        # Notification thread pool (single worker to serialize notifications)
        self._notification_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="NotificationWorker"
        )

        # Server (managed by scheduler)
        self.server: Optional["WebUIServer"] = None

        # Job event readers per workspace
        # Uses EventReader to watch .events/jobs/ directory
        self._job_event_readers: Dict[Path, EventReader] = {}
        self._job_event_readers_lock = threading.Lock()

    @staticmethod
    def has_instance() -> bool:
        """Check if a scheduler instance exists without creating one"""
        return Scheduler._instance is not None

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
        """Internal method to create and initialize scheduler"""
        instance = Scheduler(name)
        # Initialize the scheduler (sets up loop variables)
        # Don't call start() - scheduler uses EventLoopThread's loop
        instance.run()
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
        # Use experiment name as key (not workdir.name which is now run_id)
        key = xp.name
        self.experiments[key] = xp

        # Start watching job events for this workspace
        self._start_job_event_reader(xp.workspace.path)

        logger.debug("Registered experiment %s with scheduler", key)

    def unregister_experiment(self, xp: "experiment"):
        """Unregister an experiment from the scheduler"""
        key = xp.name
        if key in self.experiments:
            del self.experiments[key]
            logger.debug("Unregistered experiment %s from scheduler", key)

    def start_server(
        self,
        settings: "ServerSettings" = None,
        workspace: "Workspace" = None,  # noqa: ARG002 - kept for backward compat
        wait_for_quit: bool = False,
    ):
        """Start the web server (if not already running)

        Args:
            settings: Server settings
            workspace: Workspace instance (deprecated, not used)
            wait_for_quit: If True, server waits for explicit quit from web UI
        """
        if self.server is None:
            from experimaestro.webui import WebUIServer

            # Use the Scheduler itself as the StateProvider for live state access
            self.server = WebUIServer.instance(settings, self, wait_for_quit)
            self.server.start()
            logger.info("Web server started by scheduler")
        else:
            logger.debug("Web server already running")

    def stop_server(self):
        """Stop the web server"""
        if self.server is not None:
            self.server.stop()
            logger.info("Web server stopped by scheduler")

    def wait_for_server_quit(self):
        """Wait for explicit quit from web interface

        Only blocks if server was started with wait_for_quit=True.
        """
        if self.server is not None:
            self.server.wait()

    def run(self):
        """Initialize scheduler using EventLoopThread's event loop.

        The scheduler shares the event loop with lock management and file watching,
        ensuring all async operations happen in a single loop.
        """
        logger.debug("Initializing scheduler")
        from experimaestro.locking import EventLoopThread

        # Use the central event loop from EventLoopThread
        event_loop_thread = EventLoopThread.instance()
        self.loop = event_loop_thread.loop

        # Set loop-dependent variables (must be created in the event loop's context)
        # Schedule their creation on the event loop
        init_done = threading.Event()

        def init_loop_vars():
            self.exitCondition = asyncio.Condition()
            self.dependencyLock = asyncio.Lock()
            init_done.set()

        self.loop.call_soon_threadsafe(init_loop_vars)
        init_done.wait()

        self._ready.set()
        logger.debug("Scheduler initialized with EventLoopThread's loop")

    def addlistener(self, listener: Listener):
        with self._listeners_lock:
            self._listeners.add(listener)

    def removelistener(self, listener: Listener):
        with self._listeners_lock:
            self._listeners.discard(listener)

    def clear_listeners(self):
        """Clear all listeners (for testing purposes)"""
        with self._listeners_lock:
            self._listeners.clear()

    def wait_for_notifications(self, timeout: float = 5.0) -> bool:
        """Wait for all pending notifications to be processed.

        This submits a sentinel task and waits for it to complete,
        ensuring all previously submitted notifications have been processed.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all notifications were processed, False if timeout occurred
        """
        try:
            # Submit a no-op and wait for it to complete
            future = self._notification_executor.submit(lambda: None)
            future.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            logger.warning("Timeout waiting for notification queue to drain")
            return False

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
        logger.debug(
            "Job %s already submitted" if other else "First submission of job %s", job
        )

        # Only returns if job was already submitted and doesn't need reprocessing
        if other is not None:
            # If state is WAITING, it was just reset for resubmission and needs processing
            # If state is RUNNING or finished (DONE), no need to reprocess
            if other.state != JobState.WAITING:
                return other
            # Use 'other' for resubmission since it has the correct experiments list
            job = other

        async def submit():
            try:
                state = await self.aio_submit(job)
                # UNSCHEDULED is valid for transient jobs that weren't needed
                if isinstance(job.state, JobStateWaiting):
                    logger.error("Job ended with unexpected state: %s", state)
                elif isinstance(job.state, JobStateUnscheduled):
                    if job.transient.is_transient:
                        logger.debug(
                            "Transient job ended unscheduled (not needed): %s", state
                        )
                    else:
                        logger.error("Job ended with unexpected state: %s", state)
                else:
                    logger.debug("Job ended: %s", state)
                return state
            except Exception:
                logger.exception("Exception in aio_submit for job %s", job)

        job._future = asyncio.run_coroutine_threadsafe(submit(), self.loop)

        return other

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
            return

        # Job was already submitted
        if job.identifier in self.jobs:
            other = self.jobs[job.identifier]
            assert job.type == other.type

            # Add current experiment to the existing job's experiments list
            xp = experiment.current()
            xp.add_job(other)

            # Merge transient modes: more conservative mode wins
            # NONE(0) > TRANSIENT(1) > REMOVE(2) - lower value wins
            was_transient = other.transient.is_transient
            if job.transient < other.transient:
                other.transient = job.transient
                # If job was transient and is now non-transient, mark it as needed
                # This flag tells aio_submit not to skip the job
                if was_transient and not other.transient.is_transient:
                    other._needed_transient = True

            # Copy watched outputs from new job to existing job
            # This ensures new callbacks are registered even for resubmitted jobs
            other.watched_outputs.extend(job.watched_outputs)

            # Update max_retries if the new job has a higher value
            # This allows restarting failed jobs by increasing max_retries
            if job.max_retries > other.max_retries:
                other.max_retries = job.max_retries

            # Check if job needs to be re-started
            need_restart = False
            if other.state.is_error():
                need_restart = True
            elif (
                was_transient
                and not other.transient.is_transient
                and other.state == JobState.UNSCHEDULED
            ):
                # Job was transient and skipped, but now is non-transient - restart it
                logger.info("Re-submitting job (was transient, now non-transient)")
                need_restart = True

            if need_restart:
                # Clean up old process info so it will be re-started
                other._process = None
                if other.pidpath.is_file():
                    other.pidpath.unlink()
                # Use set_state to handle experiment statistics updates
                other.set_state(JobState.WAITING)
                self.notify_job_state(other)  # Notify listeners of re-submit
                # The calling aio_submit will continue with this job and start it
            else:
                logger.warning("Job %s already submitted", job.identifier)

            # Returns the previous job
            return other

        # Register this job
        xp = experiment.current()
        self.jobs[job.identifier] = job
        xp.add_job(job)

        # Update tags map for this experiment/run
        if job.tags:
            exp_run_key = (xp.name, xp.run_id)
            if exp_run_key not in self._tags_map:
                self._tags_map[exp_run_key] = {}
            self._tags_map[exp_run_key][job.identifier] = dict(job.tags)

        # Update dependencies map for this experiment/run
        exp_run_key = (xp.name, xp.run_id)
        if exp_run_key not in self._dependencies_map:
            self._dependencies_map[exp_run_key] = {}
        depends_on_ids = [
            dep.origin.identifier
            for dep in job.dependencies
            if isinstance(dep, JobDependency)
        ]
        if depends_on_ids:
            self._dependencies_map[exp_run_key][job.identifier] = depends_on_ids

        # Set up dependencies
        for dependency in job.dependencies:
            dependency.target = job
            # Some dependencies (like PartialDependency) don't have an origin resource
            if dependency.origin is not None:
                dependency.origin.dependents.add(dependency)

        return None

    def _start_job_event_reader(self, workspace_path: Path) -> None:
        """Start watching job events in a workspace

        Uses EventReader to watch .events/jobs/ for job progress events.
        Job state events are emitted by the job process itself.
        Only starts one reader per workspace.

        Args:
            workspace_path: Path to the workspace directory
        """
        with self._job_event_readers_lock:
            # Already watching this workspace
            if workspace_path in self._job_event_readers:
                return

            jobs_dir = workspace_path / ".events" / "jobs"

            # Create new reader for this workspace
            reader = EventReader(
                [
                    WatchedDirectory(
                        path=jobs_dir,
                        glob_pattern="*/event-*-*.jsonl",
                        entity_id_extractor=job_entity_id_extractor,
                    )
                ]
            )
            reader.start_watching(
                on_event=self._on_job_event,
            )
            self._job_event_readers[workspace_path] = reader
            logger.debug("Started job event reader for %s", jobs_dir)

    def _stop_job_event_reader(self, workspace_path: Optional[Path] = None) -> None:
        """Stop watching job events

        Args:
            workspace_path: If provided, stop only this workspace's reader.
                           If None, stop all readers.
        """
        with self._job_event_readers_lock:
            if workspace_path is not None:
                reader = self._job_event_readers.pop(workspace_path, None)
                if reader is not None:
                    reader.stop_watching()
                    logger.debug("Stopped job event reader for %s", workspace_path)
            else:
                # Stop all readers
                for path, reader in self._job_event_readers.items():
                    reader.stop_watching()
                    logger.debug("Stopped job event reader for %s", path)
                self._job_event_readers.clear()

    def _on_job_event(self, entity_id: str, event) -> None:
        """Handle job events from EventReader

        Updates job state from file-based events and notifies listeners.

        Args:
            entity_id: The job ID
            event: The event (JobProgressEvent, JobStateChangedEvent, CarbonMetricsEvent)
        """
        job = self.jobs.get(entity_id)
        if job is None:
            logger.debug(
                "Job event for unknown job %s",
                entity_id,
            )
            return
        logger.debug("Received event for job %s: %s", job, event)

        if isinstance(event, JobProgressEvent):
            # Update job's in-memory progress and notify legacy listeners
            job.set_progress(event.level, event.progress, event.desc)
            self.notify_job_state(job)
            # Notify StateProvider-style listeners (TUI/WebUI)
            state_event = JobStateChangedEvent(
                job_id=job.identifier,
                state=job.state.name.lower(),
            )
            self._notify_state_listeners_async(state_event)

        elif isinstance(event, CarbonMetricsEvent):
            # Update job's carbon metrics and notify listeners
            job.carbon_metrics = CarbonMetricsData(
                co2_kg=event.co2_kg,
                energy_kwh=event.energy_kwh,
                cpu_power_w=event.cpu_power_w,
                gpu_power_w=event.gpu_power_w,
                ram_power_w=event.ram_power_w,
                duration_s=event.duration_s,
                region=event.region,
                is_final=event.is_final,
            )
            logger.debug(
                "Updated carbon metrics for job %s: %.4f kg CO2",
                job.identifier,
                event.co2_kg,
            )
            # Notify StateProvider-style listeners (TUI/WebUI)
            self._notify_state_listeners_async(event)

    def _cleanup_job_marker_files(self, job: Job) -> None:
        """Clean up old marker files (.done/.failed) from previous runs

        Called when a job is about to start to ensure clean state.
        This is necessary when resubmitting a failed job.

        Args:
            job: The job being started
        """
        logger.debug("Cleaning up marker files for job %s", job.identifier[:8])
        for marker_file in [job.donepath, job.failedpath]:
            logger.debug(
                "  Checking marker file: %s (exists=%s)",
                marker_file,
                marker_file.exists(),
            )
            if marker_file.exists():
                try:
                    marker_file.unlink()
                    logger.debug("  Removed old marker file: %s", marker_file)
                except OSError as e:
                    logger.warning(
                        "Failed to remove marker file %s: %s", marker_file, e
                    )

    def _notify_listeners(self, notification_func, job: Job):
        """Execute notification in thread pool with error isolation.

        This runs notifications in a dedicated thread pool to avoid blocking
        the scheduler and to isolate errors from affecting other listeners.
        """

        def _do_notify():
            # Get a snapshot of listeners with the lock
            with self._listeners_lock:
                listeners_snapshot = list(self._listeners)

            for listener in listeners_snapshot:
                try:
                    notification_func(listener, job)
                except Exception:
                    logger.exception("Got an error with listener %s", listener)

        self._notification_executor.submit(_do_notify)

    def _notify_state_listeners_async(self, event):
        """Notify StateProvider-style listeners asynchronously with error isolation.

        This runs notifications in the same thread pool as _notify_listeners
        to avoid blocking the scheduler and isolate errors.
        """

        def _do_notify():
            # Get a snapshot of listeners with the lock
            with self._state_listener_lock:
                listeners_snapshot = list(self._state_listeners)

            for listener in listeners_snapshot:
                try:
                    listener(event)
                except Exception:
                    logger.exception("Got an error with state listener %s", listener)

        self._notification_executor.submit(_do_notify)

    def notify_job_submitted(self, job: Job):
        """Notify the listeners that a job has been submitted"""
        self._notify_listeners(lambda lst, j: lst.job_submitted(j), job)

        # Also notify StateProvider-style listeners (for TUI etc.)
        from experimaestro.scheduler.state_status import JobSubmittedEvent, JobTag

        # Get experiment info from job's experiments list
        for exp in job.experiments:
            experiment_id = exp.experiment_id
            run_id = exp.run_id

            # Get tags and dependencies for this job
            exp_run_key = (experiment_id, run_id)
            tags_dict = self._tags_map.get(exp_run_key, {}).get(job.identifier, {})
            tags = [JobTag(key=k, value=v) for k, v in tags_dict.items()]
            depends_on = self._dependencies_map.get(exp_run_key, {}).get(
                job.identifier, []
            )

            event = JobSubmittedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                job_id=job.identifier,
                tags=tags,
                depends_on=depends_on,
            )
            self._notify_state_listeners_async(event)

    def notify_job_state(self, job: Job):
        """Notify the listeners that a job has changed state

        Note: This does NOT write to job event files. Job events are written
        by the job process itself. The scheduler only forwards notifications
        to listeners.
        """
        # Legacy listener notification (per-experiment)
        self._notify_listeners(lambda lst, j: lst.job_state(j), job)

        # Notify StateProvider-style listeners with experiment-independent event
        from experimaestro.scheduler.state_status import JobStateChangedEvent

        event = JobStateChangedEvent(
            job_id=job.identifier,
            state=job.state.name.lower(),
        )
        self._notify_state_listeners_async(event)

    def notify_service_add(
        self, service: Service, experiment_id: str = "", run_id: str = ""
    ):
        """Notify the listeners that a service has been added"""
        self._notify_listeners(lambda lst, s: lst.service_add(s), service)

        # Store experiment info on the service for later retrieval
        if experiment_id:
            service._experiment_id = experiment_id
            service._run_id = run_id or ""

        # Store service in scheduler's services dict (persists after experiment ends)
        if experiment_id:
            key = (experiment_id, run_id or "")
            if key not in self.services:
                self.services[key] = {}
            self.services[key][service.id] = service

        # Also notify StateProvider-style listeners (for TUI etc.)
        from experimaestro.scheduler.state_status import ServiceAddedEvent

        if experiment_id:
            event = ServiceAddedEvent(
                experiment_id=experiment_id,
                run_id=run_id or "",
                service_id=service.id,
            )
            self._notify_state_listeners_async(event)

    async def aio_submit(self, job: Job) -> JobState:
        """Main scheduler function: submit a job, run it (if needed), and returns
        the status code
        """
        logger.info("Submitting job %s", job)
        job.scheduler = self
        self.waitingjobs.add(job)

        # Register watched outputs now that the job has a scheduler
        job.register_watched_outputs()

        # Note: Job metadata will be written after directory is created in aio_start

        # Creates a link into the experiment folder
        path = experiment.current().jobspath / job.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.symlink_to(job.path)

        # Now, notify that the job has been submitted
        self.notify_job_submitted(job)

        # Run the job and handle retries
        while True:
            try:
                await self.aio_submit_inner(job)
                break
            except GracefulTimeout:
                # If timeout we just start again the loop to start the job again
                # if needed
                logger.info(
                    "GracefulTimeout caught for job %s, restarting",
                    job.identifier[:8],
                )
                pass

        logger.debug("Current job state is %s", job.state)

        # Process job completion: task outputs, final status write, cleanup
        return await self.aio_final_state(job)

    async def aio_submit_inner(self, job: Job) -> None:
        """Inner job submission logic: load state, check for running process, start if needed.

        This method handles:
        1. Loading existing job state from disk
        2. Checking for already-done or exhausted-retries jobs
        3. Checking for running processes
        4. Starting the job with retry loop for GracefulTimeout

        Args:
            job: The job to submit
        """
        # Load existing job status if available (from previous run)
        # set_state(loading=True) properly handles experiment statistics
        job.load_from_disk()
        logger.debug("Job state after load_from_disk: %s", job.state_dict())

        # Check if job is already done
        if job.state == JobState.DONE:
            return

        # Check if job failed with exhausted retries
        if (
            job.state.is_error()
            and job.resumable
            and job.retry_count >= job.max_retries
        ):
            logger.info(
                "Job %s failed with retry_count=%d/max_retries=%d - not restarting",
                job.identifier[:8],
                job.retry_count,
                job.max_retries,
            )
            return

        # Job needs to run - clear transient fields and set to WAITING
        job._clear_transient_fields()
        job.set_state(JobState.WAITING)

        # Check if we have a running process
        if not job.state.finished():
            process = await job.aio_process()

            if process is not None:
                logger.info(
                    "Got process %s for job %s - waiting to complete", process, job
                )
                await self._wait_for_job_process(job, process)

        # If not done or running, start the job
        if not job.state.finished():
            # Check if this is a transient job that is not needed
            if job.transient.is_transient and not job._needed_transient:
                logger.debug("Job is transient and not needed, discarding for now")
                job.set_state(JobState.UNSCHEDULED)

            # Start the job if not skipped (state is still WAITING)
            # Loop to handle GracefulTimeout for resumable tasks
            logger.debug("No process for job %s (state %s), starting", job, job.state)
            while job.state == JobState.WAITING:
                try:
                    state = await self.aio_start(job)
                    logger.debug(
                        "aio_start returned %s for job %s", state, job.identifier[:8]
                    )
                    if state is not None:
                        job.set_state(state)
                        logger.debug(
                            "Job %s state after set_state: %s",
                            job.identifier[:8],
                            job.state,
                        )
                    break  # Exit loop on success
                except GracefulTimeout:
                    # Just re-raise
                    raise
                except Exception:
                    logger.exception("Got an exception while starting the job")
                    raise

    async def aio_final_state(self, job: Job) -> JobState:
        """Process job completion and return final state.

        This method handles:
        1. Skip processing for UNSCHEDULED jobs (transient jobs not needed)
        2. Write final status file (if different from disk)
        3. Process task outputs (with completion signaling)
        4. Remove job from waiting jobs
        5. Notify exit condition

        Args:
            job: The job that has completed

        Returns:
            The final job state
        """
        logger.debug("Processing final state for job %s", job.identifier[:8])

        # Skip processing for UNSCHEDULED jobs (transient jobs that weren't needed)
        if job.state == JobState.UNSCHEDULED:
            logger.debug(
                "Skipping final state for unscheduled job %s", job.identifier[:8]
            )

        else:
            # Finalize status: load from disk (carbon info), write if different
            await job.finalize_status()

            # Process task outputs (queues remaining events for processing)
            await job.aio_done_handler()

        # Remove from waiting jobs
        self.waitingjobs.discard(job)

        # Notify with final state
        self.notify_job_state(job)

        # Notify the experiments
        async with self.exitCondition:
            self.exitCondition.notify_all()

        # Wait for done handler to complete and notify exit condition
        return job.state

    async def _wait_for_job_process(self, job: Job, process: "Process") -> None:
        """Wait for a running job process to complete and update state.

        Args:
            job: The job with a running process
            process: The process to wait for
        """
        # Notify listeners that job is running
        job.set_state(JobState.RUNNING)
        self.notify_job_state(job)

        # And now, we wait...
        code = await process.aio_code()
        logger.info("Job %s completed with code %s", job, code)

        # Record exit code if available
        if code is not None:
            job.exit_code = code

        # Read state from .done/.failed files (contains detailed failure reason)
        state = JobState.from_path(job.path, job.name)

        # If state is a generic FAILED error, let the process determine
        # the state (it may detect launcher-specific failures like SLURM timeout)
        if (
            state is not None
            and isinstance(state, JobStateError)
            and state.failure_reason == JobFailureStatus.FAILED
            and code is not None
        ):
            process_state = process.get_job_state(code)
            if (
                isinstance(process_state, JobStateError)
                and process_state.failure_reason != JobFailureStatus.FAILED
            ):
                # Process detected a more specific failure reason
                state = process_state

        if state is None:
            if code is not None:
                # Fall back to process-specific state detection
                state = process.get_job_state(code)
            else:
                logger.error("No .done or .failed file found for job %s", job)
                state = JobState.ERROR

        job.set_state(state)
        self.notify_job_state(job)  # Notify listeners of final state

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
        from experimaestro.locking import DynamicDependencyLocks, LockError

        # Separate static and dynamic dependencies
        static_deps = [d for d in job.dependencies if not d.is_dynamic()]
        dynamic_deps = [d for d in job.dependencies if d.is_dynamic()]

        logger.debug(
            "Starting job %s with %d dependencies (%d static, %d dynamic)",
            job,
            len(job.dependencies),
            len(static_deps),
            len(dynamic_deps),
        )

        # First, wait for all static dependencies (jobs) to complete
        # These don't need the dependency lock as they can't change state
        # Static dependency locks don't need to be added to locks list
        logger.debug("Waiting for %d static dependencies", len(static_deps))
        for dependency in static_deps:
            logger.debug("Waiting for static dependency %s", dependency)
            try:
                await dependency.aio_lock()
            except RuntimeError as e:
                # Dependency failed - mark job as failed due to dependency
                logger.warning("Dependency failed: %s", e)
                return JobStateError(JobFailureStatus.DEPENDENCY)

        # We first lock the job before proceeding
        async with DynamicDependencyLocks() as locks:
            logger.debug("[starting] Locking job %s", job)
            async with job.launcher.connector.async_lock(job.lockpath):
                logger.debug("[starting] Locked job %s", job)

                state = None
                try:
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
                                        logger.info(
                                            "Could not lock %s, retrying",
                                            dependency,
                                        )
                                        # Release all locks and restart
                                        for lock in locks.locks:
                                            await lock.aio_release()
                                        locks.locks.clear()
                                        # Put failed dependency first
                                        dynamic_deps.remove(dependency)
                                        dynamic_deps.insert(0, dependency)
                                        all_locked = False
                                        break

                                if all_locked:
                                    # All locks acquired successfully
                                    break

                    # Creates the main directory
                    directory = job.path
                    logger.debug("Making directories job %s...", directory)

                    # Warn about directory cleanup for non-resumable tasks
                    # (only once per task type)
                    xpmtype = job.config.__xpmtype__
                    if (
                        directory.is_dir()
                        and not job.resumable
                        and not xpmtype.warned_clean_not_resumable
                    ):
                        xpmtype.warned_clean_not_resumable = True
                        logger.warning(
                            "In a future version, directory will be cleaned up for "
                            "non-resumable tasks (%s). Use ResumableTask if you want "
                            "to preserve the directory contents.",
                            xpmtype.identifier,
                        )

                    if not directory.is_dir():
                        directory.mkdir(parents=True, exist_ok=True)

                    # Clean up old job event files from previous runs
                    job._cleanup_event_files()

                    # Clean up old marker files (.done/.failed) from previous runs
                    self._cleanup_job_marker_files(job)

                    # Write metadata with submit and start time (after directory creation)
                    job.status_path.parent.mkdir(parents=True, exist_ok=True)
                    job.status_path.write_text(json.dumps(job.state_dict()))

                    # Notify locks before job starts (e.g., create symlinks)
                    await locks.aio_job_before_start(job)

                except Exception:
                    logger.warning("Error while locking job", exc_info=True)
                    return JobState.WAITING

                try:
                    # Rotate logs if this is a retry of a resumable task
                    if job.resumable and job.retry_count > 0:
                        job.rotate_logs()

                    # Runs the job
                    process = await job.aio_run()

                    # Notify locks that job has started
                    await locks.aio_job_started(job, process)

                    # Write locks.json for job process (if there are dynamic locks)
                    if locks.locks:
                        import tempfile

                        locks_path = job.path / "locks.json"
                        locks_data = {"dynamic_locks": locks.to_json()}
                        # Atomic write: write to temp file then rename
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            dir=job.path,
                            prefix=".locks.",
                            suffix=".json",
                            delete=False,
                        ) as tmp:
                            json.dump(locks_data, tmp)
                            tmp_path = tmp.name
                        # Rename is atomic on POSIX
                        import os

                        os.rename(tmp_path, locks_path)
                except Exception:
                    logger.warning("Error while starting job", exc_info=True)
                    return JobState.ERROR

            # Wait for job to complete while holding locks
            try:
                logger.debug("Waiting for job %s process to end", job)

                code = await process.aio_code()
                logger.debug("Got return code %s for %s", code, job)

                # Record exit code if available
                if code is not None:
                    logger.info("Job %s ended with code %s", job, code)
                    job.exit_code = code
                else:
                    logger.info("Job %s ended, reading state from files", job)

                # Read state from .done/.failed files (contains detailed failure reason)
                state = JobState.from_path(job.path, job.name)
                logger.debug(
                    "State from marker files for job %s: %s (code=%s, donepath=%s, failedpath=%s)",
                    job.identifier[:8],
                    state,
                    code,
                    job.donepath.exists(),
                    job.failedpath.exists(),
                )

                # If state is a generic FAILED error, let the process determine
                # the state (it may detect launcher-specific failures like SLURM timeout)
                if (
                    state is not None
                    and isinstance(state, JobStateError)
                    and state.failure_reason == JobFailureStatus.FAILED
                    and code is not None
                ):
                    process_state = process.get_job_state(code)
                    if (
                        isinstance(process_state, JobStateError)
                        and process_state.failure_reason != JobFailureStatus.FAILED
                    ):
                        # Process detected a more specific failure reason
                        state = process_state

                if state is None:
                    if code is not None:
                        # Fall back to process-specific state detection
                        state = process.get_job_state(code)
                    else:
                        logger.error("No .done or .failed file found for job %s", job)
                        state = JobState.ERROR

            except JobError:
                logger.warning("Error while running job")
                state = JobState.ERROR

            except Exception:
                logger.warning(
                    "Error while running job (in experimaestro)", exc_info=True
                )
                state = JobState.ERROR

            # Notify locks that job has finished (before releasing)
            await locks.aio_job_finished(job)

        # Locks are released here after job completes

        # Finalize status: load from disk (carbon info), set state, increment retry_count
        def finalize_callback(j: Job):
            j.set_state(state)
            # Increment retry_count for any failure of a resumable task
            if isinstance(state, JobStateError) and j.resumable:
                j.retry_count += 1

        logger.debug("[job ended] Finalizing")
        await job.finalize_status(callback=finalize_callback)

        # Check if we should restart a resumable task that timed out
        if (
            isinstance(state, JobStateError)
            and state.failure_reason == JobFailureStatus.TIMEOUT
            and job.resumable
            and job.retry_count <= job.max_retries
        ):
            logger.info(
                "Resumable task %s timed out - restarting (attempt %d/%d)",
                job,
                job.retry_count,
                job.max_retries,
            )

            # Clear cached process so aio_run() will create a new one
            job._process = None

            # Continue the loop to restart
            raise GracefulTimeout()

        # Job finished (success or non-recoverable error)
        # Notify scheduler listeners of job state after job completes
        self.notify_job_state(job)
        return state

    # =========================================================================
    # StateProvider abstract method implementations
    # =========================================================================

    def get_experiments(
        self,
        since: Optional[datetime] = None,  # noqa: ARG002
    ) -> List[BaseExperiment]:
        """Get list of all live experiments"""
        # Note: 'since' filter not applicable for live scheduler
        return list(self.experiments.values())

    def get_experiment(self, experiment_id: str) -> Optional[BaseExperiment]:
        """Get a specific experiment by ID"""
        return self.experiments.get(experiment_id)

    def get_experiment_runs(self, experiment_id: str) -> List[BaseExperiment]:
        """Get all runs for an experiment

        For a live scheduler, returns the live experiment directly.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return []

        # Return the live experiment (it already implements BaseExperiment)
        return [exp]

    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current run ID for an experiment"""
        exp = self.experiments.get(experiment_id)
        return exp.run_id if exp else None

    def get_jobs(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,  # noqa: ARG002 - not used in live scheduler
        task_id: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,  # noqa: ARG002 - not used in live scheduler
    ) -> List[BaseJob]:
        """Query jobs with optional filters"""
        jobs: List[BaseJob] = list(self.jobs.values())

        # Filter by experiment
        if experiment_id:
            exp = self.experiments.get(experiment_id)
            if exp:
                jobs = [j for j in jobs if j.experiments and exp in j.experiments]
            else:
                jobs = []

        # Filter by task_id
        if task_id:
            jobs = [j for j in jobs if j.task_id == task_id]

        # Filter by state
        if state:
            jobs = [j for j in jobs if j.state.name.lower() == state.lower()]

        # Filter by tags (all tags must match)
        if tags:
            jobs = [j for j in jobs if all(j.tags.get(k) == v for k, v in tags.items())]

        return jobs

    def get_job(
        self,
        job_id: str,
        experiment_id: str,  # noqa: ARG002 - job_id is sufficient in live scheduler
        run_id: Optional[str] = None,  # noqa: ARG002 - job_id is sufficient in live scheduler
    ) -> Optional[BaseJob]:
        """Get a specific job"""
        return self.jobs.get(job_id)

    def get_all_jobs(
        self,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,  # noqa: ARG002 - not used in live scheduler
    ) -> List[BaseJob]:
        """Get all jobs across all experiments"""
        jobs: List[BaseJob] = list(self.jobs.values())

        if state:
            jobs = [j for j in jobs if j.state.name.lower() == state.lower()]

        if tags:
            jobs = [j for j in jobs if all(j.tags.get(k) == v for k, v in tags.items())]

        return jobs

    def get_services(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[BaseService]:
        """Get services for an experiment

        Services are stored in the scheduler and persist after experiments finish.
        """
        if experiment_id is None:
            # Return all services from all experiments
            services = []
            for services_dict in self.services.values():
                services.extend(services_dict.values())
            return services

        # Get services for specific experiment
        services = []
        if run_id is not None:
            # Specific run requested
            key = (experiment_id, run_id)
            services_dict = self.services.get(key, {})
            services = list(services_dict.values())
        else:
            # No run_id specified - return services from all runs of this experiment
            for (exp_id, _run_id), services_dict in self.services.items():
                if exp_id == experiment_id:
                    services.extend(services_dict.values())

        logger.debug(
            "get_services(%s, %s): returning %d services",
            experiment_id,
            run_id,
            len(services),
        )
        return services

    def get_tags_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> dict[str, dict[str, str]]:
        """Get tags map for jobs in an experiment/run

        Returns a map from job_id to {tag_key: tag_value}.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {}

        # Use current run if not specified
        if run_id is None:
            run_id = exp.run_id

        exp_run_key = (experiment_id, run_id)
        return self._tags_map.get(exp_run_key, {})

    def get_dependencies_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """Get dependencies map for jobs in an experiment/run

        Returns a map from job_id to list of job_ids it depends on.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {}

        # Use current run if not specified
        if run_id is None:
            run_id = exp.run_id

        exp_run_key = (experiment_id, run_id)
        return self._dependencies_map.get(exp_run_key, {})

    def kill_job(self, job: BaseJob, perform: bool = False) -> bool:
        """Kill a running job

        For the scheduler, this is a live operation.
        """
        if not perform:
            # Just check if the job can be killed
            return job.state == JobState.RUNNING

        if job.state != JobState.RUNNING:
            return False

        # Get the actual Job from our jobs dict
        actual_job = self.jobs.get(job.identifier)
        if actual_job is None:
            return False

        # Try to kill the process via the process attribute
        process = getattr(actual_job, "process", None)
        if process is not None:
            try:
                process.kill()
                return True
            except Exception:
                logger.exception("Failed to kill job %s", job.identifier)
        return False

    def clean_job(
        self,
        job: BaseJob,  # noqa: ARG002
        perform: bool = False,  # noqa: ARG002
    ) -> bool:
        """Clean a finished job

        For the scheduler, jobs are automatically cleaned when they finish.
        """
        # Live scheduler doesn't support cleaning jobs
        return False

    def get_process_info(self, job: BaseJob):
        """Get process information for a job

        For the scheduler, we can access the actual Job and read its PID file.
        """
        from experimaestro.scheduler.state_provider import ProcessInfo

        # Get the actual Job from our jobs dict
        actual_job = self.jobs.get(job.identifier)
        if actual_job is None:
            return None

        # Try to read the PID file
        try:
            pidpath = getattr(actual_job, "pidpath", None)
            if pidpath is None or not pidpath.exists():
                return None

            pinfo = json.loads(pidpath.read_text())
            pid = pinfo.get("pid")
            proc_type = pinfo.get("type", "unknown")

            if pid is None:
                return None

            # Check if running based on job state
            running = actual_job.state == JobState.RUNNING

            return ProcessInfo(pid=pid, type=proc_type, running=running)
        except Exception:
            return None

    def close(self) -> None:
        """Close the state provider and clean up resources"""
        # Stop all job event readers
        self._stop_job_event_reader()

    @property
    def read_only(self) -> bool:
        """Live scheduler is read-write"""
        return False

    @property
    def is_remote(self) -> bool:
        """Live scheduler is local"""
        return False
