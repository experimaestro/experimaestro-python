import asyncio
import inspect
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import time
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union

from experimaestro.core.objects import WatchedOutput
from experimaestro.exceptions import HandledException

from experimaestro.scheduler.signal_handler import SIGNAL_HANDLER
from experimaestro.scheduler.jobs import Job
from experimaestro.scheduler.services import Service
from experimaestro.scheduler.workspace import RunMode, Workspace
from experimaestro.scheduler.interfaces import (
    BaseExperiment,
    BaseService,
    ExperimentJobInformation,
)
from experimaestro.settings import WorkspaceSettings, get_settings, HistorySettings
from experimaestro.experiments.configuration import DirtyGitAction
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import ExperimentStatus
    from experimaestro.scheduler.state_status import ExperimentEventWriter

ServiceClass = TypeVar("ServiceClass", bound=Service)


class FailedExperiment(HandledException):
    """Raised when an experiment failed"""

    pass


class DirtyGitError(HandledException):
    """Raised when the git repository has uncommitted changes and dirty_git=error"""

    pass


class GracefulExperimentExit(Exception):
    """Raised to exit an experiment context without waiting for running jobs.

    This is useful in tests or when you want to detach from an experiment
    while keeping jobs running (e.g., to test stray job detection).

    Example::

        with experiment(workdir, "my-experiment") as xp:
            task = MyTask.C(value=1).submit()
            # Wait for task to start...
            raise GracefulExperimentExit()  # Exit without waiting for task to finish
    """

    pass


class StateListener:
    """Listener that writes events to filesystem

    Job state events are written to per-job event files by the scheduler.
    This listener writes experiment-level events (job state, services) to
    the experiment event file.
    """

    def __init__(
        self,
        event_writer: "ExperimentEventWriter",
        experiment: "experiment",
        experiment_id: str,
        run_id: str,
    ):
        self.event_writer = event_writer
        self.experiment = experiment
        self.experiment_id = experiment_id
        self.run_id = run_id

    def job_submitted(self, job):
        # Already handled in experiment.add_job()
        pass

    def job_state(self, job):
        """Write job state change event to experiment event file"""
        from .state_status import JobStateChangedEvent
        from experimaestro.scheduler.interfaces import serialize_timestamp

        # Get failure reason if error state
        failure_reason = None
        if hasattr(job.state, "failure_reason") and job.state.failure_reason:
            failure_reason = job.state.failure_reason.name

        # Get progress as list of dicts
        progress = []
        if hasattr(job, "_progress") and job._progress:
            progress = [
                {"level": p.level, "progress": p.progress, "desc": p.desc}
                for p in job._progress
            ]

        # Serialize datetime objects to ISO strings for event storage
        event = JobStateChangedEvent(
            job_id=job.identifier,
            state=job.state.name,
            failure_reason=failure_reason,
            submitted_time=serialize_timestamp(job.submittime),
            started_time=serialize_timestamp(job.starttime),
            ended_time=serialize_timestamp(job.endtime),
            exit_code=getattr(job, "exit_code", None),
            retry_count=getattr(job, "retry_count", 0),
            progress=progress,
        )
        # Write to experiment event file
        self.event_writer.write_event(event)

    def service_add(self, service):
        """Write service added event to filesystem"""
        from experimaestro.scheduler.services import Service
        from .state_status import ServiceAddedEvent

        state_dict = Service.serialize_state_dict(service.state_dict())
        service_class = f"{service.__class__.__module__}.{service.__class__.__name__}"
        event = ServiceAddedEvent(
            service_id=service.id,
            description=service.description(),
            service_class=service_class,
            state_dict=state_dict,
        )
        self.event_writer.write_event(event)

    def service_state_changed(self, service):
        """Called when service state changes (runtime only, not persisted)"""
        # Service state is managed at runtime, not persisted
        pass


class experiment(BaseExperiment):
    """Context manager for running experiments.

    Creates a workspace, manages task submission, and optionally starts
    a web server for monitoring.

    Implements BaseExperiment interface for use with StateProvider and TUI.

    Example::

        from experimaestro import experiment

        with experiment("./workdir", "my-experiment", port=12345) as xp:
            task = MyTask.C(param=42).submit()
            result = task.wait()
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
        register_signals: bool = True,
        project_paths: Optional[list[Path]] = None,
        wait_for_quit: bool = False,
        dirty_git: DirtyGitAction = DirtyGitAction.WARN,
        no_db: bool = False,
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

        :param register_signals: Whether to register signal handlers (default: True).
            Set to False when running in a background thread.

        :param project_paths: Paths to the project files (for git info). If not
            provided, will be inferred from the caller's location.

        :param wait_for_quit: Deprecated, no longer used. Web server is no longer
            started automatically.

        :param dirty_git: Action when git repository has uncommitted changes:
            DirtyGitAction.IGNORE (don't check), DirtyGitAction.WARN (log warning,
            default), or DirtyGitAction.ERROR (raise exception).

        :param no_db: Deprecated, kept for backwards compatibility. This parameter
            is now a no-op as the database has been replaced with filesystem-based
            state tracking.

        .. deprecated::
            The ``host``, ``port``, ``token``, and ``wait_for_quit`` parameters are
            deprecated. Use ``--web`` flag with ``run-experiment`` CLI or start the
            web server separately.
        """
        import warnings

        from experimaestro.scheduler import Listener, Scheduler

        # Warn about deprecated server parameters
        if host is not None:
            warnings.warn(
                "The 'host' parameter is deprecated. Use '--web' flag with "
                "'run-experiment' CLI or start the web server separately.",
                DeprecationWarning,
                stacklevel=2,
            )
        if port is not None:
            warnings.warn(
                "The 'port' parameter is deprecated. Use '--web' flag with "
                "'run-experiment' CLI or start the web server separately.",
                DeprecationWarning,
                stacklevel=2,
            )
        if token is not None:
            warnings.warn(
                "The 'token' parameter is deprecated. Use '--web' flag with "
                "'run-experiment' CLI or start the web server separately.",
                DeprecationWarning,
                stacklevel=2,
            )
        if wait_for_quit:
            warnings.warn(
                "The 'wait_for_quit' parameter is deprecated. Use '--web' flag with "
                "'run-experiment' CLI or start the web server separately.",
                DeprecationWarning,
                stacklevel=2,
            )

        settings = get_settings()
        if not isinstance(env, WorkspaceSettings):
            env = WorkspaceSettings(id=None, path=Path(env))

        # Creates the workspace
        run_mode = run_mode or RunMode.NORMAL
        self.workspace = Workspace(settings, env, launcher=launcher, run_mode=run_mode)

        # Store experiment name for ID references
        self.name = name

        # Create experiment base directory (run directories will be created inside)
        self._experiment_base = self.workspace.experimentspath / name
        self._experiment_base.mkdir(parents=True, exist_ok=True)

        # Lock is at experiment level (prevents concurrent runs of same experiment)
        self.xplockpath = self._experiment_base / "lock"

        # workdir will be set in __enter__ after run_id is generated
        self.workdir = None
        self.xplock = None
        self.old_experiment = None
        self._services: Dict[str, Service] = {}
        self._job_listener: Optional[Listener] = None
        self._register_signals = register_signals
        self._dirty_git = dirty_git
        self._no_db = no_db

        # Capture project paths for git info
        if project_paths is not None:
            self._project_paths = project_paths
        else:
            # Fall back to caller's file path
            self._project_paths = []
            try:
                # Go up the stack to find the first frame outside this module
                for frame_info in inspect.stack():
                    frame_file = frame_info.filename
                    if "experimaestro" not in frame_file:
                        self._project_paths = [Path(frame_file).resolve().parent]
                        break
            except Exception:
                pass

        # Use singleton scheduler
        self.scheduler = Scheduler.instance()

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
        assert self.scheduler is not None, "No scheduler defined"
        return self.scheduler.loop

    @property
    def server(self):
        """Access the server via the scheduler"""
        return self.scheduler.server if self.scheduler else None

    @property
    def resultspath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "results"

    @property
    def jobspath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "jobs"

    # =========================================================================
    # BaseExperiment interface properties
    # =========================================================================

    @property
    def experiment_id(self) -> str:
        """Experiment identifier (overrides BaseExperiment.experiment_id)"""
        return self.name

    @property
    def status(self) -> "ExperimentStatus":
        """Experiment status - RUNNING for live experiments, updated on finalization"""
        from experimaestro.scheduler.interfaces import ExperimentStatus

        return getattr(self, "_status", ExperimentStatus.RUNNING)

    @property
    def jobs(self) -> Dict[str, "Job"]:
        """Jobs in this experiment"""
        return {
            job.identifier: job
            for job in self.scheduler.jobs.values()
            if self in job.experiments
        }

    @property
    def tags(self) -> Dict[str, Dict[str, str]]:
        """Tags for jobs - tracked directly in experiment"""
        return self._tags

    @property
    def dependencies(self) -> Dict[str, List[str]]:
        """Job dependencies - tracked directly in experiment"""
        return self._dependencies

    @property
    def events_count(self) -> int:
        """Number of events processed - delegated to EventWriter"""
        if self._event_writer is not None:
            return self._event_writer._count
        return 0

    @property
    def started_at(self) -> Optional[datetime]:
        """Datetime when experiment started"""
        return self._started_at

    @property
    def ended_at(self) -> Optional[datetime]:
        """Datetime when experiment ended (None if still running)"""
        return self._ended_at

    @property
    def hostname(self) -> Optional[str]:
        """Hostname where experiment is running"""
        return self._hostname

    @property
    def services(self) -> Dict[str, "BaseService"]:
        """Services in this experiment"""
        return self._services

    @property
    def alt_jobspaths(self):
        """Return potential other directories"""
        for alt_workdir in self.workspace.alt_workdirs:
            yield alt_workdir / "jobs"

    @property
    def jobs_jsonl_path(self):
        """Return the path to the jobs.jsonl file for this experiment"""
        return self.workdir / "jobs.jsonl"

    def add_job(self, job: "Job"):
        """Register a job and its tags to jobs.jsonl file and database

        Note: For NEW jobs, the unfinishedJobs counter is updated by
        job.set_state() when the state transitions from UNSCHEDULED.
        For jobs already running, we increment here since no state
        transition will occur.
        """
        from experimaestro.scheduler.interfaces import JobState

        if self in job.experiments:
            # Do not double register
            return

        # Track which experiments this job belongs to
        job.experiments.append(self)

        # If job is already being tracked (not UNSCHEDULED and not finished),
        # increment unfinishedJobs since no state transition will trigger it
        if job.state != JobState.UNSCHEDULED and not job.state.finished():
            self.unfinishedJobs += 1
            logging.debug(
                "Job %s already running, unfinished jobs for %s: %d",
                job.identifier[:8],
                self.name,
                self.unfinishedJobs,
            )

        job_info = ExperimentJobInformation(
            job_id=job.identifier,
            task_id=str(job.type.identifier),
            tags=dict(job.tags.items()) if job.tags else {},
            timestamp=time.time(),
        )

        with self.jobs_jsonl_path.open("a") as f:
            f.write(json.dumps(job_info.to_dict()) + "\n")

        # Write job submitted event to filesystem (only in NORMAL mode)
        if self._event_writer is not None:
            from .state_status import JobSubmittedEvent

            # Get dependency job IDs
            depends_on = []
            if hasattr(job, "dependencies"):
                for dep in job.dependencies:
                    if hasattr(dep, "identifier"):
                        depends_on.append(dep.identifier)

            job_tags = dict(job.tags.items()) if job.tags else {}
            event = JobSubmittedEvent(
                job_id=job.identifier,
                task_id=str(job.type.identifier),
                transient=job.transient.value if hasattr(job, "transient") else 0,
                tags=job_tags,
                depends_on=depends_on,
            )
            self._event_writer.write_event(event)

            # Track tags and dependencies directly in experiment
            if job_tags:
                self._tags[job.identifier] = job_tags
            if depends_on:
                self._dependencies[job.identifier] = depends_on

    def _finalize_run(self, status: str) -> None:
        """Finalize the run: write final status.json and archive event files

        Args:
            status: Final status ("completed" or "failed")
        """
        from experimaestro.scheduler.interfaces import ExperimentStatus
        from .state_status import RunCompletedEvent

        # Update final status in the experiment
        self._ended_at = datetime.now()
        if status in ("completed", "done"):
            self._status = ExperimentStatus.DONE
        elif status == "failed":
            self._status = ExperimentStatus.FAILED

        # Write RunCompletedEvent before closing the event writer
        event = RunCompletedEvent(status=status, ended_at=datetime.now().isoformat())
        self._event_writer.write_event(event)

        # Close the event writer to flush any buffered events
        self._event_writer.close()

        # Write final status.json using write_status()
        self.write_status()

        # Archive event files to permanent storage
        self._event_writer.archive_events()

    def stop(self):
        """Stop the experiment as soon as possible"""

        async def doStop():
            assert self.scheduler is not None
            async with self.scheduler.exitCondition:
                self.exitMode = True
                logging.debug("Setting exit mode to true")
                self.scheduler.exitCondition.notify_all()

        assert self.scheduler is not None and self.scheduler.loop is not None
        asyncio.run_coroutine_threadsafe(doStop(), self.scheduler.loop)

    def wait(self):
        """Wait until the running processes have finished"""

        async def awaitcompletion():
            assert self.scheduler is not None, "No scheduler defined"
            logger.debug("Waiting to exit scheduler...")
            async with self.scheduler.exitCondition:
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
                    await self.scheduler.exitCondition.wait()

                if self.failedJobs:
                    # Show some more information
                    from experimaestro.scheduler.jobs import (
                        JobStateError,
                        JobFailureStatus,
                    )

                    count = 0
                    for job in self.failedJobs.values():
                        # Skip dependency failures - only log direct failures
                        if isinstance(job.state, JobStateError):
                            if job.state.failure_reason != JobFailureStatus.DEPENDENCY:
                                count += 1
                                logger.error(
                                    "Job %s failed, check the log file %s",
                                    job.relpath,
                                    job.stderr,
                                )
                        else:
                            # Should not happen, but count it anyway
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
        from experimaestro.utils.environment import (
            ExperimentEnvironment,
            ExperimentRunInfo,
        )

        # Check for old experiment layout and warn
        old_xp_dir = self.workspace.path / "xp"
        if old_xp_dir.exists() and old_xp_dir.is_dir():
            logger.warning(
                "Experimaestro v2 has a modified experiment file layout. "
                "DO NOT use experimaestro v1 to cleanup orphans. "
                "You can use 'experimaestro migrate v1-to-v2 %s' to migrate old experiment "
                "folders to the new structure.",
                self.workspace.path,
            )

        # Only lock and save environment in NORMAL mode
        if self.workspace.run_mode == RunMode.NORMAL:
            logger.info("Locking experiment %s", self.xplockpath)
            lock = self.workspace.connector.lock(self.xplockpath, 0)

            # Try non-blocking first to check if lock is held
            if not lock.acquire(blocking=False):
                # Lock is held - try to find hostname from latest run's environment.json
                hostname = None
                try:
                    # Find the most recent run directory
                    run_dirs = sorted(
                        [d for d in self._experiment_base.iterdir() if d.is_dir()],
                        key=lambda d: d.stat().st_mtime,
                        reverse=True,
                    )
                    if run_dirs:
                        env_path = run_dirs[0] / "environment.json"
                        if env_path.exists():
                            env = ExperimentEnvironment.load(env_path)
                            hostname = env.run.hostname if env.run else None
                except Exception:
                    pass  # Ignore errors when trying to find hostname
                holder_info = f" (held by {hostname})" if hostname else ""
                logger.warning(
                    "Experiment is locked%s, waiting for lock to be released...",
                    holder_info,
                )
                # Now wait for the lock
                lock.acquire(blocking=True)

            self.xplock = lock
            logger.info("Experiment locked")

            # Generate run_id with collision detection
            now = datetime.now()
            base_run_id = now.strftime("%Y%m%d_%H%M%S")
            run_id = base_run_id
            suffix = 1
            while (self._experiment_base / run_id).exists():
                run_id = f"{base_run_id}.{suffix}"
                suffix += 1
            self.run_id = run_id

            # Create the run-specific workdir
            self.workdir = self._experiment_base / self.run_id
            self.workdir.mkdir(parents=True, exist_ok=True)

            # Capture and save environment info
            from experimaestro.utils.git import get_git_info
            from experimaestro.utils.environment import get_current_environment

            env_info_path = self.workdir / "environment.json"
            env = get_current_environment()

            # Capture project git info from project paths
            dirty_repos = []
            for project_path in self._project_paths:
                project_git = get_git_info(project_path)
                if project_git:
                    env.projects.append(project_git)
                    # Track dirty repositories
                    if project_git.get("dirty"):
                        dirty_repos.append(project_git.get("path", str(project_path)))

            # Handle dirty git repositories based on configured action
            if dirty_repos and self._dirty_git != DirtyGitAction.IGNORE:
                for repo_path in dirty_repos:
                    if self._dirty_git == DirtyGitAction.WARN:
                        logger.warning(
                            "Project repository has uncommitted changes: %s",
                            repo_path,
                        )
                    elif self._dirty_git == DirtyGitAction.ERROR:
                        # Release the lock before raising the error
                        raise DirtyGitError(
                            f"Project repository has uncommitted changes: {repo_path}"
                        )

            env.save(env_info_path)
        else:
            # Non-NORMAL mode: use placeholder run_id and workdir
            self.run_id = "dry-run"
            self.workdir = self._experiment_base / self.run_id
            self.workdir.mkdir(parents=True, exist_ok=True)

        # Register experiment with scheduler
        self.scheduler.register_experiment(self)

        # Set experiment start time for BaseExperiment interface
        self._started_at = datetime.now()
        self._ended_at = None

        self.workspace.__enter__()
        (self.workspace.path / ".__experimaestro__").touch()

        # Initialize filesystem-based state tracking (only in NORMAL mode)
        from .state_status import ExperimentEventWriter

        is_normal_mode = self.workspace.run_mode == RunMode.NORMAL
        self._event_writer = None
        self._state_listener = None

        # Track job tags and dependencies directly (no more StatusData)
        self._tags: Dict[str, Dict[str, str]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._hostname: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None

        if is_normal_mode:
            import socket

            # Create event writer for this experiment
            # Events are written to experiments/{experiment_id}/events-{count}.jsonl
            # Permanent storage: workdir/events/
            self._event_writer = ExperimentEventWriter(self, self.workspace.path, 0)

            # Initialize status.json for this run
            self._hostname = socket.gethostname()
            self._started_at = datetime.now()
            self._event_writer.init_status()

            # Create symlink to current run
            self._event_writer.create_symlink()

            # Add run info to environment.json
            env_path = self.workdir / "environment.json"
            env = ExperimentEnvironment.load(env_path)
            env.run = ExperimentRunInfo(
                hostname=self._hostname,
                started_at=datetime.now().isoformat(),
            )
            env.save(env_path)

            # Add state listener to write events to filesystem
            self._state_listener = StateListener(
                self._event_writer, self, self.name, self.run_id
            )
            self.scheduler.addlistener(self._state_listener)

        # Number of unfinished jobs
        self.unfinishedJobs = 0
        self.taskOutputQueueSize = 0

        # List of failed jobs
        self.failedJobs: Dict[str, Job] = {}

        # Exit mode when catching signals
        self.exitMode = False

        # Note: scheduler is already running as singleton
        self.taskOutputsWorker = TaskOutputsWorker(self)
        self.taskOutputsWorker.start()

        if self._register_signals:
            SIGNAL_HANDLER.add(self)

        self.old_experiment = experiment.CURRENT
        experiment.CURRENT = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug("Exiting scheduler context")

        # Close the different locks
        try:
            if exc_type is GracefulExperimentExit:
                # Graceful exit - don't wait for jobs, don't log error
                logger.info("Graceful experiment exit - not waiting for running jobs")
            elif exc_type:
                # import faulthandler
                # faulthandler.dump_traceback()
                logger.exception(
                    "Not waiting since an exception was thrown (some jobs may be running)"
                )
            else:
                self.wait()

            # Wait for all pending notifications to be processed
            # before removing listeners
            self.scheduler.wait_for_notifications()
        finally:
            if self._register_signals:
                SIGNAL_HANDLER.remove(self)

            # Stop services
            for service in self.services.values():
                logger.info("Closing service %s", service.description())
                service.stop()

            # Set end time for BaseExperiment interface
            self._ended_at = datetime.now()

            # Unregister experiment from scheduler
            self.scheduler.unregister_experiment(self)

            # Remove state listener and finalize run (only in NORMAL mode)
            if exc_type is GracefulExperimentExit:
                status = "detached"  # Graceful exit, jobs may still be running
            elif exc_type:
                status = "failed"
            else:
                status = "completed"

            if self._state_listener is not None:
                self.scheduler.removelistener(self._state_listener)
                self._finalize_run(status)

            # Update environment.json with run status
            if self.workspace.run_mode == RunMode.NORMAL and self.workdir:
                from experimaestro.utils.environment import ExperimentEnvironment

                env_path = self.workdir / "environment.json"
                if env_path.exists():
                    try:
                        env = ExperimentEnvironment.load(env_path)
                        if env.run:
                            env.run.ended_at = datetime.now().isoformat()
                            env.run.status = status
                            env.save(env_path)
                    except Exception as e:
                        logger.warning("Failed to update environment.json: %s", e)

            # Note: Don't stop scheduler - it's shared!

            if self.taskOutputsWorker is not None:
                logger.info("Stopping tasks outputs worker")
                self.taskOutputsWorker.queue.put(None)

            self.workspace.__exit__(exc_type, exc_value, traceback)
            if self.xplock:
                self.xplock.__exit__(exc_type, exc_value, traceback)

            # Put back old experiment as current one
            experiment.CURRENT = self.old_experiment

        if self.workspace.run_mode == RunMode.NORMAL:
            # Remove job directories for transient jobs with REMOVE mode
            if exc_type is None:
                for job in list(self.scheduler.jobs.values()):
                    if (
                        self in job.experiments
                        and job.transient.should_remove
                        and job.state.finished()
                    ):
                        job_path = job.path
                        if job_path.exists():
                            logger.info(
                                "Removing transient job directory: %s", job_path
                            )
                            rmtree(job_path)
                        # Also remove the symlink in the experiment's jobs folder
                        symlink_path = self.jobspath / job.relpath
                        if symlink_path.is_symlink():
                            symlink_path.unlink()

            # Cleanup old runs based on history settings
            try:
                cleanup_experiment_history(
                    self._experiment_base,
                    current_run_id=self.run_id,
                    current_status=status,
                    history=self._get_history_settings(),
                )
            except Exception as e:
                logger.warning("Failed to cleanup old runs: %s", e)

        # Suppress GracefulExperimentExit exception
        if exc_type is GracefulExperimentExit:
            return True

    async def update_task_output_count(self, delta: int):
        """Change in the number of task outputs to process"""
        async with self.scheduler.exitCondition:
            self.taskOutputQueueSize += delta
            logging.debug(
                "Updating queue size with %d => %d", delta, self.taskOutputQueueSize
            )
            if self.taskOutputQueueSize == 0:
                self.scheduler.exitCondition.notify_all()

    def watch_output(self, watched: "WatchedOutput"):
        """Watch an output

        :param watched: The watched output specification
        """

        self.taskOutputsWorker.watch_output(watched)

    def add_service(self, service: ServiceClass) -> ServiceClass:
        """Adds a service (e.g. tensorboard viewer) to the experiment

        :param service: A service instance
        :return: The same service instance (or existing service if already added)
        """
        existing = self.services.get(service.id)
        if existing is not None:
            if existing is service:
                # Same service instance added twice - just return it
                logger.debug("Service %s already added, ignoring duplicate", service.id)
                return service
            else:
                # Different service with same id - warn and replace
                logger.warning(
                    "Replacing service %s (old id=%s, new id=%s)",
                    service.id,
                    id(existing),
                    id(service),
                )

        self._services[service.id] = service

        # Allow service to access experiment context
        service.set_experiment(self)

        # Register state listener for state changes (writes events)
        if self._state_listener is not None:
            service.add_listener(self._state_listener)

        # Register listener for state changes
        service.add_listener(self)

        self.scheduler.notify_service_add(service, self.name, self.run_id or "")

        return service

    def service_state_changed(self, service):
        """Called when a service state changes - notify listeners"""
        state_name = service.state.name if hasattr(service.state, "name") else "UNKNOWN"
        logger.debug(
            "Service %s state changed to %s (experiment=%s)",
            service.id,
            state_name,
            self.name,
        )

        # Notify state listeners (for TUI tab title updates etc.)
        from experimaestro.scheduler.state_status import ServiceStateChangedEvent

        if self.scheduler is not None:
            event = ServiceStateChangedEvent(
                experiment_id=self.name,
                run_id=self.run_id or "",
                service_id=service.id,
                state=state_name,
            )
            self.scheduler._notify_state_listeners_async(event)

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

    def load(self, reference: str, name: str = "default", run_id: str = None):
        """Loads configuration objects from an experimental directory.

        :param reference: The name of the experiment
        :param name: The name of the saving directory (default to `default`)
        :param run_id: The run ID to load from (default: latest run)
        """
        from experimaestro import load

        exp_base = self.workspace.experimentspath / reference
        if run_id is None:
            # Find the latest run directory
            run_dirs = sorted(
                [d for d in exp_base.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if not run_dirs:
                raise FileNotFoundError(f"No runs found for experiment {reference}")
            run_dir = run_dirs[0]
        else:
            run_dir = exp_base / run_id

        path = run_dir / "data" / name
        return load(path)

    def _get_history_settings(self) -> HistorySettings:
        """Get the history settings for this experiment.

        Returns workspace-specific settings if available, otherwise global defaults.
        """
        # Check if workspace has explicit history settings
        ws_settings = self.workspace.settings
        if ws_settings and ws_settings.history:
            return ws_settings.history

        # Fall back to global settings
        settings = get_settings()
        return settings.history


def get_run_status(run_dir: Path) -> Optional[str]:
    """Get the status of a run from its status.json or environment.json.

    Args:
        run_dir: Path to the run directory

    Returns:
        'completed', 'failed', or None if status cannot be determined.
    """
    # Try environment.json first (most reliable - written on exit)
    env_path = run_dir / "environment.json"
    if env_path.exists():
        try:
            from experimaestro.utils.environment import ExperimentEnvironment

            env = ExperimentEnvironment.load(env_path)
            if env.run and env.run.status:
                return env.run.status
        except Exception:
            pass

    # Fall back to status.json
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            with status_path.open() as f:
                status = json.load(f)
                # Check the experiment status field
                exp_status = status.get("status")
                if exp_status == "done":
                    return "completed"
                elif exp_status == "failed":
                    return "failed"
                # Check job states as fallback
                jobs = status.get("jobs", {})
                if any(j.get("state") == "error" for j in jobs.values()):
                    return "failed"
                return "completed"
        except Exception:
            pass

    # Cannot determine status
    return None


def cleanup_experiment_history(
    experiment_base: Path,
    *,
    current_run_id: Optional[str] = None,
    current_status: Optional[str] = None,
    history: Optional[HistorySettings] = None,
) -> list[Path]:
    """Clean up old experiment runs based on history settings.

    This function can be called from the CLI or other contexts.

    Args:
        experiment_base: Path to the experiment directory (containing run subdirs)
        current_run_id: ID of the current run to exclude from cleanup (optional)
        current_status: Status of the current run ('completed' or 'failed'), used
            to determine if failed runs should be removed (optional)
        history: History settings to use (defaults to global settings)

    Returns:
        List of paths that were removed
    """
    if history is None:
        settings = get_settings()
        history = settings.history

    removed_paths = []

    # List all run directories (excluding the current one)
    run_dirs = []
    for d in experiment_base.iterdir():
        if d.is_dir() and d.name != current_run_id:
            run_dirs.append(d)

    # Sort by directory name (oldest first)
    # Directory names are in format YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS.N (with modifier)
    def run_sort_key(d: Path) -> tuple[str, int]:
        """Parse run_id for sorting, handling modifiers like 20250501_102315.1"""
        name = d.name
        if "." in name:
            parts = name.split(".", 1)
            try:
                return (parts[0], int(parts[1]))
            except (ValueError, IndexError):
                return (name, 0)
        return (name, 0)

    run_dirs.sort(key=run_sort_key)

    # Categorize runs by status
    completed_runs = []
    failed_runs = []

    for run_dir in run_dirs:
        status = get_run_status(run_dir)
        if status == "completed":
            completed_runs.append(run_dir)
        elif status == "failed":
            failed_runs.append(run_dir)
        # Runs with unknown status are not touched

    # If current run succeeded, remove all past failed runs (per user requirement)
    if current_status == "completed":
        # Remove all past failed runs
        # Per user requirement: "If an experiment succeed, it remove the past failed"
        for run_dir in failed_runs:
            logger.info("Removing failed run (experiment succeeded): %s", run_dir)
            try:
                rmtree(run_dir)
                removed_paths.append(run_dir)
            except Exception as e:
                logger.warning("Failed to remove run directory %s: %s", run_dir, e)
        failed_runs = []

    # Remove failed runs that come after any successful run
    # (if there's a success before a failure, that failure is stale)
    if completed_runs:
        # Find the newest completed run
        newest_completed = run_sort_key(completed_runs[-1])
        remaining_failed = []
        for run_dir in failed_runs:
            if run_sort_key(run_dir) < newest_completed:
                logger.info("Removing failed run (success exists after): %s", run_dir)
                try:
                    rmtree(run_dir)
                    removed_paths.append(run_dir)
                except Exception as e:
                    logger.warning("Failed to remove run directory %s: %s", run_dir, e)
            else:
                remaining_failed.append(run_dir)
        failed_runs = remaining_failed

    # Keep only max_done completed runs (remove oldest ones)
    while len(completed_runs) > history.max_done:
        run_dir = completed_runs.pop(0)  # Remove oldest
        logger.info(
            "Removing old completed run (keeping %d): %s", history.max_done, run_dir
        )
        try:
            rmtree(run_dir)
            removed_paths.append(run_dir)
        except Exception as e:
            logger.warning("Failed to remove run directory %s: %s", run_dir, e)

    # Keep only max_failed failed runs (remove oldest ones)
    while len(failed_runs) > history.max_failed:
        run_dir = failed_runs.pop(0)  # Remove oldest
        logger.info(
            "Removing old failed run (keeping %d): %s", history.max_failed, run_dir
        )
        try:
            rmtree(run_dir)
            removed_paths.append(run_dir)
        except Exception as e:
            logger.warning("Failed to remove run directory %s: %s", run_dir, e)

    return removed_paths


# re-export at the module level
current = experiment.current
