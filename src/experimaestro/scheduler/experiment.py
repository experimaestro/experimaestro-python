import asyncio
import json
import logging
import os
from pathlib import Path
import time
from shutil import rmtree
from typing import Any, Dict, Optional, TypeVar, Union

from experimaestro.core.objects import WatchedOutput
from experimaestro.exceptions import HandledException

from experimaestro.scheduler.signal_handler import SIGNAL_HANDLER
from experimaestro.scheduler.jobs import Job
from experimaestro.scheduler.services import Service
from experimaestro.scheduler.workspace import RunMode, Workspace
from experimaestro.settings import WorkspaceSettings, get_settings
from experimaestro.utils import logger

ServiceClass = TypeVar("ServiceClass", bound=Service)


class FailedExperiment(HandledException):
    """Raised when an experiment failed"""

    pass


class DatabaseListener:
    """Listener that updates job state in the database"""

    def __init__(self, state_provider, experiment_id: str, run_id: str):
        self.state_provider = state_provider
        self.experiment_id = experiment_id
        self.run_id = run_id

    def job_submitted(self, job):
        # Already handled in experiment.add_job()
        pass

    def job_state(self, job):
        """Update job state in database"""
        self.state_provider.update_job_state(job, self.experiment_id, self.run_id)

    def service_add(self, service):
        pass


class experiment:
    """Main experiment object

    It is a context object, i.e. an experiment is run with

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
        register_signals: bool = True,
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
        """

        from experimaestro.scheduler import Listener, Scheduler

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
        self._register_signals = register_signals

        # Get configuration settings

        if host is not None:
            settings.server.host = host

        if port is not None:
            settings.server.port = port

        if token is not None:
            settings.server.token = token

        # Use singleton scheduler
        self.scheduler = Scheduler.instance()

        # Determine if we need a server
        self._needs_server = (
            settings.server.port is not None and settings.server.port >= 0
        ) and self.workspace.run_mode == RunMode.NORMAL
        self._server_settings = settings.server if self._needs_server else None

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

    @property
    def alt_jobspaths(self):
        """Return potential other directories"""
        for alt_workdir in self.workspace.alt_workdirs:
            yield alt_workdir / "jobs"

    @property
    def jobsbakpath(self):
        """Return the directory in which results can be stored for this experiment"""
        return self.workdir / "jobs.bak"

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
                self.workdir.name,
                self.unfinishedJobs,
            )

        record = {
            "job_id": job.identifier,
            "task_id": str(job.type.identifier),
            "tags": dict(job.tags.items()) if job.tags else {},
            "timestamp": time.time(),
        }

        with self.jobs_jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        # Also register in database for TUI/monitoring
        experiment_id = self.workdir.name
        self.state_provider.update_job_submitted(job, experiment_id, self.run_id)

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
        from experimaestro.utils.environment import save_environment_info

        if self.workspace.run_mode != RunMode.DRY_RUN:
            logger.info("Locking experiment %s", self.xplockpath)
            self.xplock = self.workspace.connector.lock(self.xplockpath, 0).__enter__()
            logger.info("Experiment locked")

        # Capture and save environment info (git info for editable packages + all package versions)
        if self.workspace.run_mode == RunMode.NORMAL:
            env_info_path = self.workdir / "environment.json"
            save_environment_info(env_info_path)

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

        # Register experiment with scheduler
        self.scheduler.register_experiment(self)

        # Start server via scheduler if needed
        if self._needs_server:
            self.scheduler.start_server(self._server_settings, workspace=self.workspace)

        self.workspace.__enter__()
        (self.workspace.path / ".__experimaestro__").touch()

        # Initialize workspace state provider (singleton per workspace path)
        from .state_provider import WorkspaceStateProvider

        self.state_provider = WorkspaceStateProvider.get_instance(
            self.workspace.path,
            read_only=False,
            sync_on_start=False,  # Experiments don't sync on start
        )

        # Register experiment in database and create a run
        experiment_id = self.workdir.name
        self.state_provider.ensure_experiment(experiment_id)
        self.run_id = self.state_provider.create_run(experiment_id)

        # Add database listener to update job state in database
        self._db_listener = DatabaseListener(
            self.state_provider, experiment_id, self.run_id
        )
        self.scheduler.listeners.add(self._db_listener)

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
            if self._register_signals:
                SIGNAL_HANDLER.remove(self)

            # Stop services
            for service in self.services.values():
                logger.info("Closing service %s", service.description())
                service.stop()

            # Unregister experiment from scheduler
            self.scheduler.unregister_experiment(self)

            # Remove database listener
            self.scheduler.listeners.discard(self._db_listener)

            # Mark run as completed in database
            experiment_id = self.workdir.name
            status = "failed" if exc_type else "completed"
            self.state_provider.complete_run(experiment_id, self.run_id, status)

            # Note: Don't stop scheduler - it's shared!
            # Note: Don't stop server - it runs in daemon mode until program exit

            if self.taskOutputsWorker is not None:
                logger.info("Stopping tasks outputs worker")
                self.taskOutputsWorker.queue.put(None)

            self.workspace.__exit__(exc_type, exc_value, traceback)
            if self.xplock:
                self.xplock.__exit__(exc_type, exc_value, traceback)

            # Put back old experiment as current one
            experiment.CURRENT = self.old_experiment

        if self.workspace.run_mode == RunMode.NORMAL:
            # Write the state
            logging.info("Saving the experiment state")
            from experimaestro.scheduler.state import ExperimentState

            ExperimentState.save(
                self.workdir / "state.json", self.scheduler.jobs.values()
            )

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


# re-export at the module level
current = experiment.current
