"""State providers for accessing experiment and job information

This module provides various StateProvider implementations for accessing experiment
state from different sources (active scheduler, offline databases, SSH remotes).
"""

import abc
import json
import logging
import fasteners
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from experimaestro.scheduler.state_db import (
    JobModel,
    ServiceModel,
    initialize_database,
    close_database,
)

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.scheduler.services import Service
    from experimaestro.scheduler.base import Scheduler

logger = logging.getLogger("xpm.state")


class StateProvider(abc.ABC):
    """Abstract base class for accessing experiment and job state

    Different implementations provide access to state from different sources:
    - SchedulerStateProvider: Active scheduler (real-time updates)
    - WorkspaceStateProvider: Workspace databases (read-only, both active and completed experiments)
    - SSHClientStateProvider: Remote access via SSH
    """

    @abc.abstractmethod
    def get_experiments(self) -> List[Dict]:
        """Get list of all experiments

        Returns:
            List of experiment dictionaries with keys:
            - experiment_id: Unique identifier
            - workdir: Workspace directory path
            - total_jobs: Total number of jobs
            - finished_jobs: Number of completed jobs
            - failed_jobs: Number of failed jobs
        """
        pass

    @abc.abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get a specific experiment by ID

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment dictionary or None if not found
        """
        pass

    @abc.abstractmethod
    def get_jobs(self, experiment_id: str, task_id: Optional[str] = None) -> List[Dict]:
        """Get jobs for an experiment, optionally filtered by task_id

        Args:
            experiment_id: Experiment identifier
            task_id: Optional task class identifier to filter by

        Returns:
            List of job dictionaries
        """
        pass

    @abc.abstractmethod
    def get_job(self, experiment_id: str, job_id: str) -> Optional[Dict]:
        """Get a specific job

        Args:
            experiment_id: Experiment identifier
            job_id: Job identifier

        Returns:
            Job dictionary or None if not found
        """
        pass

    @abc.abstractmethod
    def get_services(self, experiment_id: str) -> List[Dict]:
        """Get services for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of service dictionaries
        """
        pass

    def kill_job(self, experiment_id: str, job_id: str):
        """Kill a running job

        Args:
            experiment_id: Experiment identifier
            job_id: Job identifier

        Raises:
            NotImplementedError: If this provider doesn't support killing jobs
        """
        raise NotImplementedError("This state provider does not support killing jobs")

    @abc.abstractmethod
    def close(self):
        """Close any open connections or resources"""
        pass


class ExperimentStateProvider:
    """Provides access to state for a single experiment via SQLite database

    Each experiment has its own database at: workdir/xp/{experiment_name}/experiment.db

    This class can operate in two modes:
    - Read-only: For offline viewing of completed experiments
    - Read-write: For active experiments being tracked by scheduler
    """

    def __init__(self, workdir: Path, experiment_name: str, read_only: bool = False):
        """Initialize state provider for an experiment

        Args:
            workdir: Root workspace directory
            experiment_name: Name of the experiment
            read_only: If True, open database in read-only mode
        """
        self.workdir = workdir
        self.experiment_name = experiment_name
        self.read_only = read_only

        # Database path: workdir/xp/{experiment_name}/experiment.db
        self.experiment_dir = workdir / "xp" / experiment_name
        self.db_path = self.experiment_dir / "experiment.db"
        self.state_json_path = self.experiment_dir / "state.json"
        self.lock_path = self.experiment_dir / "state.lock"

        # Initialize database connection
        self.db = None
        if self.db_path.exists() or not read_only:
            self._open_database()
        elif read_only and self.state_json_path.exists():
            # Migrate from legacy state.json
            logger.info(
                "Migrating experiment %s from state.json to database",
                experiment_name,
            )
            self._migrate_from_state_json()
            self._open_database()

    def _open_database(self):
        """Open database connection"""
        self.db = initialize_database(self.db_path, read_only=self.read_only)
        logger.debug(
            "Opened database for experiment %s (read_only=%s)",
            self.experiment_name,
            self.read_only,
        )

    def _migrate_from_state_json(self):
        """Migrate from legacy state.json to database

        This reads the state.json file and populates the database with job information.
        Uses file locking to prevent race conditions with active experiments.
        """
        if not self.state_json_path.exists():
            logger.warning(
                "No state.json found for migration at %s", self.state_json_path
            )
            return

        # Ensure lock directory exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Lock the experiment directory to prevent concurrent access
            with fasteners.InterProcessLock(self.lock_path):
                # Check again if state.json still exists (might have been deleted)
                if not self.state_json_path.exists():
                    logger.warning("state.json disappeared during lock acquisition")
                    return

                # Temporarily open database in write mode for migration
                initialize_database(self.db_path, read_only=False)

                # Read state.json
                from experimaestro.core.serialization import from_state_dict

                with self.state_json_path.open("rt") as fh:
                    content = json.load(fh)

                # Parse job information
                job_states = from_state_dict(content, as_instance=False)

                # Create job records in database
                for job_dict in job_states:
                    # Extract dependency job IDs
                    dep_ids = job_dict.get("depends_on", [])

                    JobModel.create(
                        job_id=job_dict["id"],
                        task_id=job_dict["task"].__xpmtype__.identifier,
                        locator=job_dict["id"],
                        path=str(job_dict["path"]),
                        state="done",  # Assume completed since in state.json
                        tags=json.dumps(job_dict.get("tags", {})),
                        dependencies=json.dumps(dep_ids),
                    )

                logger.info("Migrated %d jobs from state.json", len(job_states))

                # Close temporary database
                close_database()

        except Exception as e:
            logger.exception("Failed to migrate from state.json: %s", e)
            # Clean up partial database on error
            if self.db_path.exists():
                self.db_path.unlink()
            raise

    def close(self):
        """Close database connection"""
        if self.db:
            close_database()
            self.db = None

    # Read methods

    def get_experiment_info(self) -> Dict:
        """Get experiment metadata

        Returns:
            Dictionary with experiment information
        """
        # Count jobs by state
        total_jobs = JobModel.select().count()
        finished_jobs = JobModel.select().where((JobModel.state == "done")).count()
        failed_jobs = JobModel.select().where((JobModel.state == "error")).count()

        return {
            "experiment_id": self.experiment_name,
            "workdir": str(self.workdir),
            "total_jobs": total_jobs,
            "finished_jobs": finished_jobs,
            "failed_jobs": failed_jobs,
        }

    def get_jobs(self, task_id: Optional[str] = None) -> List[Dict]:
        """Get all jobs, optionally filtered by task_id

        Args:
            task_id: If provided, filter jobs by this task class identifier

        Returns:
            List of job dictionaries
        """
        query = JobModel.select()
        if task_id:
            query = query.where(JobModel.task_id == task_id)

        jobs = []
        for job_model in query:
            jobs.append(self._job_model_to_dict(job_model))

        return jobs

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get a specific job by ID

        Args:
            job_id: Job identifier

        Returns:
            Job dictionary or None if not found
        """
        try:
            job_model = JobModel.get(JobModel.job_id == job_id)
            return self._job_model_to_dict(job_model)
        except JobModel.DoesNotExist:
            return None

    def get_services(self) -> List[Dict]:
        """Get all services for this experiment

        Returns:
            List of service dictionaries
        """
        services = []
        for service_model in ServiceModel.select():
            services.append(
                {
                    "service_id": service_model.service_id,
                    "description": service_model.description,
                    "state": service_model.state,
                    "created_at": service_model.created_at.isoformat(),
                    "updated_at": service_model.updated_at.isoformat(),
                }
            )
        return services

    def _job_model_to_dict(self, job_model: JobModel) -> Dict:
        """Convert a JobModel to a dictionary in UI format

        Args:
            job_model: JobModel instance

        Returns:
            Job dictionary compatible with UI expectations (camelCase keys)
        """
        # Parse JSON fields
        tags_dict = json.loads(job_model.tags)
        progress_list = json.loads(job_model.progress)

        return {
            # UI expects camelCase keys
            "jobId": job_model.job_id,
            "taskId": job_model.task_id,
            "locator": job_model.locator,
            "status": job_model.state,  # UI expects "status" not "state"
            "submitted": self._format_time(job_model.submitted_time),
            "start": self._format_time(job_model.started_time),
            "end": self._format_time(job_model.ended_time),
            "progress": progress_list,
            "tags": list(tags_dict.items()),  # UI expects list of [key, value] tuples
            "experimentIds": [self.experiment_name],  # Job belongs to this experiment
        }

    def _format_time(self, timestamp: Optional[float]) -> str:
        """Format timestamp for UI

        Args:
            timestamp: Unix timestamp or None

        Returns:
            ISO format datetime string or empty string
        """
        if not timestamp:
            return ""
        from datetime import datetime

        return datetime.fromtimestamp(timestamp).isoformat()

    # Write methods (only available in non-read-only mode)

    def update_job_submitted(self, job: "Job"):
        """Record that a job has been submitted

        Args:
            job: Job instance

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update job in read-only mode")

        from experimaestro.scheduler.jobs import JobDependency

        # Extract dependency job IDs
        dep_ids = [
            dep.origin.identifier
            for dep in job.dependencies
            if isinstance(dep, JobDependency)
        ]

        # Create or update job record
        JobModel.insert(
            job_id=job.identifier,
            task_id=str(job.type.identifier),
            locator=job.identifier,
            path=str(job.path),
            state=job.state.name,
            submitted_time=job.submittime,
            tags=json.dumps(job.tags),
            dependencies=json.dumps(dep_ids),
        ).on_conflict(
            conflict_target=[JobModel.job_id],
            update={
                JobModel.state: job.state.name,
                JobModel.submitted_time: job.submittime,
            },
        ).execute()

        logger.debug("Recorded job submission: %s", job.identifier)

    def update_job_state(self, job: "Job"):
        """Update the state of a job

        Args:
            job: Job instance

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update job in read-only mode")

        # Build update dict
        update_data = {
            JobModel.state: job.state.name,
        }

        # Add failure reason if available
        from experimaestro.scheduler.jobs import JobStateError

        if isinstance(job.state, JobStateError) and job.state.failure_reason:
            update_data[JobModel.failure_reason] = job.state.failure_reason.name

        # Add timing information
        if job.starttime:
            update_data[JobModel.started_time] = job.starttime
        if job.endtime:
            update_data[JobModel.ended_time] = job.endtime

        # Add progress information
        if job._progress:
            update_data[JobModel.progress] = json.dumps(
                [
                    {"level": p.level, "progress": p.progress, "desc": p.desc}
                    for p in job._progress
                ]
            )

        # Update the job record
        JobModel.update(update_data).where(JobModel.job_id == job.identifier).execute()

        logger.debug("Updated job state: %s -> %s", job.identifier, job.state.name)

    def update_service(self, service: "Service"):
        """Update service information

        Args:
            service: Service instance

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update service in read-only mode")

        from datetime import datetime

        ServiceModel.insert(
            service_id=service.id,
            description=service.description(),
            state=service.state.name,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ).on_conflict(
            conflict_target=[ServiceModel.service_id],
            update={
                ServiceModel.description: service.description(),
                ServiceModel.state: service.state.name,
                ServiceModel.updated_at: datetime.now(),
            },
        ).execute()

        logger.debug("Updated service: %s", service.id)


class SchedulerStateProvider(StateProvider):
    """Provides access to state from an active scheduler

    This implementation wraps a Scheduler instance and implements the Listener
    interface to receive real-time updates. It creates and manages
    ExperimentStateProvider instances for each experiment.

    When a job belongs to multiple experiments, updates are fanned out to all
    relevant ExperimentStateProviders.
    """

    def __init__(self, scheduler: "Scheduler"):
        """Initialize with a scheduler instance

        Args:
            scheduler: The scheduler to wrap
        """
        self.scheduler = scheduler
        self.experiment_providers: Dict[str, ExperimentStateProvider] = {}

        # Register as a listener with the scheduler
        self.scheduler.addlistener(self)

        logger.info("SchedulerStateProvider initialized")

    # Listener interface implementation

    def job_submitted(self, job: "Job"):
        """Called when a job is submitted

        Args:
            job: The submitted job
        """
        # Update all experiments this job belongs to
        logger.debug(
            "job_submitted called for job %s with %d experiments",
            job.identifier,
            len(job.experiments),
        )
        for experiment in job.experiments:
            logger.debug("Processing experiment %s", experiment.workdir.name)
            try:
                provider = self._get_or_create_provider(experiment.workdir.name)
                provider.update_job_submitted(job)
            except Exception as e:
                logger.exception(
                    "Error updating job submission in experiment %s: %s",
                    experiment.workdir.name,
                    e,
                )

    def job_state(self, job: "Job"):
        """Called when a job's state changes

        Args:
            job: The job with updated state
        """
        # Update all experiments this job belongs to
        for experiment in job.experiments:
            provider = self._get_or_create_provider(experiment.workdir.name)
            try:
                provider.update_job_state(job)
            except Exception as e:
                logger.exception(
                    "Error updating job state in experiment %s: %s",
                    experiment.workdir.name,
                    e,
                )

    def service_add(self, service: "Service"):
        """Called when a service is added

        Args:
            service: The added service
        """
        # Services are associated with experiments through the scheduler
        # We update all registered experiment providers
        for provider in self.experiment_providers.values():
            try:
                provider.update_service(service)
            except Exception as e:
                logger.exception("Error updating service: %s", e)

    # StateProvider interface implementation

    def get_experiments(self) -> List[Dict]:
        """Get list of all experiments

        Returns:
            List of experiment dictionaries
        """
        experiments = []
        for experiment_name, provider in self.experiment_providers.items():
            info = provider.get_experiment_info()
            experiments.append(info)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get a specific experiment by ID

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment dictionary or None if not found
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_experiment_info()
        return None

    def get_jobs(self, experiment_id: str, task_id: Optional[str] = None) -> List[Dict]:
        """Get jobs for an experiment

        Args:
            experiment_id: Experiment identifier
            task_id: Optional task class identifier to filter by

        Returns:
            List of job dictionaries
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_jobs(task_id=task_id)
        return []

    def get_job(self, experiment_id: str, job_id: str) -> Optional[Dict]:
        """Get a specific job

        Args:
            experiment_id: Experiment identifier
            job_id: Job identifier

        Returns:
            Job dictionary or None if not found
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_job(job_id)
        return None

    def get_services(self, experiment_id: str) -> List[Dict]:
        """Get services for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of service dictionaries
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_services()
        return []

    def kill_job(self, experiment_id: str, job_id: str):
        """Kill a running job

        Args:
            experiment_id: Experiment identifier
            job_id: Job identifier

        Raises:
            ValueError: If job not found
        """
        # Find the job in the scheduler
        job = self.scheduler.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Kill the job process
        if job._process:
            job._process.kill()
            logger.info("Killed job %s", job_id)
        else:
            logger.warning("Job %s has no process to kill", job_id)

    def close(self):
        """Close all experiment providers"""
        # Unregister from scheduler
        self.scheduler.removelistener(self)

        # Close all experiment providers
        for provider in self.experiment_providers.values():
            provider.close()

        self.experiment_providers.clear()
        logger.info("SchedulerStateProvider closed")

    # Helper methods

    def _get_or_create_provider(self, experiment_name: str) -> ExperimentStateProvider:
        """Get or create an ExperimentStateProvider for an experiment

        Args:
            experiment_name: Name of the experiment

        Returns:
            ExperimentStateProvider instance
        """
        if experiment_name not in self.experiment_providers:
            # Find the experiment in the scheduler
            experiment = self.scheduler.experiments.get(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment {experiment_name} not found")

            # Create provider
            provider = ExperimentStateProvider(
                experiment.workdir.parent.parent,  # Get workspace root
                experiment_name,
                read_only=False,
            )
            self.experiment_providers[experiment_name] = provider
            logger.debug("Created ExperimentStateProvider for %s", experiment_name)

        return self.experiment_providers[experiment_name]


class WorkspaceStateProvider(StateProvider):
    """Provides access to state from experiments in a workspace

    This implementation scans a workspace directory for experiments (both active
    and completed) and provides read-only access to their state. It can optionally
    watch for new experiments using filesystem monitoring.

    Unlike SchedulerStateProvider which wraps an active scheduler, this provider
    works by reading from the persisted databases directly.
    """

    def __init__(self, workdir: Path, watch: bool = False):
        """Initialize workspace state provider

        Args:
            workdir: Root workspace directory
            watch: If True, watch filesystem for new experiments
        """
        self.workdir = workdir
        self.watch = watch
        self.experiment_providers: Dict[str, ExperimentStateProvider] = {}
        self.watcher = None

        # Initial scan
        self.scan_experiments()

        # Start watching if requested
        if watch:
            self._start_watching()

        logger.info("WorkspaceStateProvider initialized for %s", workdir)

    def scan_experiments(self):
        """Scan workspace for experiments and create providers"""
        experiments_dir = self.workdir / "xp"

        if not experiments_dir.exists():
            logger.debug("No experiments directory found at %s", experiments_dir)
            return

        # Scan for experiment directories
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            experiment_name = exp_dir.name

            # Check if experiment has state.json or experiment.db
            has_state = (exp_dir / "state.json").exists() or (
                exp_dir / "experiment.db"
            ).exists()

            if has_state and experiment_name not in self.experiment_providers:
                try:
                    provider = ExperimentStateProvider(
                        self.workdir, experiment_name, read_only=True
                    )
                    self.experiment_providers[experiment_name] = provider
                    logger.debug("Discovered experiment: %s", experiment_name)
                except Exception as e:
                    logger.exception(
                        "Failed to create provider for experiment %s: %s",
                        experiment_name,
                        e,
                    )

    def _start_watching(self):
        """Start filesystem watching for new experiments"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class ExperimentWatcher(FileSystemEventHandler):
            def __init__(self, provider: "WorkspaceStateProvider"):
                self.provider = provider
                self.debounce_time = 1.0  # 1 second debounce
                self.last_scan = 0

            def on_created(self, event):
                # Debounce: don't scan too frequently
                import time

                now = time.time()
                if now - self.last_scan > self.debounce_time:
                    self.last_scan = now
                    logger.debug("Filesystem change detected, rescanning experiments")
                    self.provider.scan_experiments()

            def on_modified(self, event):
                self.on_created(event)

        experiments_dir = self.workdir / "xp"
        if experiments_dir.exists():
            self.watcher = Observer()
            handler = ExperimentWatcher(self)
            self.watcher.schedule(handler, str(experiments_dir), recursive=True)
            self.watcher.start()
            logger.info("Started watching %s for new experiments", experiments_dir)

    # StateProvider interface implementation

    def get_experiments(self) -> List[Dict]:
        """Get list of all experiments

        Returns:
            List of experiment dictionaries
        """
        experiments = []
        for experiment_name, provider in self.experiment_providers.items():
            info = provider.get_experiment_info()
            experiments.append(info)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get a specific experiment by ID

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment dictionary or None if not found
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_experiment_info()
        return None

    def get_jobs(self, experiment_id: str, task_id: Optional[str] = None) -> List[Dict]:
        """Get jobs for an experiment

        Args:
            experiment_id: Experiment identifier
            task_id: Optional task class identifier to filter by

        Returns:
            List of job dictionaries
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_jobs(task_id=task_id)
        return []

    def get_job(self, experiment_id: str, job_id: str) -> Optional[Dict]:
        """Get a specific job

        Args:
            experiment_id: Experiment identifier
            job_id: Job identifier

        Returns:
            Job dictionary or None if not found
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_job(job_id)
        return None

    def get_services(self, experiment_id: str) -> List[Dict]:
        """Get services for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of service dictionaries
        """
        provider = self.experiment_providers.get(experiment_id)
        if provider:
            return provider.get_services()
        return []

    def kill_job(self, experiment_id: str, job_id: str):
        """Kill a running job - not supported in workspace mode

        Args:
            experiment_id: Experiment identifier
            job_id: Job identifier

        Raises:
            NotImplementedError: Always (read-only, no scheduler access)
        """
        raise NotImplementedError(
            "Cannot kill jobs in workspace mode (no scheduler access)"
        )

    def close(self):
        """Close all experiment providers and stop watching"""
        # Stop watcher if running
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
            logger.info("Stopped filesystem watcher")

        # Close all experiment providers
        for provider in self.experiment_providers.values():
            provider.close()

        self.experiment_providers.clear()
        logger.info("WorkspaceStateProvider closed")
