"""Unified workspace state provider for accessing experiment and job information

This module provides a single WorkspaceStateProvider class that accesses state
from the workspace-level database (workspace.db). This replaces the previous
multi-provider architecture with a unified approach.

Key features:
- Single workspace.db database shared across all experiments
- Support for multiple runs per experiment
- Run-scoped tags (fixes GH #128)
- Thread-safe database access via thread-local connections
- Real-time updates via scheduler listener interface
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from experimaestro.scheduler.state_db import (
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    JobTagModel,
    ServiceModel,
)

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.scheduler.services import Service

logger = logging.getLogger("xpm.state")


class WorkspaceStateProvider:
    """Unified state provider for workspace-level database (singleton per workspace path)

    Provides access to experiment and job state from a single workspace database.
    Supports both read-only (monitoring) and read-write (scheduler) modes.

    Only one WorkspaceStateProvider instance exists per workspace path. Subsequent
    requests for the same path return the existing instance.

    Thread safety:
    - Database connections are thread-local (managed by state_db module)
    - Singleton registry is protected by a lock
    - Each thread gets its own database connection

    Run tracking:
    - Each experiment can have multiple runs
    - Jobs/services are scoped to (experiment_id, run_id)
    - Tags are scoped to (job_id, experiment_id, run_id) - fixes GH #128
    """

    # Registry of state provider instances by absolute path
    _instances: Dict[Path, "WorkspaceStateProvider"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        workspace_path: Path,
        read_only: bool = False,
        sync_on_start: bool = False,
        sync_interval_minutes: int = 5,
    ) -> "WorkspaceStateProvider":
        """Get or create WorkspaceStateProvider instance for a workspace path

        Args:
            workspace_path: Root workspace directory
            read_only: If True, database is in read-only mode
            sync_on_start: If True, sync from disk on initialization
            sync_interval_minutes: Minimum interval between syncs (default: 5)

        Returns:
            WorkspaceStateProvider instance (singleton per path)
        """
        # Normalize path
        if isinstance(workspace_path, Path):
            workspace_path = workspace_path.absolute()
        else:
            workspace_path = Path(workspace_path).absolute()

        # Check if instance already exists
        with cls._lock:
            if workspace_path in cls._instances:
                return cls._instances[workspace_path]

            # Create new instance
            instance = cls(
                workspace_path, read_only, sync_on_start, sync_interval_minutes
            )
            cls._instances[workspace_path] = instance
            return instance

    def __init__(
        self,
        workspace_path: Path,
        read_only: bool = False,
        sync_on_start: bool = False,
        sync_interval_minutes: int = 5,
    ):
        """Initialize workspace state provider (called by get_instance())

        Args:
            workspace_path: Root workspace directory
            read_only: If True, database is in read-only mode
            sync_on_start: If True, sync from disk on initialization
            sync_interval_minutes: Minimum interval between syncs (default: 5)
        """
        # Normalize path
        if isinstance(workspace_path, Path):
            workspace_path = workspace_path.absolute()
        else:
            workspace_path = Path(workspace_path).absolute()

        self.workspace_path = workspace_path
        self.read_only = read_only
        self.sync_interval_minutes = sync_interval_minutes

        # Check and update workspace version
        from .workspace import WORKSPACE_VERSION

        version_file = self.workspace_path / ".__experimaestro__"

        if version_file.exists():
            # Read existing version
            content = version_file.read_text().strip()
            if content == "":
                # Empty file = v0
                workspace_version = 0
            else:
                try:
                    workspace_version = int(content)
                except ValueError:
                    raise RuntimeError(
                        f"Invalid workspace version file at {version_file}: "
                        f"expected integer, got '{content}'"
                    )

            # Check if workspace version is supported
            if workspace_version > WORKSPACE_VERSION:
                raise RuntimeError(
                    f"Workspace version {workspace_version} is not supported by "
                    f"this version of experimaestro (supports up to version "
                    f"{WORKSPACE_VERSION}). Please upgrade experimaestro."
                )
            if workspace_version < WORKSPACE_VERSION:
                raise RuntimeError(
                    f"Workspace version {workspace_version} is not supported by "
                    "this version of experimaestro (please upgrade the experimaestro "
                    "workspace)"
                )
        else:
            # New workspace - create the file
            workspace_version = WORKSPACE_VERSION

        # Write current version to file (update empty v0 workspaces)
        if not read_only and (
            not version_file.exists() or version_file.read_text().strip() == ""
        ):
            version_file.write_text(str(WORKSPACE_VERSION))

        # Initialize workspace database
        from .state_db import initialize_workspace_database

        db_path = self.workspace_path / "workspace.db"
        self.workspace_db = initialize_workspace_database(db_path, read_only=read_only)

        # Optionally sync from disk on start
        if sync_on_start:
            from .state_sync import sync_workspace_from_disk

            sync_workspace_from_disk(
                self.workspace_path,
                write_mode=not read_only,
                force=False,
                sync_interval_minutes=sync_interval_minutes,
            )

        logger.info(
            "WorkspaceStateProvider initialized (read_only=%s, workspace=%s)",
            read_only,
            workspace_path,
        )

    # Experiment management methods

    def ensure_experiment(self, experiment_id: str, workdir_path: Path):
        """Create or update experiment record

        Args:
            experiment_id: Unique identifier for the experiment
            workdir_path: Path to experiment directory
        """
        if self.read_only:
            raise RuntimeError("Cannot modify experiments in read-only mode")

        ExperimentModel.insert(
            experiment_id=experiment_id,
            workdir_path=str(workdir_path),
            created_at=datetime.now(),
        ).on_conflict(
            conflict_target=[ExperimentModel.experiment_id],
            update={ExperimentModel.workdir_path: str(workdir_path)},
        ).execute()

        logger.debug("Ensured experiment: %s", experiment_id)

    def create_run(self, experiment_id: str, run_id: Optional[str] = None) -> str:
        """Create a new run for an experiment

        Args:
            experiment_id: Experiment identifier
            run_id: Optional run ID (auto-generated from timestamp if not provided)

        Returns:
            The run_id that was created

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot create runs in read-only mode")

        # Auto-generate run_id from timestamp if not provided
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create run record
        ExperimentRunModel.insert(
            experiment_id=experiment_id,
            run_id=run_id,
            started_at=datetime.now(),
            status="active",
        ).execute()

        # Update experiment's current_run_id
        ExperimentModel.update(current_run_id=run_id).where(
            ExperimentModel.experiment_id == experiment_id
        ).execute()

        logger.info("Created run %s for experiment %s", run_id, experiment_id)
        return run_id

    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current/latest run_id for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            Current run_id or None if no runs exist
        """
        try:
            experiment = ExperimentModel.get(
                ExperimentModel.experiment_id == experiment_id
            )
            return experiment.current_run_id
        except ExperimentModel.DoesNotExist:
            return None

    def get_experiments(self) -> List[Dict]:
        """Get list of all experiments

        Returns:
            List of experiment dictionaries with keys:
            - experiment_id: Unique identifier
            - workdir_path: Workspace directory path
            - current_run_id: Current/latest run ID
            - total_jobs: Total number of jobs (for current run)
            - finished_jobs: Number of completed jobs (for current run)
            - failed_jobs: Number of failed jobs (for current run)
        """
        experiments = []

        for exp_model in ExperimentModel.select():
            # Count jobs for current run
            total_jobs = 0
            finished_jobs = 0
            failed_jobs = 0

            if exp_model.current_run_id:
                total_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.experiment_id == exp_model.experiment_id)
                        & (JobModel.run_id == exp_model.current_run_id)
                    )
                    .count()
                )
                finished_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.experiment_id == exp_model.experiment_id)
                        & (JobModel.run_id == exp_model.current_run_id)
                        & (JobModel.state == "done")
                    )
                    .count()
                )
                failed_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.experiment_id == exp_model.experiment_id)
                        & (JobModel.run_id == exp_model.current_run_id)
                        & (JobModel.state == "error")
                    )
                    .count()
                )

            experiments.append(
                {
                    "experiment_id": exp_model.experiment_id,
                    "workdir_path": exp_model.workdir_path,
                    "current_run_id": exp_model.current_run_id,
                    "total_jobs": total_jobs,
                    "finished_jobs": finished_jobs,
                    "failed_jobs": failed_jobs,
                }
            )

        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get a specific experiment by ID

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment dictionary or None if not found
        """
        try:
            exp_model = ExperimentModel.get(
                ExperimentModel.experiment_id == experiment_id
            )
        except ExperimentModel.DoesNotExist:
            return None

        # Count jobs for current run
        total_jobs = 0
        finished_jobs = 0
        failed_jobs = 0

        if exp_model.current_run_id:
            total_jobs = (
                JobModel.select()
                .where(
                    (JobModel.experiment_id == exp_model.experiment_id)
                    & (JobModel.run_id == exp_model.current_run_id)
                )
                .count()
            )
            finished_jobs = (
                JobModel.select()
                .where(
                    (JobModel.experiment_id == exp_model.experiment_id)
                    & (JobModel.run_id == exp_model.current_run_id)
                    & (JobModel.state == "done")
                )
                .count()
            )
            failed_jobs = (
                JobModel.select()
                .where(
                    (JobModel.experiment_id == exp_model.experiment_id)
                    & (JobModel.run_id == exp_model.current_run_id)
                    & (JobModel.state == "error")
                )
                .count()
            )

        return {
            "experiment_id": exp_model.experiment_id,
            "workdir_path": exp_model.workdir_path,
            "current_run_id": exp_model.current_run_id,
            "total_jobs": total_jobs,
            "finished_jobs": finished_jobs,
            "failed_jobs": failed_jobs,
        }

    def get_experiment_runs(self, experiment_id: str) -> List[Dict]:
        """Get all runs for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of run dictionaries with keys:
            - experiment_id: Experiment ID
            - run_id: Run ID
            - started_at: When run started
            - ended_at: When run completed (None if active)
            - status: Run status (active, completed, failed, abandoned)
        """
        runs = []
        for run_model in (
            ExperimentRunModel.select()
            .where(ExperimentRunModel.experiment_id == experiment_id)
            .order_by(ExperimentRunModel.started_at.desc())
        ):
            runs.append(
                {
                    "experiment_id": run_model.experiment_id,
                    "run_id": run_model.run_id,
                    "started_at": run_model.started_at.isoformat(),
                    "ended_at": (
                        run_model.ended_at.isoformat() if run_model.ended_at else None
                    ),
                    "status": run_model.status,
                }
            )
        return runs

    def complete_run(self, experiment_id: str, run_id: str, status: str = "completed"):
        """Mark a run as completed

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            status: Final status (completed, failed, abandoned)

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot modify runs in read-only mode")

        ExperimentRunModel.update(ended_at=datetime.now(), status=status).where(
            (ExperimentRunModel.experiment_id == experiment_id)
            & (ExperimentRunModel.run_id == run_id)
        ).execute()

        logger.info("Marked run %s/%s as %s", experiment_id, run_id, status)

    # Job operations

    def get_jobs(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Query jobs with optional filters

        Args:
            experiment_id: Filter by experiment (None = all experiments)
            run_id: Filter by run (None = current run if experiment_id provided)
            task_id: Filter by task class identifier
            state: Filter by job state
            tags: Filter by tags (all tags must match)

        Returns:
            List of job dictionaries (UI format with camelCase keys)
        """
        # Build base query
        query = JobModel.select()

        # Apply experiment filter
        if experiment_id is not None:
            # If experiment_id provided but not run_id, use current run
            if run_id is None:
                current_run = self.get_current_run(experiment_id)
                if current_run is None:
                    return []  # No runs exist for this experiment
                run_id = current_run

            query = query.where(
                (JobModel.experiment_id == experiment_id) & (JobModel.run_id == run_id)
            )

        # Apply task_id filter
        if task_id is not None:
            query = query.where(JobModel.task_id == task_id)

        # Apply state filter
        if state is not None:
            query = query.where(JobModel.state == state)

        # Apply tag filters
        if tags:
            for tag_key, tag_value in tags.items():
                # Join with JobTagModel for each tag filter
                query = query.join(
                    JobTagModel,
                    on=(
                        (JobTagModel.job_id == JobModel.job_id)
                        & (JobTagModel.experiment_id == JobModel.experiment_id)
                        & (JobTagModel.run_id == JobModel.run_id)
                        & (JobTagModel.tag_key == tag_key)
                        & (JobTagModel.tag_value == tag_value)
                    ),
                )

        # Execute query and convert to dictionaries
        jobs = []
        for job_model in query:
            # Get tags for this job
            job_tags = self._get_job_tags(
                job_model.job_id, job_model.experiment_id, job_model.run_id
            )

            jobs.append(self._job_model_to_dict(job_model, job_tags))

        return jobs

    def get_job(
        self, job_id: str, experiment_id: str, run_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Get a specific job

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current run)

        Returns:
            Job dictionary or None if not found
        """
        # Use current run if not specified
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return None

        try:
            job_model = JobModel.get(
                (JobModel.job_id == job_id)
                & (JobModel.experiment_id == experiment_id)
                & (JobModel.run_id == run_id)
            )
        except JobModel.DoesNotExist:
            return None

        # Get tags for this job
        job_tags = self._get_job_tags(job_id, experiment_id, run_id)

        return self._job_model_to_dict(job_model, job_tags)

    def update_job_submitted(self, job: "Job", experiment_id: str, run_id: str):
        """Record that a job has been submitted

        Args:
            job: Job instance
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update jobs in read-only mode")

        # Create or update job record
        JobModel.insert(
            job_id=job.identifier,
            experiment_id=experiment_id,
            run_id=run_id,
            task_id=str(job.type.identifier),
            locator=job.identifier,
            path=str(job.path),
            state=job.state.name,
            submitted_time=job.submittime,
        ).on_conflict(
            conflict_target=[JobModel.job_id, JobModel.experiment_id, JobModel.run_id],
            update={
                JobModel.state: job.state.name,
                JobModel.submitted_time: job.submittime,
            },
        ).execute()

        # Update tags (run-scoped)
        self.update_job_tags(job.identifier, experiment_id, run_id, job.tags)

        logger.debug(
            "Recorded job submission: %s (experiment=%s, run=%s)",
            job.identifier,
            experiment_id,
            run_id,
        )

    def update_job_state(self, job: "Job", experiment_id: str, run_id: str):
        """Update the state of a job

        Args:
            job: Job instance
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update jobs in read-only mode")

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
        JobModel.update(update_data).where(
            (JobModel.job_id == job.identifier)
            & (JobModel.experiment_id == experiment_id)
            & (JobModel.run_id == run_id)
        ).execute()

        logger.debug(
            "Updated job state: %s -> %s (experiment=%s, run=%s)",
            job.identifier,
            job.state.name,
            experiment_id,
            run_id,
        )

    def update_job_tags(
        self, job_id: str, experiment_id: str, run_id: str, tags_dict: Dict[str, str]
    ):
        """Update tags for a job (run-scoped - fixes GH #128)

        Deletes existing tags for this (job_id, experiment_id, run_id) combination
        and inserts new tags. This ensures that the same job in different runs can
        have different tags.

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier
            tags_dict: Dictionary of tag key-value pairs

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update tags in read-only mode")

        # Delete existing tags for this job/experiment/run
        JobTagModel.delete().where(
            (JobTagModel.job_id == job_id)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ).execute()

        # Insert new tags
        if tags_dict:
            tag_records = [
                {
                    "job_id": job_id,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "tag_key": key,
                    "tag_value": value,
                }
                for key, value in tags_dict.items()
            ]
            JobTagModel.insert_many(tag_records).execute()

        logger.debug(
            "Updated tags for job %s (experiment=%s, run=%s): %s",
            job_id,
            experiment_id,
            run_id,
            tags_dict,
        )

    def delete_job(self, job_id: str, experiment_id: str, run_id: str):
        """Remove a job and its tags

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot delete jobs in read-only mode")

        # Delete tags first (foreign key constraint)
        JobTagModel.delete().where(
            (JobTagModel.job_id == job_id)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ).execute()

        # Delete job
        JobModel.delete().where(
            (JobModel.job_id == job_id)
            & (JobModel.experiment_id == experiment_id)
            & (JobModel.run_id == run_id)
        ).execute()

        logger.debug(
            "Deleted job %s (experiment=%s, run=%s)", job_id, experiment_id, run_id
        )

    # Service operations

    def update_service(
        self,
        service_id: str,
        experiment_id: str,
        run_id: str,
        description: str,
        state: str,
    ):
        """Update service information

        Args:
            service_id: Service identifier
            experiment_id: Experiment identifier
            run_id: Run identifier
            description: Human-readable description
            state: Service state

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update services in read-only mode")

        ServiceModel.insert(
            service_id=service_id,
            experiment_id=experiment_id,
            run_id=run_id,
            description=description,
            state=state,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ).on_conflict(
            conflict_target=[
                ServiceModel.service_id,
                ServiceModel.experiment_id,
                ServiceModel.run_id,
            ],
            update={
                ServiceModel.description: description,
                ServiceModel.state: state,
                ServiceModel.updated_at: datetime.now(),
            },
        ).execute()

        logger.debug(
            "Updated service %s (experiment=%s, run=%s)",
            service_id,
            experiment_id,
            run_id,
        )

    def get_services(
        self, experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[Dict]:
        """Get services, optionally filtered by experiment/run

        Args:
            experiment_id: Filter by experiment (None = all)
            run_id: Filter by run (None = current run if experiment_id provided)

        Returns:
            List of service dictionaries
        """
        query = ServiceModel.select()

        if experiment_id is not None:
            # Use current run if not specified
            if run_id is None:
                run_id = self.get_current_run(experiment_id)
                if run_id is None:
                    return []

            query = query.where(
                (ServiceModel.experiment_id == experiment_id)
                & (ServiceModel.run_id == run_id)
            )

        services = []
        for service_model in query:
            services.append(
                {
                    "service_id": service_model.service_id,
                    "experiment_id": service_model.experiment_id,
                    "run_id": service_model.run_id,
                    "description": service_model.description,
                    "state": service_model.state,
                    "created_at": service_model.created_at.isoformat(),
                    "updated_at": service_model.updated_at.isoformat(),
                }
            )
        return services

    # Utility methods

    def close(self):
        """Close the database connection and remove from registry

        This should be called when done with the workspace to free resources.
        """
        # Close database connection
        if hasattr(self, "workspace_db") and self.workspace_db is not None:
            from .state_db import close_workspace_database

            close_workspace_database(self.workspace_db)
            self.workspace_db = None

        # Remove from registry
        with WorkspaceStateProvider._lock:
            if self.workspace_path in WorkspaceStateProvider._instances:
                del WorkspaceStateProvider._instances[self.workspace_path]

        logger.debug("WorkspaceStateProvider closed for %s", self.workspace_path)

    # Helper methods

    def _get_job_tags(
        self, job_id: str, experiment_id: str, run_id: str
    ) -> Dict[str, str]:
        """Get tags for a job

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier

        Returns:
            Dictionary of tag key-value pairs
        """
        tags = {}
        for tag_model in JobTagModel.select().where(
            (JobTagModel.job_id == job_id)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ):
            tags[tag_model.tag_key] = tag_model.tag_value
        return tags

    def _job_model_to_dict(self, job_model: JobModel, tags: Dict[str, str]) -> Dict:
        """Convert a JobModel to a dictionary in UI format

        Args:
            job_model: JobModel instance
            tags: Dictionary of tags for this job

        Returns:
            Job dictionary compatible with UI expectations (camelCase keys)
        """
        # Parse progress JSON
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
            "tags": list(tags.items()),  # UI expects list of [key, value] tuples
            "experimentIds": [
                job_model.experiment_id
            ],  # Job belongs to this experiment
            "runId": job_model.run_id,  # Include run information
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
        return datetime.fromtimestamp(timestamp).isoformat()


# Scheduler listener adapter
class SchedulerListener:
    """Adapter to connect scheduler events to WorkspaceStateProvider

    This class implements the scheduler listener interface and forwards
    events to the WorkspaceStateProvider. It tracks which experiment/run
    each job belongs to for proper database updates.
    """

    def __init__(self, state_provider: WorkspaceStateProvider):
        """Initialize listener

        Args:
            state_provider: WorkspaceStateProvider instance to update
        """
        self.state_provider = state_provider
        # Map job_id -> (experiment_id, run_id) for tracking
        self.job_experiments: Dict[str, tuple] = {}

        logger.info("SchedulerListener initialized")

    def job_submitted(self, job: "Job", experiment_id: str, run_id: str):
        """Called when a job is submitted

        Args:
            job: The submitted job
            experiment_id: Experiment this job belongs to
            run_id: Run this job belongs to
        """
        # Track job's experiment/run
        self.job_experiments[job.identifier] = (experiment_id, run_id)

        # Update state provider
        try:
            self.state_provider.update_job_submitted(job, experiment_id, run_id)
        except Exception as e:
            logger.exception(
                "Error updating job submission for %s: %s", job.identifier, e
            )

    def job_state(self, job: "Job"):
        """Called when a job's state changes

        Args:
            job: The job with updated state
        """
        # Look up job's experiment/run
        if job.identifier not in self.job_experiments:
            logger.warning(
                "State change for unknown job %s (not tracked by listener)",
                job.identifier,
            )
            return

        experiment_id, run_id = self.job_experiments[job.identifier]

        # Update state provider
        try:
            self.state_provider.update_job_state(job, experiment_id, run_id)
        except Exception as e:
            logger.exception("Error updating job state for %s: %s", job.identifier, e)

    def service_add(self, service: "Service", experiment_id: str, run_id: str):
        """Called when a service is added

        Args:
            service: The added service
            experiment_id: Experiment identifier
            run_id: Run identifier
        """
        try:
            self.state_provider.update_service(
                service.id,
                experiment_id,
                run_id,
                service.description(),
                service.state.name,
            )
        except Exception as e:
            logger.exception("Error updating service %s: %s", service.id, e)
