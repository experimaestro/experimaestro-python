"""Database models for experiment state persistence

This module provides peewee ORM models for storing job and service state
in a workspace-level SQLite database. The workspace has a single database
file (.experimaestro/workspace.db) with WAL mode enabled for concurrent
read/write access.

Key design:
- One database per workspace at: workdir/.experimaestro/workspace.db
- Experiments can be run multiple times, each run tracked separately
- Jobs and services are scoped to (experiment_id, run_id)
- Tags are scoped to (job_id, experiment_id, run_id) - fixes GH #128
- Current state and progress stored in JobModel - no history tracking
- Database instance is passed explicitly to avoid global state
"""

from pathlib import Path
from peewee import (
    Model,
    SqliteDatabase,
    CharField,
    FloatField,
    IntegerField,
    TextField,
    DateTimeField,
    CompositeKey,
    IntegrityError,
    OperationalError,
)
from datetime import datetime
import fasteners


class BaseModel(Model):
    """Base model for workspace database tables

    Models are unbound by default. Use database.bind_ctx() when querying:

        with workspace.workspace_db.bind_ctx([ExperimentModel, JobModel, ...]):
            experiments = ExperimentModel.select()

    Or use the convenience method bind_models() defined below.
    """

    class Meta:
        database = None  # Unbound - will be bound when used


class ExperimentModel(BaseModel):
    """Experiment metadata - tracks experiment definitions

    An experiment can be run multiple times. This table tracks the experiment
    itself and points to the current/latest run.

    Fields:
        experiment_id: Unique identifier for the experiment
        current_run_id: Points to the current/latest run (null if no runs yet)
        created_at: When experiment was first created
        updated_at: When experiment was last modified (for incremental queries)

    Note: Experiment path is derivable: {workspace}/xp/{experiment_id}
    """

    experiment_id = CharField(primary_key=True)
    current_run_id = CharField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now, index=True)

    class Meta:
        table_name = "experiments"


class ExperimentRunModel(BaseModel):
    """Individual experiment runs

    Each time an experiment is executed, a new run is created.
    Runs are identified by (experiment_id, run_id) composite key.

    run_id format: timestamp-based like "20250120_143022" or sequential counter

    Fields:
        experiment_id: ID of the experiment this run belongs to
        run_id: Unique ID for this run (timestamp or sequential)
        started_at: When this run started
        ended_at: When this run completed (null if still active)
        status: Run status (active, completed, failed, abandoned)
    """

    experiment_id = CharField(index=True)
    run_id = CharField(index=True)
    started_at = DateTimeField(default=datetime.now)
    ended_at = DateTimeField(null=True)
    status = CharField(default="active", index=True)

    class Meta:
        table_name = "experiment_runs"
        primary_key = CompositeKey("experiment_id", "run_id")
        indexes = ((("experiment_id", "started_at"), False),)  # For finding latest run


class WorkspaceSyncMetadata(BaseModel):
    """Workspace-level metadata for disk sync tracking

    Single-row table to track when the last disk sync occurred.
    Used to throttle sync operations and prevent excessive disk scanning.

    Fields:
        id: Always "workspace" (single row table)
        last_sync_time: When last sync completed
        sync_interval_minutes: Minimum interval between syncs
    """

    id = CharField(primary_key=True, default="workspace")
    last_sync_time = DateTimeField(null=True)
    sync_interval_minutes = IntegerField(default=5)

    class Meta:
        table_name = "workspace_sync_metadata"


class JobModel(BaseModel):
    """Job information linked to specific experiment run

    Jobs are tied to a specific run of an experiment via (experiment_id, run_id).
    The same job can appear in multiple runs with different states/tags.

    Fields:
        job_id: Unique identifier for the job (from task identifier)
        experiment_id: ID of the experiment this job belongs to
        run_id: ID of the run this job belongs to
        task_id: Task class identifier
        locator: Full task locator (identifier)
        state: Current job state (e.g., "unscheduled", "waiting", "running", "done", "error")
        failure_reason: Optional failure reason for error states (e.g., "TIMEOUT", "DEPENDENCY")
        submitted_time: When job was submitted (Unix timestamp)
        started_time: When job started running (Unix timestamp)
        ended_time: When job finished (Unix timestamp)
        progress: JSON-encoded list of progress updates
        updated_at: When job was last modified (for incremental queries)

    Note: Job path is derivable: {workspace}/jobs/{task_id}/{job_id}
    Note: Tags are stored in separate JobTagModel table (run-scoped)
    Note: Dependencies are NOT stored in DB (available in state.json only)
    """

    job_id = CharField(index=True)
    experiment_id = CharField(index=True)
    run_id = CharField(index=True)
    task_id = CharField(index=True)
    locator = CharField()
    state = CharField(default="unscheduled", index=True)
    failure_reason = CharField(null=True)
    submitted_time = FloatField(null=True)
    started_time = FloatField(null=True)
    ended_time = FloatField(null=True)
    progress = TextField(default="[]")
    updated_at = DateTimeField(default=datetime.now, index=True)

    class Meta:
        table_name = "jobs"
        primary_key = CompositeKey("job_id", "experiment_id", "run_id")
        indexes = (
            (
                ("experiment_id", "run_id", "state"),
                False,
            ),  # Query jobs by run and state
            (
                ("experiment_id", "run_id", "task_id"),
                False,
            ),  # Query jobs by run and task
            (
                ("experiment_id", "run_id", "updated_at"),
                False,
            ),  # Query jobs by run and update time
        )


class JobTagModel(BaseModel):
    """Job tags for efficient searching (fixes GH #128)

    **FIX FOR GH ISSUE #128**: Tags are now experiment-run-dependent, not job-dependent.
    The same job in different experiment runs can have different tags, because tags
    are scoped to the (job_id, experiment_id, run_id) combination.

    Tags are stored as key-value pairs in a separate table for efficient indexing.
    Each job can have multiple tags within an experiment run context.

    Key change from old behavior:
    - OLD: Tags were global per job_id (broken - same job in different experiments/runs shared tags)
    - NEW: Tags are scoped per (job_id, experiment_id, run_id) - same job can have different tags in different runs

    Fields:
        job_id: ID of the job
        experiment_id: ID of the experiment
        run_id: ID of the run
        tag_key: Tag name
        tag_value: Tag value
    """

    job_id = CharField(index=True)
    experiment_id = CharField(index=True)
    run_id = CharField(index=True)
    tag_key = CharField(index=True)
    tag_value = CharField(index=True)

    class Meta:
        table_name = "job_tags"
        primary_key = CompositeKey("job_id", "experiment_id", "run_id", "tag_key")
        indexes = (
            (("tag_key", "tag_value"), False),  # For tag-based queries
            (
                ("experiment_id", "run_id", "tag_key"),
                False,
            ),  # For experiment run tag queries
        )


class ServiceModel(BaseModel):
    """Service information linked to specific experiment run

    Services are tied to a specific run of an experiment via (experiment_id, run_id).

    Fields:
        service_id: Unique identifier for the service
        experiment_id: ID of the experiment this service belongs to
        run_id: ID of the run this service belongs to
        description: Human-readable description
        state: Service state (e.g., "running", "stopped")
        created_at: When service was created
        updated_at: Timestamp of last update
    """

    service_id = CharField()
    experiment_id = CharField(index=True)
    run_id = CharField(index=True)
    description = TextField(default="")
    state = CharField()
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "services"
        primary_key = CompositeKey("service_id", "experiment_id", "run_id")


class PartialModel(BaseModel):
    """Partial directory tracking for subparameters

    Tracks partial directories that are shared across jobs with different
    parameter values (but same partial identifier). These directories are
    at WORKSPACE/partials/TASK_ID/SUBPARAM_NAME/PARTIAL_ID/ (reconstructible).

    Fields:
        partial_id: Hex hash of the partial identifier
        task_id: Task class identifier
        subparameters_name: Name of the subparameters definition
        created_at: When this partial directory was first created
    """

    partial_id = CharField(primary_key=True)
    task_id = CharField(index=True)
    subparameters_name = CharField(index=True)
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "partials"
        indexes = ((("task_id", "subparameters_name"), False),)


class JobPartialModel(BaseModel):
    """Links jobs to partial directories they use

    Tracks which jobs reference which partial directories. This enables
    cleanup of orphan partials when all referencing jobs are deleted.

    A job can use multiple partials (different subparameters definitions),
    and a partial can be used by multiple jobs.

    Fields:
        job_id: ID of the job using this partial
        experiment_id: ID of the experiment
        run_id: ID of the run
        partial_id: ID of the partial directory being used
    """

    job_id = CharField(index=True)
    experiment_id = CharField(index=True)
    run_id = CharField(index=True)
    partial_id = CharField(index=True)

    class Meta:
        table_name = "job_partials"
        primary_key = CompositeKey("job_id", "experiment_id", "run_id", "partial_id")
        indexes = ((("partial_id",), False),)  # For finding jobs using a partial


# List of all models for binding
ALL_MODELS = [
    ExperimentModel,
    ExperimentRunModel,
    WorkspaceSyncMetadata,
    JobModel,
    JobTagModel,
    ServiceModel,
    PartialModel,
    JobPartialModel,
]


def initialize_workspace_database(
    db_path: Path, read_only: bool = False
) -> SqliteDatabase:
    """Initialize a workspace database connection with proper configuration

    Creates and configures a SQLite database connection for the workspace.
    Models must be bound to this database before querying.

    Uses file-based locking to prevent multiple processes from initializing
    the database simultaneously, which could cause SQLite locking issues.

    Args:
        db_path: Path to the workspace SQLite database file
        read_only: If True, open database in read-only mode

    Returns:
        Configured SqliteDatabase instance
    """
    # Ensure parent directory exists (unless read-only)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # Use file-based lock to prevent concurrent initialization from multiple processes
    # This prevents SQLite locking issues during table creation
    lock_path = db_path.parent / f".{db_path.name}.init.lock"
    lock = fasteners.InterProcessLock(str(lock_path))

    # Acquire lock (blocking) - only one process can initialize at a time
    with lock:
        # Create database connection
        # check_same_thread=False allows the connection to be used from multiple threads
        # This is safe with WAL mode and proper locking
        db = SqliteDatabase(
            str(db_path),
            pragmas={
                "journal_mode": "wal",  # Write-Ahead Logging for concurrent reads
                "foreign_keys": 1,  # Enable foreign key constraints
                "ignore_check_constraints": 0,
                "synchronous": 1,  # NORMAL mode (balance safety/speed)
                "busy_timeout": 5000,  # Wait up to 5 seconds for locks
            },
            check_same_thread=False,
        )

        if read_only:
            # Set query-only mode for read-only access
            db.execute_sql("PRAGMA query_only = ON")

        # Bind all models to this database
        db.bind(ALL_MODELS)

        # Create tables if they don't exist (only in write mode)
        if not read_only:
            db.create_tables(ALL_MODELS, safe=True)

            # Initialize WorkspaceSyncMetadata with default row if not exists
            # Use try/except to handle race condition (shouldn't happen with lock, but be safe)
            try:
                WorkspaceSyncMetadata.get_or_create(
                    id="workspace",
                    defaults={"last_sync_time": None, "sync_interval_minutes": 5},
                )
            except (IntegrityError, OperationalError):
                # If get_or_create fails, the row likely already exists
                pass

    return db


def close_workspace_database(db: SqliteDatabase):
    """Close a workspace database connection

    Args:
        db: The database connection to close
    """
    if db and not db.is_closed():
        db.close()
