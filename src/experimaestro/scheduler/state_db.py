"""Database models for experiment state persistence

This module provides peewee ORM models for storing job and service state
in SQLite databases. Each experiment gets its own database file (experiment.db)
with WAL mode enabled for concurrent read access.

Key design:
- One database per experiment at: workdir/xp/{experiment_name}/experiment.db
- No ExperimentModel needed - the database file itself represents the experiment
- Jobs can belong to multiple experiments and will appear in each experiment's database
- Current state and progress stored in JobModel - no history tracking
"""

from pathlib import Path
from peewee import (
    Model,
    Proxy,
    SqliteDatabase,
    CharField,
    FloatField,
    TextField,
    DateTimeField,
)
from datetime import datetime


# Database proxy - will be bound to specific file per experiment
database_proxy = Proxy()


class BaseModel(Model):
    """Base model with common configuration"""

    class Meta:
        database = database_proxy


class JobModel(BaseModel):
    """Job information

    Jobs can belong to multiple experiments. Each experiment database contains
    rows only for jobs associated with that experiment. The job_id is globally
    unique across all experiments.

    Fields:
        job_id: Unique identifier for the job (from task identifier)
        task_id: Task class identifier
        locator: Full task locator (identifier)
        path: Path to job directory
        state: Current job state (e.g., "unscheduled", "waiting", "running", "done", "error")
        failure_reason: Optional failure reason for error states (e.g., "TIMEOUT", "DEPENDENCY")
        submitted_time: When job was submitted (Unix timestamp)
        started_time: When job started running (Unix timestamp)
        ended_time: When job finished (Unix timestamp)
        progress: JSON-encoded list of progress updates
        tags: JSON-encoded tag dictionary
        dependencies: JSON-encoded list of dependency job IDs
    """

    job_id = CharField(primary_key=True)
    task_id = CharField(index=True)
    locator = CharField()
    path = CharField()
    state = CharField(default="unscheduled", index=True)
    failure_reason = CharField(null=True)
    submitted_time = FloatField(null=True)
    started_time = FloatField(null=True)
    ended_time = FloatField(null=True)
    progress = TextField(default="[]")  # JSON list of progress updates
    tags = TextField(default="{}")  # JSON
    dependencies = TextField(default="[]")  # JSON list of job IDs

    class Meta:
        table_name = "jobs"


class ServiceModel(BaseModel):
    """Service information

    Services are associated with experiments. Each experiment database contains
    only the services for that experiment.

    Fields:
        service_id: Unique identifier for the service
        description: Human-readable description
        state: Service state (e.g., "running", "stopped")
        created_at: When service was created
        updated_at: Timestamp of last update
    """

    service_id = CharField(primary_key=True)
    description = TextField(default="")
    state = CharField()
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "services"


def initialize_database(db_path: Path, read_only: bool = False) -> SqliteDatabase:
    """Initialize a database connection with proper configuration

    Args:
        db_path: Path to the SQLite database file
        read_only: If True, open database in read-only mode

    Returns:
        Configured SqliteDatabase instance
    """
    # Ensure parent directory exists (unless read-only)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create database connection
    db = SqliteDatabase(
        str(db_path),
        pragmas={
            "journal_mode": "wal",  # Write-Ahead Logging for concurrent reads
            "foreign_keys": 1,  # Enable foreign key constraints
            "ignore_check_constraints": 0,
            "synchronous": 1,  # NORMAL mode (balance safety/speed)
        },
    )

    if read_only:
        # Set query-only mode for read-only access
        db.execute_sql("PRAGMA query_only = ON")

    # Bind database to models
    database_proxy.initialize(db)

    # Create tables if they don't exist (only in write mode)
    if not read_only:
        db.create_tables([JobModel, ServiceModel], safe=True)

    return db


def close_database():
    """Close the current database connection"""
    if database_proxy.obj is not None:
        database_proxy.close()
        database_proxy.initialize(None)
