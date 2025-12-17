"""Tests for database models and ExperimentStateProvider"""

import json
import pytest
from pathlib import Path
from experimaestro.scheduler.state_db import (
    JobModel,
    ServiceModel,
    initialize_database,
    close_database,
)
from experimaestro.scheduler.state_provider import ExperimentStateProvider
from experimaestro import Task, Param


class SimpleTask(Task):
    """Simple task for testing"""

    x: Param[int]

    def execute(self):
        pass


def test_database_initialization(tmp_path: Path):
    """Test that database is initialized correctly"""
    db_path = tmp_path / "test.db"

    # Initialize database
    _ = initialize_database(db_path, read_only=False)

    # Verify tables were created
    assert JobModel.table_exists()
    assert ServiceModel.table_exists()

    # Cleanup
    close_database()


def test_job_model_crud(tmp_path: Path):
    """Test creating, reading, updating, and deleting jobs"""
    db_path = tmp_path / "test.db"
    _ = initialize_database(db_path, read_only=False)

    # Create a job
    job = JobModel.create(
        job_id="test_job_1",
        task_id="SimpleTask",
        locator="test_job_1",
        path="/tmp/test/job1",
        state="running",
        tags=json.dumps({"env": "test"}),
        dependencies=json.dumps(["dep1", "dep2"]),
    )

    # Read the job
    retrieved = JobModel.get(JobModel.job_id == "test_job_1")
    assert retrieved.job_id == "test_job_1"
    assert retrieved.state == "running"
    assert json.loads(retrieved.tags) == {"env": "test"}
    assert json.loads(retrieved.dependencies) == ["dep1", "dep2"]

    # Update the job
    JobModel.update(state="done").where(JobModel.job_id == "test_job_1").execute()
    retrieved = JobModel.get(JobModel.job_id == "test_job_1")
    assert retrieved.state == "done"

    # Delete the job
    job.delete_instance()
    assert JobModel.select().where(JobModel.job_id == "test_job_1").count() == 0

    # Cleanup
    close_database()


def test_experiment_state_provider_basic(tmp_path: Path):
    """Test basic ExperimentStateProvider functionality"""
    workdir = tmp_path / "workspace"
    experiment_dir = workdir / "xp" / "test_exp"
    experiment_dir.mkdir(parents=True)

    # Create provider
    provider = ExperimentStateProvider(workdir, "test_exp", read_only=False)

    # Initially no jobs
    info = provider.get_experiment_info()
    assert info["experiment_id"] == "test_exp"
    assert info["total_jobs"] == 0
    assert info["finished_jobs"] == 0

    # Manually insert a job
    JobModel.create(
        job_id="job1",
        task_id="SimpleTask",
        locator="job1",
        path="/tmp/job1",
        state="done",
    )

    # Check job count
    info = provider.get_experiment_info()
    assert info["total_jobs"] == 1
    assert info["finished_jobs"] == 1

    # Get jobs
    jobs = provider.get_jobs()
    assert len(jobs) == 1
    assert jobs[0]["jobId"] == "job1"  # Changed to camelCase

    # Get specific job
    job = provider.get_job("job1")
    assert job is not None
    assert job["jobId"] == "job1"  # Changed to camelCase
    assert job["status"] == "done"  # Changed from "state" to "status"

    # Cleanup
    provider.close()


def test_state_json_migration(tmp_path: Path):
    """Test migration from legacy state.json to database"""
    workdir = tmp_path / "workspace"
    experiment_dir = workdir / "xp" / "test_exp"
    experiment_dir.mkdir(parents=True)

    # Create a mock state.json
    state_json_path = experiment_dir / "state.json"

    # Create a simple task instance for the state
    task = SimpleTask.C(x=42)

    # Write state.json using experimaestro's serialization format
    from experimaestro.core.serialization import save_definition
    from experimaestro.core.context import SerializationContext

    state_data = [
        {
            "id": "job_1",
            "path": str(tmp_path / "job_1"),
            "task": task,
            "tags": {"test": "value"},
            "depends_on": [],
        }
    ]

    save_definition(state_data, SerializationContext(), state_json_path)

    # Create provider in read-only mode (should trigger migration)
    provider = ExperimentStateProvider(workdir, "test_exp", read_only=True)

    # Verify database was created
    db_path = experiment_dir / "experiment.db"
    assert db_path.exists()

    # Verify job was migrated
    jobs = provider.get_jobs()
    assert len(jobs) == 1
    assert jobs[0]["jobId"] == "job_1"  # Changed to camelCase
    assert (
        jobs[0]["status"] == "done"
    )  # Migrated jobs assumed complete (changed from "state")

    # Cleanup
    provider.close()


def test_read_only_mode(tmp_path: Path):
    """Test that read-only mode prevents writes"""
    workdir = tmp_path / "workspace"
    experiment_dir = workdir / "xp" / "test_exp"
    experiment_dir.mkdir(parents=True)

    # Create a database first
    db_path = experiment_dir / "experiment.db"
    _ = initialize_database(db_path, read_only=False)
    close_database()

    # Open in read-only mode
    provider = ExperimentStateProvider(workdir, "test_exp", read_only=True)

    # Create a mock job object
    class MockState:
        name = "running"

    class MockType:
        identifier = "MockTask"

    class MockJob:
        identifier = "test_job"
        type = MockType()
        state = MockState()
        submittime = 1234567890.0
        tags = {}
        dependencies = []

    # Attempt to write should raise error
    with pytest.raises(RuntimeError, match="read-only mode"):
        provider.update_job_submitted(MockJob())

    # Cleanup
    provider.close()


def test_concurrent_read_access(tmp_path: Path):
    """Test that WAL mode allows concurrent reads"""
    workdir = tmp_path / "workspace"
    experiment_dir = workdir / "xp" / "test_exp"
    experiment_dir.mkdir(parents=True)

    # Create and populate a database
    provider1 = ExperimentStateProvider(workdir, "test_exp", read_only=False)
    JobModel.create(
        job_id="job1",
        task_id="SimpleTask",
        locator="job1",
        path="/tmp/job1",
        state="done",
    )

    # Open second provider in read-only mode while first is still open
    provider2 = ExperimentStateProvider(workdir, "test_exp", read_only=True)

    # Both should be able to read
    jobs1 = provider1.get_jobs()
    jobs2 = provider2.get_jobs()

    assert len(jobs1) == 1
    assert len(jobs2) == 1

    # Cleanup
    provider1.close()
    provider2.close()
