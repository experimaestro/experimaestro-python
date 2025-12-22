"""Tests for workspace-level database models"""

import pytest
from pathlib import Path
from experimaestro.scheduler.state_db import (
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    JobTagModel,
    ServiceModel,
    WorkspaceSyncMetadata,
    initialize_workspace_database,
    close_workspace_database,
    ALL_MODELS,
)
from experimaestro.scheduler.state_sync import sync_workspace_from_disk
from experimaestro import Task, Param
from experimaestro.tests.utils import TemporaryExperiment


def test_database_initialization(tmp_path: Path):
    """Test that workspace database is initialized correctly"""
    db_path = tmp_path / "workspace.db"

    # Initialize database
    db = initialize_workspace_database(db_path, read_only=False)

    # Verify all tables were created
    assert ExperimentModel.table_exists()
    assert ExperimentRunModel.table_exists()
    assert JobModel.table_exists()
    assert JobTagModel.table_exists()
    assert ServiceModel.table_exists()
    assert WorkspaceSyncMetadata.table_exists()

    # Verify WorkspaceSyncMetadata was initialized
    metadata = WorkspaceSyncMetadata.get_or_none(
        WorkspaceSyncMetadata.id == "workspace"
    )
    assert metadata is not None
    assert metadata.sync_interval_minutes == 5

    # Cleanup
    close_workspace_database(db)


def test_experiment_and_run_models(tmp_path: Path):
    """Test creating experiments and runs"""
    db_path = tmp_path / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create an experiment
        ExperimentModel.create(experiment_id="test_exp")

        # Create a run for this experiment
        ExperimentRunModel.create(
            experiment_id="test_exp", run_id="run_001", status="active"
        )

        # Update experiment to point to this run
        ExperimentModel.update(current_run_id="run_001").where(
            ExperimentModel.experiment_id == "test_exp"
        ).execute()

        # Verify
        retrieved_exp = ExperimentModel.get(ExperimentModel.experiment_id == "test_exp")
        assert retrieved_exp.current_run_id == "run_001"

        retrieved_run = ExperimentRunModel.get(
            (ExperimentRunModel.experiment_id == "test_exp")
            & (ExperimentRunModel.run_id == "run_001")
        )
        assert retrieved_run.status == "active"

    close_workspace_database(db)


def test_job_model_with_composite_key(tmp_path: Path):
    """Test job model with composite primary key (job_id, experiment_id, run_id)"""
    db_path = tmp_path / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and run first
        ExperimentModel.create(experiment_id="exp1", current_run_id="run1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")

        # Create a job
        JobModel.create(
            job_id="job_abc",
            experiment_id="exp1",
            run_id="run1",
            task_id="MyTask",
            locator="",
            state="running",
            submitted_time=1234567890.0,
        )

        # Same job in different run
        ExperimentRunModel.create(experiment_id="exp1", run_id="run2")
        JobModel.create(
            job_id="job_abc",
            experiment_id="exp1",
            run_id="run2",
            task_id="MyTask",
            locator="",
            state="done",
            submitted_time=1234567891.0,
        )

        # Both should exist independently
        jobs = list(JobModel.select().where(JobModel.job_id == "job_abc"))
        assert len(jobs) == 2

        # Can update one without affecting the other
        JobModel.update(state="done").where(
            (JobModel.job_id == "job_abc")
            & (JobModel.experiment_id == "exp1")
            & (JobModel.run_id == "run1")
        ).execute()

        job_run1 = JobModel.get(
            (JobModel.job_id == "job_abc")
            & (JobModel.experiment_id == "exp1")
            & (JobModel.run_id == "run1")
        )
        assert job_run1.state == "done"

    close_workspace_database(db)


def test_job_tags_model(tmp_path: Path):
    """Test run-scoped job tags (fixes GH #128)"""
    db_path = tmp_path / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and runs
        ExperimentModel.create(experiment_id="exp1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run2")

        # Create job in both runs
        JobModel.create(
            job_id="job1",
            experiment_id="exp1",
            run_id="run1",
            task_id="Task",
            locator="",
            state="done",
        )
        JobModel.create(
            job_id="job1",
            experiment_id="exp1",
            run_id="run2",
            task_id="Task",
            locator="",
            state="done",
        )

        # Add different tags to same job in different runs
        JobTagModel.create(
            job_id="job1",
            experiment_id="exp1",
            run_id="run1",
            tag_key="env",
            tag_value="production",
        )

        JobTagModel.create(
            job_id="job1",
            experiment_id="exp1",
            run_id="run2",
            tag_key="env",
            tag_value="testing",
        )

        # Verify tags are independent per run
        run1_tags = list(
            JobTagModel.select().where(
                (JobTagModel.job_id == "job1")
                & (JobTagModel.experiment_id == "exp1")
                & (JobTagModel.run_id == "run1")
            )
        )
        assert len(run1_tags) == 1
        assert run1_tags[0].tag_value == "production"

        run2_tags = list(
            JobTagModel.select().where(
                (JobTagModel.job_id == "job1")
                & (JobTagModel.experiment_id == "exp1")
                & (JobTagModel.run_id == "run2")
            )
        )
        assert len(run2_tags) == 1
        assert run2_tags[0].tag_value == "testing"

    close_workspace_database(db)


def test_multiple_experiments_same_workspace(tmp_path: Path):
    """Test that multiple experiments can coexist in same workspace database"""
    db_path = tmp_path / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create two experiments
        ExperimentModel.create(experiment_id="exp1")
        ExperimentModel.create(experiment_id="exp2")

        # Create runs for each
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")
        ExperimentRunModel.create(experiment_id="exp2", run_id="run1")

        # Create jobs for each experiment
        JobModel.create(
            job_id="job1",
            experiment_id="exp1",
            run_id="run1",
            task_id="Task",
            locator="",
            state="done",
        )
        JobModel.create(
            job_id="job2",
            experiment_id="exp2",
            run_id="run1",
            task_id="Task",
            locator="",
            state="running",
        )

        # Query jobs for specific experiment
        exp1_jobs = list(JobModel.select().where(JobModel.experiment_id == "exp1"))
        assert len(exp1_jobs) == 1
        assert exp1_jobs[0].job_id == "job1"

        exp2_jobs = list(JobModel.select().where(JobModel.experiment_id == "exp2"))
        assert len(exp2_jobs) == 1
        assert exp2_jobs[0].job_id == "job2"

    close_workspace_database(db)


def test_read_only_mode(tmp_path: Path):
    """Test that read-only mode prevents writes"""
    db_path = tmp_path / "workspace.db"

    # Create database with write mode
    db_write = initialize_workspace_database(db_path, read_only=False)
    with db_write.bind_ctx(ALL_MODELS):
        ExperimentModel.create(experiment_id="exp1")
    close_workspace_database(db_write)

    # Open in read-only mode
    db_read = initialize_workspace_database(db_path, read_only=True)

    with db_read.bind_ctx(ALL_MODELS):
        # Can read
        exp = ExperimentModel.get(ExperimentModel.experiment_id == "exp1")
        assert exp.experiment_id == "exp1"

        # Cannot write (SQLite will raise OperationalError)
        with pytest.raises(Exception):  # Could be OperationalError or similar
            ExperimentModel.create(experiment_id="exp2", workdir_path="/tmp/exp2")

    close_workspace_database(db_read)


def test_upsert_on_conflict(tmp_path: Path):
    """Test that on_conflict works for updating existing records"""
    db_path = tmp_path / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and run
        ExperimentModel.create(experiment_id="exp1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")

        # Create job
        JobModel.insert(
            job_id="job1",
            experiment_id="exp1",
            run_id="run1",
            task_id="Task",
            locator="",
            state="running",
        ).execute()

        # Upsert with different state (disk wins)
        JobModel.insert(
            job_id="job1",
            experiment_id="exp1",
            run_id="run1",
            task_id="Task",
            locator="",
            state="done",
        ).on_conflict(
            conflict_target=[JobModel.job_id, JobModel.experiment_id, JobModel.run_id],
            update={JobModel.state: "done"},
        ).execute()

        # Verify state was updated
        job = JobModel.get(
            (JobModel.job_id == "job1")
            & (JobModel.experiment_id == "exp1")
            & (JobModel.run_id == "run1")
        )
        assert job.state == "done"

        # Only one job should exist
        assert JobModel.select().where(JobModel.job_id == "job1").count() == 1

    close_workspace_database(db)


# Define test task
class SimpleTask(Task):
    value: Param[int]

    def execute(self):
        # Write a marker file to indicate completion
        (self.__taskdir__ / "output.txt").write_text(f"value={self.value}")


def test_database_recovery_from_disk(tmp_path: Path):
    """Test recovering database from jobs.jsonl and disk state"""

    # Step 1: Run experiment with tasks and tags
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    with TemporaryExperiment("recovery", maxwait=0, workdir=workdir):
        # Submit first task with tags
        task1 = SimpleTask.C(value=42).tag("priority", "high").tag("env", "test")
        task1.submit()

        # Submit second task with different tags
        task2 = SimpleTask.C(value=100).tag("priority", "low").tag("env", "prod")
        task2.submit()

    # Step 2: Verify jobs.jsonl was created
    jobs_jsonl_path = workdir / "xp" / "recovery" / "jobs.jsonl"
    assert jobs_jsonl_path.exists(), "jobs.jsonl should have been created"

    # Verify database exists
    workspace_db_path = workdir / ".experimaestro" / "workspace.db"
    assert workspace_db_path.exists()

    # Get workspace state provider and access database
    from experimaestro.scheduler.state_provider import WorkspaceStateProvider

    provider = WorkspaceStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        # Get original state
        original_jobs = list(JobModel.select())

        # If no jobs in DB yet, sync from disk first
        if len(original_jobs) == 0:
            sync_workspace_from_disk(workdir, write_mode=True, force=True)
            original_jobs = list(JobModel.select())

        assert len(original_jobs) == 2

        original_job_ids = {job.job_id for job in original_jobs}
        assert len(original_job_ids) == 2

        # Get original tags
        original_tags = {}
        for job in original_jobs:
            job_tags = list(
                JobTagModel.select().where(
                    (JobTagModel.job_id == job.job_id)
                    & (JobTagModel.experiment_id == job.experiment_id)
                    & (JobTagModel.run_id == job.run_id)
                )
            )
            original_tags[job.job_id] = {tag.tag_key: tag.tag_value for tag in job_tags}

        assert len(original_tags) == 2
        # Verify we have tags for both jobs
        for job_id, tags in original_tags.items():
            assert "priority" in tags
            assert "env" in tags

    # Close provider to cleanup
    provider.close()

    # Step 3: Delete the database
    workspace_db_path.unlink()
    assert not workspace_db_path.exists()

    # Step 4: Recover from disk by syncing - get new provider instance
    provider2 = WorkspaceStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    sync_workspace_from_disk(
        workdir, write_mode=True, force=True, sync_interval_minutes=0
    )

    # Step 5: Verify recovered state matches original
    with provider2.workspace_db.bind_ctx(ALL_MODELS):
        # Check jobs were recovered
        recovered_jobs = list(JobModel.select())
        assert len(recovered_jobs) == 2

        recovered_job_ids = {job.job_id for job in recovered_jobs}
        assert recovered_job_ids == original_job_ids

        # Check tags were recovered
        recovered_tags = {}
        for job in recovered_jobs:
            job_tags = list(
                JobTagModel.select().where(
                    (JobTagModel.job_id == job.job_id)
                    & (JobTagModel.experiment_id == job.experiment_id)
                    & (JobTagModel.run_id == job.run_id)
                )
            )
            recovered_tags[job.job_id] = {
                tag.tag_key: tag.tag_value for tag in job_tags
            }

        assert len(recovered_tags) == 2

        # Verify tags match
        for job_id in original_job_ids:
            assert job_id in recovered_tags
            assert recovered_tags[job_id] == original_tags[job_id]
