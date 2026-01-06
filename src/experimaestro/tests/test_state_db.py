"""Tests for workspace-level database models"""

import pytest
from pathlib import Path
from experimaestro.scheduler.state_db import (
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    JobTagModel,
    JobExperimentsModel,
    ServiceModel,
    WorkspaceSyncMetadata,
    initialize_workspace_database,
    close_workspace_database,
    ALL_MODELS,
    DatabaseVersionError,
    CURRENT_DB_VERSION,
)
from experimaestro.scheduler.state_sync import sync_workspace_from_disk
from experimaestro import Task, Param
from experimaestro.tests.utils import TemporaryExperiment


def test_database_initialization(tmp_path: Path):
    """Test that workspace database is initialized correctly"""
    db_path = tmp_path / "workspace.db"

    # Initialize database
    db, _ = initialize_workspace_database(db_path, read_only=False)

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
    db, _ = initialize_workspace_database(db_path, read_only=False)

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
    """Test job model with composite primary key (job_id, task_id)"""
    db_path = tmp_path / "workspace.db"
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and run first
        ExperimentModel.create(experiment_id="exp1", current_run_id="run1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run2")

        # Create a job (job_id + task_id form primary key)
        JobModel.create(
            job_id="job_abc",
            task_id="MyTask",
            state="running",
            submitted_time=1234567890.0,
        )

        # Link job to both runs via JobExperimentsModel
        JobExperimentsModel.create(
            job_id="job_abc", experiment_id="exp1", run_id="run1"
        )
        JobExperimentsModel.create(
            job_id="job_abc", experiment_id="exp1", run_id="run2"
        )

        # Only one job exists (same job_id, task_id)
        jobs = list(JobModel.select().where(JobModel.job_id == "job_abc"))
        assert len(jobs) == 1

        # Can update job state
        JobModel.update(state="done").where(
            (JobModel.job_id == "job_abc") & (JobModel.task_id == "MyTask")
        ).execute()

        job = JobModel.get(
            (JobModel.job_id == "job_abc") & (JobModel.task_id == "MyTask")
        )
        assert job.state == "done"

        # Job is linked to both runs
        links = list(
            JobExperimentsModel.select().where(JobExperimentsModel.job_id == "job_abc")
        )
        assert len(links) == 2

    close_workspace_database(db)


def test_job_tags_model(tmp_path: Path):
    """Test run-scoped job tags (fixes GH #128)"""
    db_path = tmp_path / "workspace.db"
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and runs
        ExperimentModel.create(experiment_id="exp1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run2")

        # Create job (job_id + task_id are primary key)
        JobModel.create(
            job_id="job1",
            task_id="Task",
            state="done",
        )

        # Link job to both runs via JobExperimentsModel
        JobExperimentsModel.create(job_id="job1", experiment_id="exp1", run_id="run1")
        JobExperimentsModel.create(job_id="job1", experiment_id="exp1", run_id="run2")

        # Add different tags to same job in different runs
        # Tags are scoped by (job_id, experiment_id, run_id)
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


def test_job_experiments_model(tmp_path: Path):
    """Test job-experiment relationship model"""
    db_path = tmp_path / "workspace.db"
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiments and runs
        ExperimentModel.create(experiment_id="exp1")
        ExperimentModel.create(experiment_id="exp2")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run2")
        ExperimentRunModel.create(experiment_id="exp2", run_id="run1")

        # Create jobs (job_id + task_id are primary key)
        JobModel.create(
            job_id="job1",
            task_id="Task",
            state="done",
        )
        JobModel.create(
            job_id="job2",
            task_id="Task",
            state="done",
        )

        # Create job-experiment relationships
        JobExperimentsModel.create(job_id="job1", experiment_id="exp1", run_id="run1")
        JobExperimentsModel.create(job_id="job1", experiment_id="exp1", run_id="run2")
        JobExperimentsModel.create(job_id="job2", experiment_id="exp2", run_id="run1")

        # Query jobs by experiment
        exp1_jobs = list(
            JobExperimentsModel.select().where(
                JobExperimentsModel.experiment_id == "exp1"
            )
        )
        assert len(exp1_jobs) == 2

        # Query experiments for a job
        job1_experiments = list(
            JobExperimentsModel.select().where(JobExperimentsModel.job_id == "job1")
        )
        assert len(job1_experiments) == 2
        assert all(je.experiment_id == "exp1" for je in job1_experiments)

        # Query by run
        run1_jobs = list(
            JobExperimentsModel.select().where(
                (JobExperimentsModel.experiment_id == "exp1")
                & (JobExperimentsModel.run_id == "run1")
            )
        )
        assert len(run1_jobs) == 1
        assert run1_jobs[0].job_id == "job1"

    close_workspace_database(db)


def test_multiple_experiments_same_workspace(tmp_path: Path):
    """Test that multiple experiments can coexist in same workspace database"""
    db_path = tmp_path / "workspace.db"
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create two experiments
        ExperimentModel.create(experiment_id="exp1")
        ExperimentModel.create(experiment_id="exp2")

        # Create runs for each
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")
        ExperimentRunModel.create(experiment_id="exp2", run_id="run1")

        # Create jobs (job_id + task_id are primary key)
        JobModel.create(
            job_id="job1",
            task_id="Task",
            state="done",
        )
        JobModel.create(
            job_id="job2",
            task_id="Task",
            state="running",
        )

        # Link jobs to experiments via JobExperimentsModel
        JobExperimentsModel.create(job_id="job1", experiment_id="exp1", run_id="run1")
        JobExperimentsModel.create(job_id="job2", experiment_id="exp2", run_id="run1")

        # Query jobs for specific experiment using JobExperimentsModel
        exp1_job_ids = JobExperimentsModel.select(JobExperimentsModel.job_id).where(
            JobExperimentsModel.experiment_id == "exp1"
        )
        exp1_jobs = list(JobModel.select().where(JobModel.job_id.in_(exp1_job_ids)))
        assert len(exp1_jobs) == 1
        assert exp1_jobs[0].job_id == "job1"

        exp2_job_ids = JobExperimentsModel.select(JobExperimentsModel.job_id).where(
            JobExperimentsModel.experiment_id == "exp2"
        )
        exp2_jobs = list(JobModel.select().where(JobModel.job_id.in_(exp2_job_ids)))
        assert len(exp2_jobs) == 1
        assert exp2_jobs[0].job_id == "job2"

    close_workspace_database(db)


def test_read_only_mode(tmp_path: Path):
    """Test that read-only mode prevents writes"""
    db_path = tmp_path / "workspace.db"

    # Create database with write mode
    db_write, _ = initialize_workspace_database(db_path, read_only=False)
    with db_write.bind_ctx(ALL_MODELS):
        ExperimentModel.create(experiment_id="exp1")
    close_workspace_database(db_write)

    # Open in read-only mode
    db_read, _ = initialize_workspace_database(db_path, read_only=True)

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
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and run
        ExperimentModel.create(experiment_id="exp1")
        ExperimentRunModel.create(experiment_id="exp1", run_id="run1")

        # Create job (job_id + task_id are primary key)
        JobModel.insert(
            job_id="job1",
            task_id="Task",
            state="running",
        ).execute()

        # Link job to experiment
        JobExperimentsModel.create(job_id="job1", experiment_id="exp1", run_id="run1")

        # Upsert with different state (disk wins)
        # The conflict target is now (job_id, task_id)
        JobModel.insert(
            job_id="job1",
            task_id="Task",
            state="done",
        ).on_conflict(
            conflict_target=[JobModel.job_id, JobModel.task_id],
            update={JobModel.state: "done"},
        ).execute()

        # Verify state was updated
        job = JobModel.get((JobModel.job_id == "job1") & (JobModel.task_id == "Task"))
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
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    provider = DbStateProvider.get_instance(
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

        # Get original tags (use JobExperimentsModel to find experiment/run for each job)
        original_tags = {}
        for job in original_jobs:
            # Find the experiment/run link for this job
            job_exp = JobExperimentsModel.get_or_none(
                JobExperimentsModel.job_id == job.job_id
            )
            if job_exp:
                job_tags = list(
                    JobTagModel.select().where(
                        (JobTagModel.job_id == job.job_id)
                        & (JobTagModel.experiment_id == job_exp.experiment_id)
                        & (JobTagModel.run_id == job_exp.run_id)
                    )
                )
                original_tags[job.job_id] = {
                    tag.tag_key: tag.tag_value for tag in job_tags
                }

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
    provider2 = DbStateProvider.get_instance(
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

        # Check tags were recovered (use JobExperimentsModel to find experiment/run)
        recovered_tags = {}
        for job in recovered_jobs:
            # Find the experiment/run link for this job
            job_exp = JobExperimentsModel.get_or_none(
                JobExperimentsModel.job_id == job.job_id
            )
            if job_exp:
                job_tags = list(
                    JobTagModel.select().where(
                        (JobTagModel.job_id == job.job_id)
                        & (JobTagModel.experiment_id == job_exp.experiment_id)
                        & (JobTagModel.run_id == job_exp.run_id)
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


# =============================================================================
# Round-Trip Serialization Tests
# =============================================================================


def test_mockjob_serialization_roundtrip(tmp_path: Path):
    """Test MockJob serialization and deserialization round-trip"""
    from experimaestro.scheduler.state_provider import MockJob
    from experimaestro.scheduler.interfaces import BaseJob
    from experimaestro.notifications import LevelInformation

    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    # Create a MockJob with all fields populated
    # Note: tags, experiment_id, run_id are no longer part of MockJob
    original = MockJob(
        identifier="abc123",
        task_id="my.Task",
        path=workspace_path / "jobs" / "my.Task" / "abc123",
        state="running",
        submittime=1234567890.0,
        starttime=1234567891.0,
        endtime=None,
        progress=[LevelInformation(level=0, progress=0.5, desc="halfway")],
        updated_at="2024-01-01T00:00:00",
        exit_code=None,
        retry_count=2,
    )

    # Serialize to dict
    serialized = original.db_state_dict()

    # Deserialize from dict
    restored = MockJob.from_db_state_dict(serialized, workspace_path)

    # Verify equality using db_state_eq
    assert BaseJob.db_state_eq(original, restored)


def test_mockexperiment_serialization_roundtrip(tmp_path: Path):
    """Test MockExperiment serialization and deserialization round-trip"""
    from experimaestro.scheduler.state_provider import MockExperiment
    from experimaestro.scheduler.interfaces import BaseExperiment

    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "xp" / "test_exp").mkdir(parents=True)

    # Create a MockExperiment with all fields populated
    original = MockExperiment(
        workdir=workspace_path / "xp" / "test_exp",
        current_run_id="run_001",
        total_jobs=10,
        finished_jobs=5,
        failed_jobs=1,
        updated_at="2024-01-01T12:00:00",
        started_at=1234567890.0,
        ended_at=None,
        hostname="testhost",
    )

    # Serialize to dict
    serialized = original.db_state_dict()

    # Add additional fields for from_db_state_dict
    serialized["total_jobs"] = original.total_jobs
    serialized["finished_jobs"] = original.finished_jobs
    serialized["failed_jobs"] = original.failed_jobs
    serialized["updated_at"] = original.updated_at
    serialized["started_at"] = original.started_at
    serialized["ended_at"] = original.ended_at
    serialized["hostname"] = original.hostname

    # Deserialize from dict
    restored = MockExperiment.from_db_state_dict(serialized, workspace_path)

    # Verify equality using db_state_eq
    assert BaseExperiment.db_state_eq(original, restored)


def test_mockservice_serialization_roundtrip():
    """Test MockService serialization and deserialization round-trip"""
    from experimaestro.scheduler.state_provider import MockService
    from experimaestro.scheduler.interfaces import BaseService

    # Create a MockService with all fields populated
    original = MockService(
        service_id="svc_123",
        description_text="Test service description",
        state_dict_data={"port": 8080, "host": "localhost"},
        experiment_id="exp1",
        run_id="run1",
        url="http://localhost:8080",
        state="RUNNING",
    )

    # Serialize to dict
    serialized = original.db_state_dict()

    # Add additional fields for from_db_state_dict
    serialized["experiment_id"] = original.experiment_id
    serialized["run_id"] = original.run_id
    serialized["url"] = original.url

    # Deserialize from dict
    restored = MockService.from_db_state_dict(serialized)

    # Verify equality using db_state_eq
    assert BaseService.db_state_eq(original, restored)


def test_experimentrun_serialization_roundtrip():
    """Test ExperimentRun serialization and deserialization round-trip"""
    from experimaestro.scheduler.interfaces import ExperimentRun

    # Create an ExperimentRun with all fields populated
    original = ExperimentRun(
        run_id="run_001",
        experiment_id="test_exp",
        hostname="testhost",
        started_at=1234567890.0,
        ended_at=1234567900.0,
        status="completed",
        total_jobs=10,
        finished_jobs=9,
        failed_jobs=1,
    )

    # Serialize to dict
    serialized = original.db_state_dict()

    # Deserialize from dict
    restored = ExperimentRun.from_db_state_dict(serialized)

    # Verify equality using db_state_eq
    assert ExperimentRun.db_state_eq(original, restored)


# =============================================================================
# Scheduler-to-DB Sync Tests
# =============================================================================


class SimpleDbSyncTask(Task):
    """Task used for testing DB sync"""

    value: Param[int]

    def execute(self):
        (self.__taskdir__ / "output.txt").write_text(f"value={self.value}")


def test_scheduler_syncs_jobs_to_db(tmp_path: Path):
    """Test that jobs submitted via scheduler are synced to the database"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # Run experiment with a simple task
    with TemporaryExperiment("db_sync_test", maxwait=0, workdir=workdir):
        task = SimpleDbSyncTask.C(value=42).tag("test_tag", "test_value")
        task.submit()

    # Get the existing provider (created by experiment) - use same read_only mode
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        # Check experiment was created
        experiments = list(ExperimentModel.select())
        assert len(experiments) >= 1
        exp = next(e for e in experiments if e.experiment_id == "db_sync_test")
        assert exp is not None

        # Check run was created
        runs = list(
            ExperimentRunModel.select().where(
                ExperimentRunModel.experiment_id == "db_sync_test"
            )
        )
        assert len(runs) >= 1

        # Check job was created (find via JobExperimentsModel)
        job_ids_in_exp = JobExperimentsModel.select(JobExperimentsModel.job_id).where(
            JobExperimentsModel.experiment_id == "db_sync_test"
        )
        jobs = list(JobModel.select().where(JobModel.job_id.in_(job_ids_in_exp)))
        assert len(jobs) == 1
        job = jobs[0]

        # Verify job state was updated (should be done after experiment completes)
        assert job.state == "done"

        # Verify tags were synced
        tags = list(
            JobTagModel.select().where(
                (JobTagModel.job_id == job.job_id)
                & (JobTagModel.experiment_id == "db_sync_test")
            )
        )
        tag_dict = {t.tag_key: t.tag_value for t in tags}
        assert tag_dict.get("test_tag") == "test_value"

    # Clean up the singleton (close() removes from _instances dict)
    provider.close()


def test_scheduler_syncs_job_state_changes(tmp_path: Path):
    """Test that job state changes are synced to the database"""

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # Submit a task and wait for it to complete
    with TemporaryExperiment("state_change_test", maxwait=0, workdir=workdir):
        task1 = SimpleDbSyncTask.C(value=1)
        task1.submit()

        task2 = SimpleDbSyncTask.C(value=2)
        task2.submit()

    # Verify job states in database
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    # Use same read_only mode as the experiment created
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        # Find jobs via JobExperimentsModel
        job_ids_in_exp = JobExperimentsModel.select(JobExperimentsModel.job_id).where(
            JobExperimentsModel.experiment_id == "state_change_test"
        )
        jobs = list(JobModel.select().where(JobModel.job_id.in_(job_ids_in_exp)))
        assert len(jobs) == 2

        # Both jobs should be done
        for job in jobs:
            assert job.state == "done", (
                f"Job {job.job_id} has unexpected state: {job.state}"
            )

    # Clean up the singleton (close() removes from _instances dict)
    provider.close()


def test_database_version_error_on_newer_version(tmp_path: Path):
    """Test that DatabaseVersionError is raised when DB version is newer than code"""
    db_path = tmp_path / "workspace.db"

    # First, create a valid database
    db, _ = initialize_workspace_database(db_path, read_only=False)

    # Now update the version to be higher than CURRENT_DB_VERSION
    future_version = CURRENT_DB_VERSION + 1
    with db.bind_ctx(ALL_MODELS):
        WorkspaceSyncMetadata.update(db_version=future_version).where(
            WorkspaceSyncMetadata.id == "workspace"
        ).execute()

    close_workspace_database(db)

    # Trying to open the database should raise DatabaseVersionError
    with pytest.raises(DatabaseVersionError) as exc_info:
        initialize_workspace_database(db_path, read_only=False)

    assert exc_info.value.db_version == future_version
    assert exc_info.value.code_version == CURRENT_DB_VERSION
    assert "newer than code version" in str(exc_info.value)


def test_database_version_error_on_read_only_too(tmp_path: Path):
    """Test that DatabaseVersionError is raised even in read-only mode"""
    db_path = tmp_path / "workspace.db"

    # First, create a valid database
    db, _ = initialize_workspace_database(db_path, read_only=False)

    # Now update the version to be higher than CURRENT_DB_VERSION
    future_version = CURRENT_DB_VERSION + 10
    with db.bind_ctx(ALL_MODELS):
        WorkspaceSyncMetadata.update(db_version=future_version).where(
            WorkspaceSyncMetadata.id == "workspace"
        ).execute()

    close_workspace_database(db)

    # Even in read-only mode, DatabaseVersionError should be raised
    with pytest.raises(DatabaseVersionError):
        initialize_workspace_database(db_path, read_only=True)
