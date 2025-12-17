"""Tests for SchedulerStateProvider integration with Scheduler"""

from pathlib import Path
from experimaestro import Task, Param, experiment
from experimaestro.scheduler.state_provider import SchedulerStateProvider


class TestTask(Task):
    """Simple task for testing"""

    value: Param[int]

    def execute(self):
        # Task just completes immediately
        return


def test_scheduler_creates_state_provider(tmp_path: Path):
    """Test that scheduler automatically creates a state provider"""
    from experimaestro.scheduler.base import Scheduler

    scheduler = Scheduler.instance()

    # Verify state provider was created
    assert scheduler.state_provider is not None
    assert isinstance(scheduler.state_provider, SchedulerStateProvider)


def test_state_provider_tracks_experiments(tmp_path: Path):
    """Test that state provider tracks experiments"""
    workdir = tmp_path / "workspace"

    with experiment(workdir, "test_exp1", port=0) as xp:
        # Submit a task
        _ = TestTask.C(value=42).submit()

        # Get state provider
        scheduler = xp.scheduler
        state_provider = scheduler.state_provider

        # Wait for the experiment to complete (ensures all listeners have been called)
        xp.wait()

        # Check experiments
        experiments = state_provider.get_experiments()
        assert len(experiments) >= 1

        # Find our experiment
        our_exp = next(
            (e for e in experiments if e["experiment_id"] == "test_exp1"), None
        )
        assert our_exp is not None
        assert our_exp["total_jobs"] >= 1

        # Check jobs
        jobs = state_provider.get_jobs("test_exp1")
        assert len(jobs) >= 1

        # Find our job (taskId format changed to camelCase)
        our_job = next((j for j in jobs if "testtask" in j["taskId"].lower()), None)
        assert our_job is not None
        assert our_job["status"] in ["unscheduled", "waiting", "running", "done"]


def test_state_provider_multi_experiment(tmp_path: Path):
    """Test that state provider handles multiple experiments"""
    workdir = tmp_path / "workspace"

    with experiment(workdir, "exp1", port=0) as xp1:
        _ = TestTask.C(value=1).submit()

        # Create second experiment in same workspace
        with experiment(workdir, "exp2", port=0) as xp2:
            _ = TestTask.C(value=2).submit()

            # Get state provider (should be same instance)
            assert xp1.scheduler.state_provider is xp2.scheduler.state_provider

            state_provider = xp1.scheduler.state_provider

            # Wait for both experiments to complete
            xp1.wait()
            xp2.wait()

            # Should have both experiments
            experiments = state_provider.get_experiments()
            exp_ids = [e["experiment_id"] for e in experiments]

            assert "exp1" in exp_ids
            assert "exp2" in exp_ids

            # Each experiment should have its own jobs
            jobs1 = state_provider.get_jobs("exp1")
            jobs2 = state_provider.get_jobs("exp2")

            assert len(jobs1) >= 1
            assert len(jobs2) >= 1


def test_state_provider_job_state_updates(tmp_path: Path):
    """Test that state provider receives job state updates"""
    workdir = tmp_path / "workspace"

    with experiment(workdir, "state_test", port=0) as xp:
        _ = TestTask.C(value=99).submit()

        state_provider = xp.scheduler.state_provider

        # Wait for experiment to complete
        xp.wait()

        # Get the job
        jobs = state_provider.get_jobs("state_test")
        assert len(jobs) >= 1

        our_job = next((j for j in jobs if "testtask" in j["taskId"].lower()), None)
        assert our_job is not None

        # Job should have timing information (at least submission time)
        assert our_job["submitted"] is not None


def test_state_provider_persistence(tmp_path: Path):
    """Test that state is persisted to database"""
    workdir = tmp_path / "workspace"

    # Create experiment and submit task
    with experiment(workdir, "persist_test", port=0) as xp:
        _ = TestTask.C(value=123).submit()

        # Wait for experiment to complete
        xp.wait()

    # Verify database file was created
    db_path = workdir / "xp" / "persist_test" / "experiment.db"
    assert db_path.exists()

    # Open database in read-only mode and verify data
    from experimaestro.scheduler.state_provider import ExperimentStateProvider

    provider = ExperimentStateProvider(workdir, "persist_test", read_only=True)
    jobs = provider.get_jobs()

    assert len(jobs) >= 1
    our_job = next((j for j in jobs if "testtask" in j["taskId"].lower()), None)
    assert our_job is not None

    provider.close()
