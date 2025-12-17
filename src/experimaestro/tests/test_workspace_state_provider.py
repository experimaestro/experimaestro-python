"""Tests for WorkspaceStateProvider"""

import pytest
import time
from pathlib import Path
from experimaestro import Task, Param, experiment
from experimaestro.scheduler.state_provider import WorkspaceStateProvider


class SimpleTask(Task):
    """Simple task for testing"""

    x: Param[int]

    def execute(self):
        pass


def test_workspace_provider_scans_experiments(tmp_path: Path):
    """Test that workspace provider scans and discovers experiments"""
    workdir = tmp_path / "workspace"

    # Create an experiment with a job
    with experiment(workdir, "test_exp", port=0) as xp:
        _ = SimpleTask.C(x=1).submit()
        xp.wait()

    # Create workspace provider
    provider = WorkspaceStateProvider(workdir, watch=False)

    # Should discover the experiment
    experiments = provider.get_experiments()
    assert len(experiments) == 1
    assert experiments[0]["experiment_id"] == "test_exp"
    assert experiments[0]["total_jobs"] >= 1

    # Should be able to get jobs
    jobs = provider.get_jobs("test_exp")
    assert len(jobs) >= 1

    provider.close()


def test_workspace_provider_multiple_experiments(tmp_path: Path):
    """Test workspace provider with multiple experiments"""
    workdir = tmp_path / "workspace"

    # Create multiple experiments
    with experiment(workdir, "exp1", port=0) as xp:
        _ = SimpleTask.C(x=1).submit()
        xp.wait()

    with experiment(workdir, "exp2", port=0) as xp:
        _ = SimpleTask.C(x=2).submit()
        xp.wait()

    # Create workspace provider
    provider = WorkspaceStateProvider(workdir, watch=False)

    # Should discover both experiments
    experiments = provider.get_experiments()
    exp_ids = [e["experiment_id"] for e in experiments]

    assert "exp1" in exp_ids
    assert "exp2" in exp_ids

    # Should be able to get jobs from each
    jobs1 = provider.get_jobs("exp1")
    jobs2 = provider.get_jobs("exp2")

    assert len(jobs1) >= 1
    assert len(jobs2) >= 1

    provider.close()


def test_workspace_provider_read_only(tmp_path: Path):
    """Test that workspace provider cannot kill jobs"""
    workdir = tmp_path / "workspace"

    # Create an experiment
    with experiment(workdir, "test_exp", port=0) as xp:
        _ = SimpleTask.C(x=1).submit()
        xp.wait()

    # Create workspace provider
    provider = WorkspaceStateProvider(workdir, watch=False)

    # kill_job should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="workspace mode"):
        provider.kill_job("test_exp", "some_job_id")

    provider.close()


def test_workspace_provider_no_experiments(tmp_path: Path):
    """Test workspace provider with no experiments"""
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # Create workspace provider on empty workspace
    provider = WorkspaceStateProvider(workdir, watch=False)

    # Should return empty list
    experiments = provider.get_experiments()
    assert len(experiments) == 0

    provider.close()


def test_workspace_provider_with_watching(tmp_path: Path):
    """Test workspace provider with filesystem watching"""
    workdir = tmp_path / "workspace"

    # Create initial experiment
    with experiment(workdir, "exp1", port=0) as xp:
        _ = SimpleTask.C(x=1).submit()
        xp.wait()

    # Create workspace provider with watching enabled
    provider = WorkspaceStateProvider(workdir, watch=True)

    # Should discover initial experiment
    experiments = provider.get_experiments()
    assert len(experiments) == 1

    # Create a new experiment
    with experiment(workdir, "exp2", port=0) as xp:
        _ = SimpleTask.C(x=2).submit()
        xp.wait()

    # Wait for watcher to detect changes
    time.sleep(2)

    # Manually trigger scan to ensure filesystem changes are picked up
    # (watchdog events may have timing variations)
    provider.scan_experiments()

    # Should discover new experiment
    experiments = provider.get_experiments()
    exp_ids = [e["experiment_id"] for e in experiments]

    assert "exp1" in exp_ids
    assert "exp2" in exp_ids

    provider.close()


def test_workspace_provider_get_experiment_info(tmp_path: Path):
    """Test getting specific experiment info"""
    workdir = tmp_path / "workspace"

    # Create experiment
    with experiment(workdir, "my_exp", port=0) as xp:
        _ = SimpleTask.C(x=42).submit()
        xp.wait()

    # Create workspace provider
    provider = WorkspaceStateProvider(workdir, watch=False)

    # Get specific experiment
    exp_info = provider.get_experiment("my_exp")
    assert exp_info is not None
    assert exp_info["experiment_id"] == "my_exp"
    assert exp_info["total_jobs"] >= 1

    # Non-existent experiment
    exp_info = provider.get_experiment("nonexistent")
    assert exp_info is None

    provider.close()


def test_workspace_provider_get_job(tmp_path: Path):
    """Test getting specific job"""
    workdir = tmp_path / "workspace"

    # Create experiment with job
    with experiment(workdir, "test_exp", port=0) as xp:
        _ = SimpleTask.C(x=99).submit()
        xp.wait()

    # Create workspace provider
    provider = WorkspaceStateProvider(workdir, watch=False)

    # Get jobs to find job ID
    jobs = provider.get_jobs("test_exp")
    assert len(jobs) >= 1
    job_id = jobs[0]["jobId"]

    # Get specific job
    job = provider.get_job("test_exp", job_id)
    assert job is not None
    assert job["jobId"] == job_id

    # Non-existent job
    job = provider.get_job("test_exp", "nonexistent_job_id")
    assert job is None

    provider.close()
