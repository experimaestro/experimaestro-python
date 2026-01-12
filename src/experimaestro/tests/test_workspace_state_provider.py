"""Tests for WorkspaceStateProvider

Tests cover:
1. Detection of v1 and v2 experiment layouts
2. Event detection when events-*.jsonl files are updated
3. Reading jobs, tags, dependencies from status.json
4. Getting experiment runs (multiple runs per experiment)
"""

import json
import pytest
import time
from pathlib import Path
from typing import Optional

from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.scheduler.state_status import (
    JobStateChangedEvent,
    JobProgressEvent,
    ExperimentUpdatedEvent,
)


# =============================================================================
# Mock Workspace Helpers - Reusable for other tests
# =============================================================================


def create_v1_experiment(
    workspace: Path,
    experiment_id: str,
    jobs: list[
        tuple[str, str, str]
    ],  # (task_id, job_id, status: "done"|"error"|"running")
) -> Path:
    """Create a v1 layout experiment.

    v1 layout: xp/{exp-id}/jobs/{task_id}/{job_id} -> symlink to jobs/{task_id}/{job_id}
    """
    exp_dir = workspace / "xp" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    jobs_dir = exp_dir / "jobs"
    jobs_dir.mkdir(exist_ok=True)

    # Create actual job directories in workspace/jobs/
    actual_jobs_dir = workspace / "jobs"

    for task_id, job_id, status in jobs:
        # Create actual job directory
        job_path = actual_jobs_dir / task_id / job_id
        job_path.mkdir(parents=True, exist_ok=True)

        # Create status file based on status
        scriptname = task_id.rsplit(".", 1)[-1]
        if status == "done":
            (job_path / f"{scriptname}.done").touch()
        elif status == "error":
            (job_path / f"{scriptname}.failed").touch()
        # "running" has no status file

        # Create symlink in experiment jobs/
        task_jobs_dir = jobs_dir / task_id
        task_jobs_dir.mkdir(exist_ok=True)
        link_path = task_jobs_dir / job_id
        link_path.symlink_to(job_path)

    return exp_dir


def create_v2_experiment(
    workspace: Path,
    experiment_id: str,
    runs: list[tuple[str, str, list[tuple[str, str, str]]]],  # (run_id, status, jobs)
    current_run: Optional[str] = None,
) -> Path:
    """Create a v2 layout experiment with multiple runs.

    v2 layout:
      experiments/{exp-id}/{run-id}/status.json
      .events/experiments/{exp-id} -> ../../experiments/{exp-id}/{current_run}
    """
    exp_dir = workspace / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    for run_id, run_status, jobs in runs:
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Create status.json with jobs
        jobs_dict = {}
        tags_dict = {}
        for task_id, job_id, job_status in jobs:
            jobs_dict[job_id] = {
                "job_id": job_id,
                "task_id": task_id,
                "state": job_status,  # "waiting", "running", "done", "error"
                "path": str(workspace / "jobs" / task_id / job_id),
            }
            # Add some tags for testing
            tags_dict[job_id] = {"task": task_id.split(".")[-1]}

        status_data = {
            "version": 1,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "events_count": 0,
            "hostname": "test-host",
            "started_at": "2026-01-01T10:00:00",
            "ended_at": "2026-01-01T11:00:00" if run_status != "active" else None,
            "status": run_status,
            "jobs": jobs_dict,
            "tags": tags_dict,
            "dependencies": {},
            "services": {},
        }
        (run_dir / "status.json").write_text(json.dumps(status_data))

        # Also create environment.json for compatibility
        env_data = {"run": {"status": run_status}}
        (run_dir / "environment.json").write_text(json.dumps(env_data))

    # Create symlink for current run
    if current_run:
        symlinks_dir = workspace / ".events" / "experiments"
        symlinks_dir.mkdir(parents=True, exist_ok=True)
        symlink = symlinks_dir / experiment_id
        if symlink.exists():
            symlink.unlink()
        target = Path("../..") / "experiments" / experiment_id / current_run
        symlink.symlink_to(target)

    return exp_dir


@pytest.fixture
def mock_workspace(tmp_path):
    """Create a comprehensive mock workspace with v1 and v2 experiments.

    Structure:
      workspace/
        .events/experiments/
          v2-multi-run -> ../../experiments/v2-multi-run/20260101_120000
          v2-failed -> ../../experiments/v2-failed/20260101_110000
        xp/                              # v1 layout
          v1-mixed/
            jobs/
              pkg.TaskA/job-a1 -> ...    # done
              pkg.TaskB/job-b1 -> ...    # error
        experiments/                     # v2 layout
          v2-multi-run/
            20260101_100000/             # older run, completed
            20260101_120000/             # current run, active
          v2-failed/
            20260101_110000/             # failed run
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # v1 experiment with mixed job statuses
    create_v1_experiment(
        workspace,
        "v1-mixed",
        jobs=[
            ("pkg.TaskA", "job-a1", "done"),
            ("pkg.TaskA", "job-a2", "done"),
            ("pkg.TaskB", "job-b1", "error"),
        ],
    )

    # v2 experiment with multiple runs
    create_v2_experiment(
        workspace,
        "v2-multi-run",
        runs=[
            (
                "20260101_100000",
                "completed",
                [
                    ("pkg.OldTask", "old-job-1", "done"),
                ],
            ),
            (
                "20260101_120000",
                "active",
                [
                    ("pkg.NewTask", "new-job-1", "running"),
                    ("pkg.NewTask", "new-job-2", "waiting"),
                    ("pkg.NewTask", "new-job-3", "done"),
                ],
            ),
        ],
        current_run="20260101_120000",
    )

    # v2 experiment that failed
    create_v2_experiment(
        workspace,
        "v2-failed",
        runs=[
            (
                "20260101_110000",
                "failed",
                [
                    ("pkg.FailTask", "fail-job-1", "error"),
                ],
            ),
        ],
        current_run="20260101_110000",
    )

    return workspace


# =============================================================================
# Tests: Experiment Detection
# =============================================================================


class TestGetExperiments:
    """Tests for get_experiments() method"""

    def test_detects_v1_and_v2_experiments(self, mock_workspace):
        """Both v1 and v2 layout experiments should be detected"""
        provider = WorkspaceStateProvider(mock_workspace)
        experiments = provider.get_experiments()

        exp_ids = {e.experiment_id for e in experiments}
        assert "v1-mixed" in exp_ids, "v1 experiment not detected"
        assert "v2-multi-run" in exp_ids, "v2 experiment not detected"
        assert "v2-failed" in exp_ids, "v2 failed experiment not detected"

    def test_experiment_stats_v1(self, mock_workspace):
        """v1 experiment should report correct job counts"""
        provider = WorkspaceStateProvider(mock_workspace)
        exp = provider.get_experiment("v1-mixed")

        assert exp is not None
        assert exp.total_jobs == 3
        assert exp.finished_jobs == 2  # 2 done
        assert exp.failed_jobs == 1  # 1 error

    def test_experiment_stats_v2(self, mock_workspace):
        """v2 experiment should report correct job counts from current run"""
        provider = WorkspaceStateProvider(mock_workspace)
        exp = provider.get_experiment("v2-multi-run")

        assert exp is not None
        assert exp.run_id == "20260101_120000"
        assert exp.total_jobs == 3
        assert exp.finished_jobs == 1  # 1 done
        assert exp.failed_jobs == 0

    def test_failed_experiment_stats(self, mock_workspace):
        """Failed experiment should report error jobs"""
        provider = WorkspaceStateProvider(mock_workspace)
        exp = provider.get_experiment("v2-failed")

        assert exp is not None
        assert exp.total_jobs == 1
        assert exp.failed_jobs == 1


# =============================================================================
# Tests: Experiment Runs
# =============================================================================


class TestGetExperimentRuns:
    """Tests for get_experiment_runs() method"""

    def test_v2_multiple_runs(self, mock_workspace):
        """v2 experiment should return all runs"""
        provider = WorkspaceStateProvider(mock_workspace)
        runs = provider.get_experiment_runs("v2-multi-run")

        assert len(runs) == 2
        run_ids = {r.run_id for r in runs}
        assert "20260101_100000" in run_ids
        assert "20260101_120000" in run_ids

    def test_v2_run_metadata(self, mock_workspace):
        """Run should contain correct metadata"""
        from experimaestro.scheduler.interfaces import ExperimentStatus

        provider = WorkspaceStateProvider(mock_workspace)
        runs = provider.get_experiment_runs("v2-multi-run")

        current_run = next(r for r in runs if r.run_id == "20260101_120000")
        assert current_run.status == ExperimentStatus.RUNNING
        assert current_run.hostname == "test-host"
        assert current_run.total_jobs == 3

    def test_v1_synthetic_run(self, mock_workspace):
        """v1 experiment should return synthetic 'v1' run"""
        provider = WorkspaceStateProvider(mock_workspace)
        runs = provider.get_experiment_runs("v1-mixed")

        assert len(runs) == 1
        assert runs[0].run_id == "v1"
        assert runs[0].total_jobs == 3


# =============================================================================
# Tests: Jobs
# =============================================================================


class TestGetJobs:
    """Tests for get_jobs() method"""

    def test_get_jobs_v2_current_run(self, mock_workspace):
        """Should return jobs from current run"""
        provider = WorkspaceStateProvider(mock_workspace)
        jobs = provider.get_jobs("v2-multi-run")

        job_ids = {j.identifier for j in jobs}
        assert "new-job-1" in job_ids
        assert "new-job-2" in job_ids
        assert "new-job-3" in job_ids
        assert "old-job-1" not in job_ids  # From older run

    def test_get_jobs_specific_run(self, mock_workspace):
        """Should return jobs from specified run"""
        provider = WorkspaceStateProvider(mock_workspace)
        jobs = provider.get_jobs("v2-multi-run", run_id="20260101_100000")

        job_ids = {j.identifier for j in jobs}
        assert "old-job-1" in job_ids
        assert "new-job-1" not in job_ids

    def test_get_jobs_v1(self, mock_workspace):
        """Should return jobs from v1 experiment"""
        provider = WorkspaceStateProvider(mock_workspace)
        jobs = provider.get_jobs("v1-mixed")

        job_ids = {j.identifier for j in jobs}
        assert "job-a1" in job_ids
        assert "job-b1" in job_ids


# =============================================================================
# Tests: Event Detection (File Watcher)
# =============================================================================


class TestEventWatcher:
    """Tests for EventFileWatcher - event detection when files change"""

    def test_detects_new_job_event(self, mock_workspace):
        """Should detect new events written to events-*.jsonl"""
        provider = WorkspaceStateProvider(mock_workspace)

        events_received = []

        def listener(event):
            events_received.append(event)

        provider.add_listener(listener)

        try:
            # Create events file in new subdirectory format
            exp_dir = mock_workspace / ".events" / "experiments" / "v2-multi-run"
            exp_dir.mkdir(parents=True, exist_ok=True)
            events_file = exp_dir / "events-1.jsonl"

            # Write a job state change event
            event_data = {
                "event_type": "JobStateChangedEvent",
                "job_id": "new-job-1",
                "state": "done",
                "timestamp": time.time(),
            }
            with open(events_file, "w") as f:
                f.write(json.dumps(event_data) + "\n")

            # Wait for watcher to pick up
            time.sleep(1.0)

            # Check that event was detected
            job_events = [
                e for e in events_received if isinstance(e, JobStateChangedEvent)
            ]
            assert len(job_events) >= 1, f"Expected job event, got: {events_received}"
            assert any(e.job_id == "new-job-1" for e in job_events)
        finally:
            provider.close()

    def test_detects_multiple_events(self, mock_workspace):
        """Should detect multiple events appended to file"""
        provider = WorkspaceStateProvider(mock_workspace)

        events_received = []
        provider.add_listener(lambda e: events_received.append(e))

        try:
            # Create events file in new subdirectory format
            exp_dir = mock_workspace / ".events" / "experiments" / "v2-multi-run"
            exp_dir.mkdir(parents=True, exist_ok=True)
            events_file = exp_dir / "events-1.jsonl"

            # Write first event
            with open(events_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "event_type": "JobStateChangedEvent",
                            "job_id": "job-1",
                            "state": "running",
                        }
                    )
                    + "\n"
                )

            time.sleep(0.7)

            # Append second event
            with open(events_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "event_type": "JobStateChangedEvent",
                            "job_id": "job-2",
                            "state": "done",
                        }
                    )
                    + "\n"
                )

            time.sleep(0.7)

            job_events = [
                e for e in events_received if isinstance(e, JobStateChangedEvent)
            ]
            job_ids = {e.job_id for e in job_events}
            assert "job-1" in job_ids
            assert "job-2" in job_ids
        finally:
            provider.close()

    def test_detects_job_state_from_job_events(self, mock_workspace):
        """Should detect job state changes from job event files"""
        provider = WorkspaceStateProvider(mock_workspace)

        events_received = []
        provider.add_listener(lambda e: events_received.append(e))

        try:
            # Create job events file in .events/jobs/{task_id}/event-{job_id}-{count}.jsonl
            task_id = "my.test.task"
            job_id = "test-job-123"
            task_dir = mock_workspace / ".events" / "jobs" / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            events_file = task_dir / f"event-{job_id}-0.jsonl"

            # Write job state changed event (from job process)
            with open(events_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "event_type": "JobStateChangedEvent",
                            "job_id": job_id,
                            "state": "running",
                            "started_time": time.time(),
                        }
                    )
                    + "\n"
                )

            time.sleep(1.0)

            # Should have received job updated event
            job_events = [
                e for e in events_received if isinstance(e, JobStateChangedEvent)
            ]
            assert len(job_events) >= 1, f"Expected job event, got: {events_received}"
            assert any(e.job_id == job_id for e in job_events)
        finally:
            provider.close()

    def test_detects_job_progress_from_job_events(self, mock_workspace):
        """Should detect job progress updates from job event files"""
        provider = WorkspaceStateProvider(mock_workspace)

        events_received = []
        provider.add_listener(lambda e: events_received.append(e))

        try:
            # Create job events file in .events/jobs/{task_id}/event-{job_id}-{count}.jsonl
            task_id = "my.progress.task"
            job_id = "test-job-progress"
            task_dir = mock_workspace / ".events" / "jobs" / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            events_file = task_dir / f"event-{job_id}-0.jsonl"

            # Write job progress event
            with open(events_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "event_type": "JobProgressEvent",
                            "job_id": job_id,
                            "level": 0,
                            "progress": 0.5,
                            "desc": "Halfway done",
                        }
                    )
                    + "\n"
                )

            time.sleep(1.0)

            # Should have received job progress event
            job_events = [e for e in events_received if isinstance(e, JobProgressEvent)]
            assert len(job_events) >= 1, (
                f"Expected job progress event, got: {events_received}"
            )
            assert any(e.job_id == job_id for e in job_events)
        finally:
            provider.close()

    def test_detects_experiment_finalization(self, mock_workspace):
        """Should detect when experiment finalizes (events file deleted)"""
        provider = WorkspaceStateProvider(mock_workspace)

        events_received = []
        provider.add_listener(lambda e: events_received.append(e))

        try:
            # Create events file in new subdirectory format
            exp_dir = mock_workspace / ".events" / "experiments" / "v2-multi-run"
            exp_dir.mkdir(parents=True, exist_ok=True)
            events_file = exp_dir / "events-1.jsonl"

            # Create then delete events file (simulates finalization)
            events_file.touch()
            time.sleep(0.7)
            events_file.unlink()
            time.sleep(0.7)

            # Should have received experiment updated event
            exp_events = [
                e for e in events_received if isinstance(e, ExperimentUpdatedEvent)
            ]
            assert len(exp_events) >= 1
            assert any(e.experiment_id == "v2-multi-run" for e in exp_events)
        finally:
            provider.close()


# =============================================================================
# Tests: Tags and Dependencies
# =============================================================================


class TestTagsAndDependencies:
    """Tests for get_tags_map() and get_dependencies_map()"""

    def test_get_tags_map(self, mock_workspace):
        """Should return tags for jobs"""
        provider = WorkspaceStateProvider(mock_workspace)
        tags = provider.get_tags_map("v2-multi-run")

        assert "new-job-1" in tags
        assert tags["new-job-1"]["task"] == "NewTask"

    def test_get_dependencies_map(self, mock_workspace):
        """Should return dependencies (empty in mock)"""
        provider = WorkspaceStateProvider(mock_workspace)
        deps = provider.get_dependencies_map("v2-multi-run")

        # Our mock doesn't set dependencies, so should be empty
        assert isinstance(deps, dict)


# =============================================================================
# Tests: Orphan Job Detection
# =============================================================================


def create_orphan_job(workspace: Path, task_id: str, job_id: str) -> Path:
    """Create an orphan job (not linked to any experiment)"""
    job_path = workspace / "jobs" / task_id / job_id
    job_path.mkdir(parents=True, exist_ok=True)

    # Create a .done marker to indicate it's finished
    scriptname = task_id.rsplit(".", 1)[-1]
    (job_path / f"{scriptname}.done").touch()

    return job_path


class TestOrphanJobDetection:
    """Tests for get_orphan_jobs() method"""

    def test_no_orphans_in_empty_workspace(self, tmp_path):
        """Empty workspace should have no orphans"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        provider = WorkspaceStateProvider(workspace)
        orphans = provider.get_orphan_jobs()

        assert len(orphans) == 0

    def test_v1_jobs_not_detected_as_orphans(self, mock_workspace):
        """Jobs in v1 experiments should NOT be detected as orphans"""
        provider = WorkspaceStateProvider(mock_workspace)
        orphans = provider.get_orphan_jobs()

        # The mock_workspace has v1-mixed experiment with jobs:
        # job-a1, job-a2 (TaskA), job-b1 (TaskB)
        # These should NOT be in orphans
        orphan_ids = {j.identifier for j in orphans}

        assert "job-a1" not in orphan_ids, "v1 job-a1 incorrectly detected as orphan"
        assert "job-a2" not in orphan_ids, "v1 job-a2 incorrectly detected as orphan"
        assert "job-b1" not in orphan_ids, "v1 job-b1 incorrectly detected as orphan"

    def test_v2_jobs_not_detected_as_orphans(self, mock_workspace):
        """Jobs in v2 experiments should NOT be detected as orphans"""
        provider = WorkspaceStateProvider(mock_workspace)
        orphans = provider.get_orphan_jobs()

        # The mock_workspace has v2-multi-run experiment with jobs in status.json
        # These should NOT be in orphans
        orphan_ids = {j.identifier for j in orphans}

        # Jobs from v2-multi-run (both runs)
        assert "old-job-1" not in orphan_ids, (
            "v2 old-job-1 incorrectly detected as orphan"
        )
        assert "new-job-1" not in orphan_ids, (
            "v2 new-job-1 incorrectly detected as orphan"
        )
        assert "new-job-2" not in orphan_ids, (
            "v2 new-job-2 incorrectly detected as orphan"
        )
        assert "new-job-3" not in orphan_ids, (
            "v2 new-job-3 incorrectly detected as orphan"
        )

        # Jobs from v2-failed
        assert "fail-job-1" not in orphan_ids, (
            "v2 fail-job-1 incorrectly detected as orphan"
        )

    def test_orphan_job_is_detected(self, mock_workspace):
        """Jobs not in any experiment should be detected as orphans"""
        # Create an orphan job
        orphan_path = create_orphan_job(
            mock_workspace, "pkg.OrphanTask", "orphan-job-1"
        )

        provider = WorkspaceStateProvider(mock_workspace)
        orphans = provider.get_orphan_jobs()

        orphan_ids = {j.identifier for j in orphans}
        assert "orphan-job-1" in orphan_ids, "Orphan job not detected"

        # Verify the orphan has correct properties
        orphan = next(j for j in orphans if j.identifier == "orphan-job-1")
        assert orphan.task_id == "pkg.OrphanTask"
        assert orphan.path == orphan_path

    def test_multiple_orphans_detected(self, mock_workspace):
        """Multiple orphan jobs should all be detected"""
        # Create multiple orphan jobs
        create_orphan_job(mock_workspace, "pkg.Task1", "orphan-1")
        create_orphan_job(mock_workspace, "pkg.Task2", "orphan-2")
        create_orphan_job(mock_workspace, "pkg.Task2", "orphan-3")

        provider = WorkspaceStateProvider(mock_workspace)
        orphans = provider.get_orphan_jobs()

        orphan_ids = {j.identifier for j in orphans}
        assert "orphan-1" in orphan_ids
        assert "orphan-2" in orphan_ids
        assert "orphan-3" in orphan_ids

    def test_experiment_jobs_never_detected_as_orphans(self, mock_workspace):
        """All jobs that exist in experiments should never be orphans"""
        provider = WorkspaceStateProvider(mock_workspace)

        # Get all jobs from all experiments
        all_experiment_jobs = set()

        # v1 experiment jobs
        v1_jobs = provider.get_jobs("v1-mixed")
        for j in v1_jobs:
            all_experiment_jobs.add(j.identifier)

        # v2 experiment jobs (all runs)
        for run in provider.get_experiment_runs("v2-multi-run"):
            jobs = provider.get_jobs("v2-multi-run", run_id=run.run_id)
            for j in jobs:
                all_experiment_jobs.add(j.identifier)

        # v2 failed experiment jobs
        for run in provider.get_experiment_runs("v2-failed"):
            jobs = provider.get_jobs("v2-failed", run_id=run.run_id)
            for j in jobs:
                all_experiment_jobs.add(j.identifier)

        # Get orphans
        orphans = provider.get_orphan_jobs()
        orphan_ids = {j.identifier for j in orphans}

        # Verify no experiment job is in orphans
        intersection = all_experiment_jobs & orphan_ids
        assert len(intersection) == 0, (
            f"Experiment jobs incorrectly detected as orphans: {intersection}"
        )

    def test_orphan_with_v1_and_v2_mixed(self, tmp_path):
        """Test orphan detection with both v1 and v2 experiments present"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create v1 experiment with one job
        create_v1_experiment(
            workspace,
            "v1-exp",
            jobs=[("pkg.TaskA", "v1-job", "done")],
        )

        # Create v2 experiment with one job
        create_v2_experiment(
            workspace,
            "v2-exp",
            runs=[
                ("20260101_100000", "completed", [("pkg.TaskB", "v2-job", "done")]),
            ],
            current_run="20260101_100000",
        )

        # Create two orphan jobs
        create_orphan_job(workspace, "pkg.TaskC", "orphan-1")
        create_orphan_job(workspace, "pkg.TaskD", "orphan-2")

        provider = WorkspaceStateProvider(workspace)
        orphans = provider.get_orphan_jobs()
        orphan_ids = {j.identifier for j in orphans}

        # Only the orphans should be detected
        assert len(orphans) == 2
        assert "orphan-1" in orphan_ids
        assert "orphan-2" in orphan_ids
        assert "v1-job" not in orphan_ids
        assert "v2-job" not in orphan_ids
