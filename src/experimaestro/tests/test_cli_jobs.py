"""Functional tests for CLI jobs commands

Tests the jobs list, kill, and clean commands using the WorkspaceStateProvider.
"""

import pytest
import time
from click.testing import CliRunner

from experimaestro.cli import cli
from experimaestro.scheduler.state_provider import WorkspaceStateProvider
from experimaestro.scheduler.state_db import (
    initialize_workspace_database,
    close_workspace_database,
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    ALL_MODELS,
)
from experimaestro.scheduler.workspace import WORKSPACE_VERSION


@pytest.fixture
def workspace_path(tmp_path):
    """Create a workspace directory with database initialized"""
    ws_path = tmp_path / "workspace"
    ws_path.mkdir()

    # Create version file with current workspace version
    (ws_path / ".__experimaestro__").write_text(str(WORKSPACE_VERSION))

    # Initialize database
    xpm_dir = ws_path / ".experimaestro"
    xpm_dir.mkdir()
    db_path = xpm_dir / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    # Create jobs directory
    jobs_dir = ws_path / "jobs"
    jobs_dir.mkdir()

    yield ws_path

    close_workspace_database(db)


@pytest.fixture
def workspace_with_jobs(workspace_path):
    """Create a workspace with some test jobs in the database"""
    db_path = workspace_path / ".experimaestro" / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create experiment and run
        ExperimentModel.create(experiment_id="test_exp", current_run_id="run_001")
        ExperimentRunModel.create(
            experiment_id="test_exp", run_id="run_001", status="active"
        )

        # Create jobs with different states
        jobs_data = [
            ("job_done_1", "mymodule.DoneTask", "done"),
            ("job_done_2", "mymodule.DoneTask", "done"),
            ("job_error_1", "mymodule.ErrorTask", "error"),
            ("job_running_1", "mymodule.RunningTask", "running"),
        ]

        jobs_dir = workspace_path / "jobs"
        for job_id, task_id, state in jobs_data:
            JobModel.create(
                job_id=job_id,
                experiment_id="test_exp",
                run_id="run_001",
                task_id=task_id,
                locator=job_id,
                state=state,
                submitted_time=time.time(),
            )
            # Create job directories
            job_dir = jobs_dir / task_id / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            # Create marker files based on state
            script_name = task_id.rsplit(".", 1)[-1]
            if state == "done":
                (job_dir / f"{script_name}.done").touch()
            elif state == "error":
                (job_dir / f"{script_name}.failed").write_text("{}")

    close_workspace_database(db)

    yield workspace_path


def test_jobs_list_empty_workspace(workspace_path):
    """Test jobs list on empty workspace"""
    runner = CliRunner()
    result = runner.invoke(cli, ["jobs", "--workdir", str(workspace_path), "list"])

    assert result.exit_code == 0
    assert "No jobs found" in result.output


def test_jobs_list_with_jobs(workspace_with_jobs):
    """Test jobs list shows all jobs"""
    runner = CliRunner()
    result = runner.invoke(cli, ["jobs", "--workdir", str(workspace_with_jobs), "list"])

    assert result.exit_code == 0
    assert "job_done_1" in result.output
    assert "job_done_2" in result.output
    assert "job_error_1" in result.output
    assert "job_running_1" in result.output
    assert "DONE" in result.output
    assert "FAIL" in result.output


def test_jobs_list_with_filter(workspace_with_jobs):
    """Test jobs list with filter expression"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "list",
            "--filter",
            '@state = "done"',
        ],
    )

    assert result.exit_code == 0
    assert "job_done_1" in result.output
    assert "job_done_2" in result.output
    assert "job_error_1" not in result.output
    assert "job_running_1" not in result.output


def test_jobs_list_fullpath(workspace_with_jobs):
    """Test jobs list with fullpath option"""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["jobs", "--workdir", str(workspace_with_jobs), "list", "--fullpath"]
    )

    assert result.exit_code == 0
    # Should show full paths instead of task_id/job_id format
    assert str(workspace_with_jobs / "jobs") in result.output


@pytest.fixture
def workspace_with_timed_jobs(workspace_path):
    """Create a workspace with jobs that have different submission times"""
    db_path = workspace_path / ".experimaestro" / "workspace.db"
    db = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        ExperimentModel.create(experiment_id="test_exp", current_run_id="run_001")
        ExperimentRunModel.create(
            experiment_id="test_exp", run_id="run_001", status="active"
        )

        # Create jobs with different submission times (oldest to newest)
        base_time = time.time()
        jobs_data = [
            ("job_oldest", "mymodule.Task", "done", base_time - 3600),  # 1 hour ago
            ("job_middle", "mymodule.Task", "done", base_time - 1800),  # 30 min ago
            ("job_newest", "mymodule.Task", "done", base_time),  # now
        ]

        jobs_dir = workspace_path / "jobs"
        for job_id, task_id, state, submit_time in jobs_data:
            JobModel.create(
                job_id=job_id,
                experiment_id="test_exp",
                run_id="run_001",
                task_id=task_id,
                locator=job_id,
                state=state,
                submitted_time=submit_time,
            )
            job_dir = jobs_dir / task_id / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            script_name = task_id.rsplit(".", 1)[-1]
            (job_dir / f"{script_name}.done").touch()

    close_workspace_database(db)
    yield workspace_path


def test_jobs_list_sorted_by_date(workspace_with_timed_jobs):
    """Test that jobs are sorted by submission date (most recent first)"""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["jobs", "--workdir", str(workspace_with_timed_jobs), "list"]
    )

    assert result.exit_code == 0
    output = result.output

    # Verify order: newest should appear before middle, middle before oldest
    newest_pos = output.find("job_newest")
    middle_pos = output.find("job_middle")
    oldest_pos = output.find("job_oldest")

    assert newest_pos < middle_pos < oldest_pos, (
        f"Jobs should be sorted by date (newest first). "
        f"Positions: newest={newest_pos}, middle={middle_pos}, oldest={oldest_pos}"
    )


def test_jobs_list_with_count(workspace_with_timed_jobs):
    """Test jobs list with --count option"""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["jobs", "--workdir", str(workspace_with_timed_jobs), "list", "-c", "2"]
    )

    assert result.exit_code == 0
    output = result.output

    # Should only show 2 most recent jobs
    assert "job_newest" in output
    assert "job_middle" in output
    assert "job_oldest" not in output


def test_jobs_list_count_zero_shows_all(workspace_with_timed_jobs):
    """Test that --count 0 shows all jobs (default behavior)"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["jobs", "--workdir", str(workspace_with_timed_jobs), "list", "--count", "0"],
    )

    assert result.exit_code == 0
    output = result.output

    # All jobs should be present
    assert "job_newest" in output
    assert "job_middle" in output
    assert "job_oldest" in output


def test_jobs_list_with_experiment_filter(workspace_with_jobs):
    """Test jobs list filtered by experiment"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "list",
            "--experiment",
            "test_exp",
        ],
    )

    assert result.exit_code == 0
    assert "job_done_1" in result.output

    # Test with non-existent experiment
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "list",
            "--experiment",
            "nonexistent",
        ],
    )
    assert result.exit_code == 0
    assert "No jobs found" in result.output


def test_jobs_clean_dry_run(workspace_with_jobs):
    """Test jobs clean without --perform (dry run)"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "clean",
            "--filter",
            '@state = "done"',
        ],
    )

    assert result.exit_code == 0
    assert "dry run" in result.output.lower()

    # Verify directories still exist
    jobs_dir = workspace_with_jobs / "jobs"
    assert (jobs_dir / "mymodule.DoneTask" / "job_done_1").exists()
    assert (jobs_dir / "mymodule.DoneTask" / "job_done_2").exists()


def test_jobs_clean_with_perform(workspace_with_jobs):
    """Test jobs clean with --perform actually deletes jobs"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "clean",
            "--filter",
            '@state = "done"',
            "--perform",
        ],
    )

    assert result.exit_code == 0
    assert "Cleaned" in result.output

    # Verify directories are deleted
    jobs_dir = workspace_with_jobs / "jobs"
    assert not (jobs_dir / "mymodule.DoneTask" / "job_done_1").exists()
    assert not (jobs_dir / "mymodule.DoneTask" / "job_done_2").exists()

    # Verify database entries are deleted
    provider = WorkspaceStateProvider.get_instance(workspace_with_jobs, read_only=True)
    try:
        jobs = provider.get_all_jobs()
        job_ids = [j.identifier for j in jobs]
        assert "job_done_1" not in job_ids
        assert "job_done_2" not in job_ids
        # Error and running jobs should still exist
        assert "job_error_1" in job_ids
        assert "job_running_1" in job_ids
    finally:
        provider.close()


def test_jobs_clean_does_not_clean_running(workspace_with_jobs):
    """Test that jobs clean does not clean running jobs"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["jobs", "--workdir", str(workspace_with_jobs), "clean", "--perform"],
    )

    assert result.exit_code == 0

    # Verify running job is NOT cleaned
    provider = WorkspaceStateProvider.get_instance(workspace_with_jobs, read_only=True)
    try:
        jobs = provider.get_all_jobs()
        job_ids = [j.identifier for j in jobs]
        assert "job_running_1" in job_ids
    finally:
        provider.close()


def test_jobs_kill_dry_run(workspace_with_jobs):
    """Test jobs kill without --perform (dry run)"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "kill",
            "--filter",
            '@state = "running"',
        ],
    )

    assert result.exit_code == 0
    assert "dry run" in result.output.lower()


def test_jobs_kill_with_perform(workspace_with_jobs):
    """Test jobs kill with --perform actually kills jobs and updates DB"""
    import json
    from unittest.mock import MagicMock, patch

    # Create a PID file for the running job so kill can find it
    jobs_dir = workspace_with_jobs / "jobs"
    running_job_dir = jobs_dir / "mymodule.RunningTask" / "job_running_1"
    pid_file = running_job_dir / "RunningTask.pid"
    pid_file.write_text(json.dumps({"type": "local", "pid": 12345}))

    # Create a mock process
    mock_process = MagicMock()
    mock_process.kill = MagicMock()

    with patch(
        "experimaestro.connectors.Process.fromDefinition", return_value=mock_process
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "jobs",
                "--workdir",
                str(workspace_with_jobs),
                "kill",
                "--filter",
                '@state = "running"',
                "--perform",
            ],
        )

        assert result.exit_code == 0
        assert "KILLED" in result.output

        # Verify kill was called on the mock process
        mock_process.kill.assert_called_once()

    # Verify database state was updated to error
    provider = WorkspaceStateProvider.get_instance(workspace_with_jobs, read_only=True)
    try:
        jobs = provider.get_all_jobs()
        running_job = next((j for j in jobs if j.identifier == "job_running_1"), None)
        assert running_job is not None
        # State should be updated to error after kill
        from experimaestro.scheduler import JobState

        assert running_job.state == JobState.ERROR
    finally:
        provider.close()


def test_jobs_path(workspace_with_jobs):
    """Test jobs path command"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "path",
            "mymodule.DoneTask/job_done_1",
        ],
    )

    assert result.exit_code == 0
    expected_path = workspace_with_jobs / "jobs" / "mymodule.DoneTask" / "job_done_1"
    assert str(expected_path) in result.output


def test_jobs_path_nonexistent(workspace_with_jobs):
    """Test jobs path command for non-existent job"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "path",
            "mymodule.Task/nonexistent",
        ],
    )

    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_jobs_log_nonexistent(workspace_with_jobs):
    """Test jobs log command for non-existent log"""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "jobs",
            "--workdir",
            str(workspace_with_jobs),
            "log",
            "mymodule.DoneTask/job_done_1",
        ],
    )

    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_sync_detects_dead_process_and_updates_disk(tmp_path):
    """Test that sync detects a dead 'running' job and updates disk state

    This tests the scenario where a job crashed without updating its state:
    - Job has a .pid file (marked as running)
    - Process is no longer running (dead)
    - Sync should detect this and create a .failed file
    """
    import json
    from unittest.mock import MagicMock, patch

    from experimaestro.scheduler.state_sync import check_process_alive

    # Create a job directory with a PID file for a non-existent process
    job_path = tmp_path / "jobs" / "mymodule.CrashedTask" / "job_crashed_1"
    job_path.mkdir(parents=True)

    scriptname = "CrashedTask"
    pid_file = job_path / f"{scriptname}.pid"
    pid_file.write_text(
        json.dumps({"type": "local", "pid": 999999})
    )  # Non-existent PID

    # Create a mock process that reports as finished/dead
    mock_process = MagicMock()
    mock_state = MagicMock()
    mock_state.finished = True
    mock_state.exitcode = 137  # Killed by signal

    # Mock aio_state to return the finished state
    async def mock_aio_state(timeout):
        return mock_state

    mock_process.aio_state = mock_aio_state

    with patch(
        "experimaestro.connectors.Process.fromDefinition", return_value=mock_process
    ):
        # Check that process is detected as dead
        is_alive = check_process_alive(job_path, scriptname, update_disk=True)

        assert is_alive is False

        # Verify .failed file was created
        failed_file = job_path / f"{scriptname}.failed"
        assert failed_file.exists(), ".failed file should be created for dead process"

        # Verify .failed file contains correct data
        failed_data = json.loads(failed_file.read_text())
        assert failed_data["exit_code"] == 137
        assert failed_data["failure_status"] == "UNKNOWN"

        # Verify .pid file was removed
        assert not pid_file.exists(), ".pid file should be removed for dead process"


def test_sync_job_state_detects_dead_process(tmp_path):
    """Test that scan_job_state_from_disk detects dead 'running' jobs"""
    import json
    from unittest.mock import MagicMock, patch

    from experimaestro.scheduler.state_sync import scan_job_state_from_disk

    # Create a job directory with only a PID file (no .done or .failed)
    job_path = tmp_path / "jobs" / "mymodule.CrashedTask" / "job_crashed_2"
    job_path.mkdir(parents=True)

    scriptname = "CrashedTask"
    pid_file = job_path / f"{scriptname}.pid"
    pid_file.write_text(json.dumps({"type": "local", "pid": 999998}))

    # Mock process as dead
    mock_process = MagicMock()
    mock_state = MagicMock()
    mock_state.finished = True
    mock_state.exitcode = 1

    async def mock_aio_state(timeout):
        return mock_state

    mock_process.aio_state = mock_aio_state

    with patch(
        "experimaestro.connectors.Process.fromDefinition", return_value=mock_process
    ):
        # Scan job state - should detect dead process
        job_state = scan_job_state_from_disk(job_path, scriptname, check_running=True)

        assert job_state is not None
        assert job_state["state"] == "error"
        assert job_state["failure_reason"] == "UNKNOWN"


def test_check_process_alive_with_lock_held(tmp_path):
    """Test that check_process_alive returns True when lock is held (job running)"""
    import json
    from unittest.mock import patch, MagicMock

    from experimaestro.scheduler.state_sync import check_process_alive

    # Create a job directory with a PID file
    job_path = tmp_path / "jobs" / "mymodule.RunningTask" / "job_locked"
    job_path.mkdir(parents=True)

    scriptname = "RunningTask"
    pid_file = job_path / f"{scriptname}.pid"
    pid_file.write_text(json.dumps({"type": "local", "pid": 12345}))

    # Mock the lock to simulate it being held by another process
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = False  # Simulate lock already held

    with patch(
        "experimaestro.scheduler.state_sync.fasteners.InterProcessLock",
        return_value=mock_lock,
    ):
        # check_process_alive should return True because it can't acquire the lock
        is_alive = check_process_alive(job_path, scriptname, update_disk=True)

        assert is_alive is True, "Should assume job is running when lock is held"

        # .pid file should still exist (not removed)
        assert pid_file.exists(), ".pid file should not be removed when lock is held"

        # No .failed file should be created
        failed_file = job_path / f"{scriptname}.failed"
        assert (
            not failed_file.exists()
        ), ".failed file should not be created when lock is held"
