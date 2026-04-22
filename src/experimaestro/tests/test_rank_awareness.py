import os
import pytest
from unittest.mock import patch
from experimaestro.run import TaskRunner
import experimaestro.taskglobals as taskglobals

@pytest.fixture
def env():
    instance = taskglobals.Env.instance()
    # Reset env
    instance.slave = False
    return instance

@patch("experimaestro.run.atexit.register")
@patch("experimaestro.run.signal.signal")
@patch("experimaestro.run.os.chdir")
@patch("experimaestro.run.os.register_at_fork")
@patch("experimaestro.run.create_file_lock")
@patch("experimaestro.run.start_of_job")
@patch("experimaestro.run.run")
@patch("experimaestro.run.TaskRunner._update_status_running")
@patch("experimaestro.run.TaskRunner._load_mock_job")
def test_task_runner_rank_detection(
    mock_load_mock_job,
    mock_update_status_running,
    mock_run_task,
    mock_start_of_job,
    mock_create_file_lock,
    mock_register_at_fork,
    mock_os_chdir,
    mock_signal,
    mock_atexit,
    env,
    tmp_path
):
    script_path = tmp_path / "test.py"
    script_path.touch()
    lockfiles = [str(tmp_path / "test.lock")]

    # Case 1: Main process (rank 0)
    with patch.dict(os.environ, {"SLURM_PROCID": "0", "LOCAL_RANK": "0"}):
        runner = TaskRunner(str(script_path), lockfiles)
        # We need to stop run() from exiting or failing
        # We'll mock its internal exit/cleanup calls if necessary,
        # or just catch the SystemExit
        try:
            runner.run()
        except SystemExit:
            pass

        assert env.slave is False
        mock_create_file_lock.assert_called()
        mock_start_of_job.assert_called()
        mock_update_status_running.assert_called()

    # Reset mocks for Case 2
    mock_create_file_lock.reset_mock()
    mock_start_of_job.reset_mock()
    mock_update_status_running.reset_mock()
    env.slave = False

    # Case 2: Slave process (rank > 0 via LOCAL_RANK)
    with patch.dict(os.environ, {"SLURM_PROCID": "0", "LOCAL_RANK": "1"}):
        runner = TaskRunner(str(script_path), lockfiles)
        try:
            runner.run()
        except SystemExit:
            pass

        assert env.slave is True
        mock_create_file_lock.assert_not_called()
        mock_start_of_job.assert_not_called()
        mock_update_status_running.assert_not_called()

    # Case 3: Slave process (rank > 0 via SLURM_PROCID)
    env.slave = False
    with patch.dict(os.environ, {"SLURM_PROCID": "2"}):
        runner = TaskRunner(str(script_path), lockfiles)
        try:
            runner.run()
        except SystemExit:
            pass

        assert env.slave is True
        mock_create_file_lock.assert_not_called()
