"""Tests for the experiment copy functionality."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from experimaestro.copy_experiment import (
    copy_experiment,
    list_experiments,
    list_runs,
    path_exists,
    read_jobs_jsonl,
    resolve_current_run,
    rsync_path,
    run_rsync,
    ssh_args,
)
from experimaestro.settings import SSHSettings, WorkspaceSettings


# --- Fixtures ---


def _make_local_ws(tmp_path: Path, ws_id: str = "local") -> WorkspaceSettings:
    """Create a local WorkspaceSettings pointing at tmp_path."""
    ws_path = tmp_path / ws_id
    ws_path.mkdir(parents=True, exist_ok=True)
    return WorkspaceSettings(id=ws_id, path=ws_path)


def _make_remote_ws(
    ws_id: str = "remote",
    host: str = "user@remotehost",
    raw_path: str = "/data/experiments",
    options: list[str] | None = None,
) -> WorkspaceSettings:
    """Create a remote WorkspaceSettings (no real filesystem)."""
    ssh = SSHSettings(host=host, options=options or [])
    ws = WorkspaceSettings(id=ws_id, path=Path(raw_path), ssh=ssh)
    return ws


def _write_jobs_jsonl(ws_path: Path, experiment_id: str, run_id: str, jobs: list[dict]):
    """Write a jobs.jsonl file in the workspace."""
    run_dir = ws_path / "experiments" / experiment_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Also create a status.json placeholder
    (run_dir / "status.json").write_text("{}")
    with (run_dir / "jobs.jsonl").open("w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    return run_dir


def _create_job_dir(
    ws_path: Path, task_id: str, job_id: str, workspace_path: str | None = None
):
    """Create a job directory with a placeholder output file and params.json."""
    job_dir = ws_path / "jobs" / task_id / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output.txt").write_text("result")
    if workspace_path is not None:
        params = {
            "workspace": workspace_path,
            "version": 2,
            "objects": [],
        }
        (job_dir / "params.json").write_text(json.dumps(params))
    return job_dir


# --- Unit tests for helpers ---


class TestRsyncPath:
    def test_local_path(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        result = rsync_path(ws, "jobs/task1/abc123")
        assert result == str(ws.path / "jobs/task1/abc123")

    def test_remote_path(self):
        ws = _make_remote_ws()
        result = rsync_path(ws, "jobs/task1/abc123")
        assert result == "user@remotehost:/data/experiments/jobs/task1/abc123"

    def test_remote_path_trailing_slash(self):
        ws = _make_remote_ws(raw_path="/data/experiments/")
        result = rsync_path(ws, "jobs/task1/abc123")
        assert result == "user@remotehost:/data/experiments/jobs/task1/abc123"


class TestSshArgs:
    def test_no_options(self):
        ws = _make_remote_ws(options=[])
        assert ssh_args(ws) == []

    def test_with_options(self):
        ws = _make_remote_ws(options=["-p", "2222"])
        assert ssh_args(ws) == ["-e", "ssh -p 2222"]

    def test_local_workspace(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        assert ssh_args(ws) == []


class TestRunRsync:
    def test_remote_to_remote_raises(self):
        ws1 = _make_remote_ws(ws_id="r1", host="host1")
        ws2 = _make_remote_ws(ws_id="r2", host="host2")
        with pytest.raises(RuntimeError, match="Remote-to-remote"):
            run_rsync(ws1, ws2, "a", "b")


class TestReadJobsJsonl:
    def test_local(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        jobs_data = [
            {"job_id": "j1", "task_id": "t1", "tags": {}, "timestamp": 1.0},
            {"job_id": "j2", "task_id": "t2", "tags": {"k": "v"}, "timestamp": 2.0},
        ]
        _write_jobs_jsonl(ws.path, "exp1", "run1", jobs_data)

        result = read_jobs_jsonl(ws, "exp1", "run1")
        assert len(result) == 2
        assert result[0].job_id == "j1"
        assert result[0].task_id == "t1"
        assert result[1].tags == {"k": "v"}


class TestPathExists:
    def test_local_exists(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        (ws.path / "jobs" / "t1" / "j1").mkdir(parents=True)
        assert path_exists(ws, "jobs/t1/j1") is True

    def test_local_not_exists(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        assert path_exists(ws, "jobs/t1/j1") is False


class TestResolveCurrentRun:
    def test_local_symlink(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        exp_dir = ws.path / "experiments" / "exp1"
        run_dir = exp_dir / "20240101-1200"
        run_dir.mkdir(parents=True)
        (exp_dir / "current").symlink_to(run_dir)

        result = resolve_current_run(ws, "exp1")
        assert result == "20240101-1200"

    def test_local_no_symlink(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        (ws.path / "experiments" / "exp1").mkdir(parents=True)
        assert resolve_current_run(ws, "exp1") is None


class TestListExperiments:
    def test_local(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        (ws.path / "experiments" / "alpha").mkdir(parents=True)
        (ws.path / "experiments" / "beta").mkdir(parents=True)
        result = list_experiments(ws)
        assert result == ["alpha", "beta"]

    def test_local_empty(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        assert list_experiments(ws) == []


class TestListRuns:
    def test_local(self, tmp_path):
        ws = _make_local_ws(tmp_path)
        exp_dir = ws.path / "experiments" / "exp1"
        (exp_dir / "run-a").mkdir(parents=True)
        (exp_dir / "run-b").mkdir(parents=True)
        # "current" symlink should be excluded
        (exp_dir / "current").symlink_to(exp_dir / "run-b")
        result = list_runs(ws, "exp1")
        assert result == ["run-a", "run-b"]


# --- Local-to-local integration test ---


class TestLocalToLocalCopy:
    def test_copy_experiment_full(self, tmp_path):
        """End-to-end local copy: creates source workspace structure, copies, verifies."""
        src_ws = _make_local_ws(tmp_path, "src")
        dst_ws = _make_local_ws(tmp_path, "dst")

        src_ws_abs = str(src_ws.path.resolve())
        dst_ws_abs = str(dst_ws.path.resolve())

        # Create experiment marker
        (src_ws.path / ".__experimaestro__").touch()
        (dst_ws.path / ".__experimaestro__").touch()

        # Create jobs.jsonl with two jobs
        jobs_data = [
            {"job_id": "j1", "task_id": "t1", "tags": {}, "timestamp": 1.0},
            {
                "job_id": "j2",
                "task_id": "t2",
                "tags": {"env": "test"},
                "timestamp": 2.0,
            },
        ]
        run_dir = _write_jobs_jsonl(src_ws.path, "myexp", "run001", jobs_data)

        # Create additional experiment files
        (run_dir / "objects.jsonl").write_text('{"type": "config"}\n')
        (run_dir / "events.jsonl").write_text('{"event": "start"}\n')

        # Create job directories with params.json referencing source workspace
        _create_job_dir(src_ws.path, "t1", "j1", workspace_path=src_ws_abs)
        _create_job_dir(src_ws.path, "t2", "j2", workspace_path=src_ws_abs)

        # Run copy
        result = copy_experiment(src_ws, dst_ws, "myexp", "run001")

        # Verify result
        assert result.jobs_copied == 2
        assert result.jobs_skipped == 0
        assert result.errors == []

        # Verify experiment run dir was copied
        dst_run = dst_ws.path / "experiments" / "myexp" / "run001"
        assert dst_run.is_dir()
        assert (dst_run / "jobs.jsonl").is_file()
        assert (dst_run / "status.json").is_file()
        assert (dst_run / "objects.jsonl").is_file()

        # Verify job directories were copied
        assert (dst_ws.path / "jobs" / "t1" / "j1" / "output.txt").is_file()
        assert (dst_ws.path / "jobs" / "t2" / "j2" / "output.txt").is_file()

        # Verify params.json workspace path was rewritten
        for task_id, job_id in [("t1", "j1"), ("t2", "j2")]:
            params_path = dst_ws.path / "jobs" / task_id / job_id / "params.json"
            params = json.loads(params_path.read_text())
            assert params["workspace"] == dst_ws_abs, (
                f"params.json workspace not rewritten: {params['workspace']}"
            )

    def test_copy_skips_existing_jobs(self, tmp_path):
        """Jobs that already exist at destination are skipped."""
        src_ws = _make_local_ws(tmp_path, "src")
        dst_ws = _make_local_ws(tmp_path, "dst")

        jobs_data = [
            {"job_id": "j1", "task_id": "t1", "tags": {}, "timestamp": 1.0},
            {"job_id": "j2", "task_id": "t2", "tags": {}, "timestamp": 2.0},
        ]
        _write_jobs_jsonl(src_ws.path, "myexp", "run001", jobs_data)
        _create_job_dir(src_ws.path, "t1", "j1")
        _create_job_dir(src_ws.path, "t2", "j2")

        # Pre-create j1 at destination
        _create_job_dir(dst_ws.path, "t1", "j1")

        result = copy_experiment(src_ws, dst_ws, "myexp", "run001")

        assert result.jobs_copied == 1
        assert result.jobs_skipped == 1
        assert result.errors == []


# --- SSH remote copy test (mocked subprocess) ---


class TestRemoteCopy:
    def test_remote_to_local_rsync_commands(self, tmp_path):
        """Verify correct rsync/ssh commands for remote-to-local copy."""
        src_ws = _make_remote_ws(
            ws_id="remote-src",
            host="user@cluster",
            raw_path="/data/xp",
            options=["-p", "2222"],
        )
        dst_ws = _make_local_ws(tmp_path, "local-dst")

        jobs_jsonl_content = (
            '{"job_id": "j1", "task_id": "t1", "tags": {}, "timestamp": 1.0}\n'
            '{"job_id": "j2", "task_id": "t2", "tags": {}, "timestamp": 2.0}\n'
        )

        rsync_calls = []

        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()

            if cmd[0] == "ssh":
                remote_cmd = cmd[-1]
                if remote_cmd.startswith("cat "):
                    mock_result.returncode = 0
                    mock_result.stdout = jobs_jsonl_content
                elif remote_cmd.startswith("test "):
                    mock_result.returncode = 1  # not exists
                elif remote_cmd.startswith("realpath "):
                    mock_result.returncode = 0
                    mock_result.stdout = "/data/xp"
                else:
                    mock_result.returncode = 0
                    mock_result.stdout = ""
            elif cmd[0] == "rsync":
                rsync_calls.append(cmd)
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_result.stderr = ""
                # For local dst: create the staging target dirs so move works
                if kwargs.get("check", False) or True:
                    # Find the local destination path (last arg without remote prefix)
                    dst_arg = cmd[-1]
                    if not dst_arg.startswith("user@"):
                        dst_path = Path(dst_arg.rstrip("/"))
                        dst_path.mkdir(parents=True, exist_ok=True)
            else:
                mock_result.returncode = 0
                mock_result.stdout = ""

            return mock_result

        # Pre-create local dst job dir for j2 to test skipping
        _create_job_dir(dst_ws.path, "t2", "j2")

        with patch(
            "experimaestro.copy_experiment.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = copy_experiment(src_ws, dst_ws, "exp1", "run-abc")

        assert result.jobs_copied == 1
        assert result.jobs_skipped == 1
        assert result.errors == []

        # Verify rsync calls
        assert len(rsync_calls) == 2  # 1 experiment dir + 1 job dir (j1)

        # First call: experiment run dir
        exp_rsync = rsync_calls[0]
        assert "-z" in exp_rsync  # compression for remote
        assert "-e" in exp_rsync
        ssh_idx = exp_rsync.index("-e")
        assert "ssh -p 2222" in exp_rsync[ssh_idx + 1]
        assert "--exclude=jobs" in exp_rsync
        # Source should have remote prefix
        assert any(
            "user@cluster:/data/xp/experiments/exp1/run-abc/" in arg
            for arg in exp_rsync
        )

        # Second call: job j1
        job_rsync = rsync_calls[1]
        assert any("user@cluster:/data/xp/jobs/t1/j1/" in arg for arg in job_rsync)

    def test_local_to_remote_rsync_commands(self, tmp_path):
        """Verify correct rsync commands for local-to-remote copy."""
        src_ws = _make_local_ws(tmp_path, "local-src")
        dst_ws = _make_remote_ws(
            ws_id="remote-dst",
            host="user@cluster",
            raw_path="/data/xp",
        )

        # Create source experiment structure
        jobs_data = [
            {"job_id": "j1", "task_id": "t1", "tags": {}, "timestamp": 1.0},
        ]
        _write_jobs_jsonl(src_ws.path, "exp1", "run-abc", jobs_data)
        _create_job_dir(
            src_ws.path, "t1", "j1", workspace_path=str(src_ws.path.resolve())
        )

        rsync_calls = []

        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()

            if cmd[0] == "ssh":
                remote_cmd = cmd[-1]
                if remote_cmd.startswith("test "):
                    mock_result.returncode = 1  # not exists
                elif remote_cmd.startswith("realpath "):
                    mock_result.returncode = 0
                    mock_result.stdout = "/data/xp"
                elif remote_cmd.startswith("mkdir "):
                    mock_result.returncode = 0
                    mock_result.stdout = ""
                elif remote_cmd.startswith("cat "):
                    # For rewriting params.json on remote
                    mock_result.returncode = 0
                    mock_result.stdout = json.dumps(
                        {
                            "workspace": str(src_ws.path.resolve()),
                            "version": 2,
                            "objects": [],
                        }
                    )
                elif remote_cmd.startswith("printf "):
                    mock_result.returncode = 0
                    mock_result.stdout = ""
                elif "mv " in remote_cmd:
                    mock_result.returncode = 0
                    mock_result.stdout = ""
                elif remote_cmd.startswith("rm "):
                    mock_result.returncode = 0
                    mock_result.stdout = ""
                else:
                    mock_result.returncode = 0
                    mock_result.stdout = ""
            elif cmd[0] == "rsync":
                rsync_calls.append(cmd)
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_result.stderr = ""
            else:
                mock_result.returncode = 0
                mock_result.stdout = ""

            return mock_result

        with patch(
            "experimaestro.copy_experiment.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            result = copy_experiment(src_ws, dst_ws, "exp1", "run-abc")

        assert result.jobs_copied == 1
        assert result.errors == []

        # Verify rsync destination paths go through staging with remote prefix
        for call in rsync_calls:
            dst_arg = call[-1]  # last arg is destination
            assert dst_arg.startswith("user@cluster:/data/xp/")
