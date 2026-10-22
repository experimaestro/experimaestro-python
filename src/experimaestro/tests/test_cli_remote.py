"""Command line on a remote workspace

The remote host is the local one: a fake ``ssh`` (first on the ``PATH``) runs
the remote command locally, so that both the remote state provider and the
commands run remotely are exercised.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from experimaestro import Param, Task
from experimaestro.cli.remote import RemoteWorkspace, SSHOptions
from experimaestro.settings import SSHSettings, WorkspaceSettings
from experimaestro.tests.utils import TemporaryDirectory, TemporaryExperiment

pytestmark = pytest.mark.tasks

FAKE_SSH = """#!/bin/sh
# Skips the SSH options and the host, then runs the command locally
while [ $# -gt 0 ]; do
  case "$1" in
    -o|-O|-p|-i|-l|-F) shift 2;;
    -*) shift;;
    *) break;;
  esac
done
shift
[ $# -eq 0 ] && exit 0
exec /bin/sh -c "$*"
"""


class RemoteCliTask(Task):
    x: Param[int]

    def execute(self):
        pass


@pytest.fixture(scope="module")
def workspace():
    with TemporaryDirectory(prefix="xpm-remote-cli") as workdir:
        with TemporaryExperiment("remote-xp", workdir=workdir):
            RemoteCliTask.C(x=1).submit()
        yield workdir


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(Path(__file__).parents[2]),
    }


def xpm(env, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "experimaestro", *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def remote(workspace):
    return [
        "--ssh",
        f"localhost:{workspace}",
        "--remote-xpm",
        f"{sys.executable} -m experimaestro",
    ]


@pytest.mark.parametrize(
    "command,expected",
    [
        # Through the remote state provider
        (["jobs", "list"], "remoteclitask/"),
        (["experiments", "list"], "remote-xp"),
        (["experiments", "services", "list", "remote-xp"], "No services found"),
        (["experiments", "dag", "remote-xp", "--json", "-"], "remoteclitask"),
        # Run on the remote host
        (["experiments", "runs", "list", "remote-xp"], "remote-xp"),
        (["workspace", "orphans"], "No orphan jobs found"),
    ],
)
def test_remote_command_output_matches_local(workspace, env, command, expected):
    local = xpm(env, "--workdir", workspace, *command)
    assert local.returncode == 0, local.stderr
    assert expected in local.stdout

    result = xpm(env, *remote(workspace), *command)
    assert result.returncode == 0, result.stderr
    assert result.stdout == local.stdout


def test_remote_command_exit_code(workspace, env):
    result = xpm(env, *remote(workspace), "experiments", "runs", "list", "unknown")
    assert result.returncode != 0


def test_run_refuses_remote_workspace(workspace, env, tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text("id: test\nfile: experiment\n")
    result = xpm(env, *remote(workspace), "run", params)
    assert result.returncode != 0
    assert "remote workspace" in result.stderr


def test_ssh_and_workdir_are_exclusive(workspace, env):
    result = xpm(env, *remote(workspace), "--workdir", workspace, "jobs", "list")
    assert result.returncode != 0
    assert "cannot be used together" in result.stderr


def test_remote_workspace_options_override_settings():
    settings = WorkspaceSettings(
        "cluster",
        Path("/remote/ws"),
        ssh=SSHSettings(
            host="cluster", shell_init="module load python", options=["-p", "22"]
        ),
    )
    options = SSHOptions()
    options.update(remote_xpm="/opt/xpm", ssh_option=("-p", "2222"))
    ws = RemoteWorkspace.resolve(options, settings)

    assert (ws.host, ws.path) == ("cluster", "/remote/ws")
    assert ws.command(["jobs", "list"], tty=True) == [
        "ssh",
        "-t",
        "-p",
        "2222",
        "cluster",
        "module load python; /opt/xpm --workdir /remote/ws jobs list",
    ]


def test_remote_workspace_connect_timeout():
    """The connection timeout comes from the option, then the settings"""
    settings = WorkspaceSettings(
        "cluster",
        Path("/remote/ws"),
        ssh=SSHSettings(host="cluster", connect_timeout=120.0),
    )
    assert RemoteWorkspace.resolve(SSHOptions(), settings).connect_timeout == 120.0

    options = SSHOptions()
    options.update(connect_timeout=90.0)
    assert RemoteWorkspace.resolve(options, settings).connect_timeout == 90.0

    options = SSHOptions()
    options.set_target("cluster:/remote/ws")
    assert RemoteWorkspace.resolve(options, None).connect_timeout == 60.0
