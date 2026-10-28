"""Command line layout: root workspace options and the old command names"""

import json

import pytest
from click.testing import CliRunner

import experimaestro.run
from experimaestro.cli import _is_task_parameters, cli


@pytest.fixture
def workspace(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "experiments").mkdir(parents=True)
    (workspace / ".__experimaestro__").touch()
    return workspace


def invoke(*args):
    return CliRunner().invoke(cli, [str(arg) for arg in args])


def test_hidden_aliases_not_in_help():
    result = invoke("--help")
    assert result.exit_code == 0, result.output
    for name in [
        "run",
        "monitor",
        "experiments",
        "jobs",
        "workspace",
        "tools",
        "inspect",
    ]:
        assert f"  {name} " in result.output
    for name in ["run-experiment", "orphans", "rpyc-server", "parameters-difference"]:
        assert name not in result.output


@pytest.mark.parametrize(
    "old,new",
    [
        (["run-experiment"], ["run"]),
        (["dag"], ["experiments", "dag"]),
        (["inspect"], ["jobs", "inspect"]),
        (["parameters-difference"], ["jobs", "diff"]),
        (["cleanup"], ["workspace", "cleanup"]),
        (["workspaces", "list"], ["workspace", "list"]),
        (["migrate", "v1-to-v2"], ["workspace", "migrate", "v1-to-v2"]),
        (["check-documentation"], ["tools", "check-documentation"]),
        (["find-launchers"], ["tools", "find-launchers"]),
        (["install-skill"], ["tools", "install-skill"]),
        (["refactor", "default-values"], ["tools", "refactor", "default-values"]),
        (["deprecated", "list"], ["tools", "deprecated", "list"]),
        (["experiments", "history"], ["experiments", "runs", "list"]),
        (["experiments", "history-prune"], ["experiments", "runs", "prune"]),
        (["experiments", "tag"], ["experiments", "runs", "tag"]),
        (["experiments", "monitor"], ["monitor"]),
        (["jobs", "stray"], ["workspace", "stray"]),
        (["xp", "list"], ["experiments", "list"]),
        (["ws", "list"], ["workspace", "list"]),
    ],
)
def test_old_names_have_the_same_options(old, new):
    old_help = invoke(*old, "--help")
    new_help = invoke(*new, "--help")
    assert old_help.exit_code == 0, old_help.output
    assert new_help.exit_code == 0, new_help.output
    options = lambda text: [line for line in text.splitlines() if "--" in line]  # noqa: E731
    assert options(old_help.output) == options(new_help.output)


def test_deprecated_alias_warns(workspace):
    result = invoke("migrate", "v1-to-v2", workspace, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "use `experimaestro workspace migrate`" in result.output


def test_old_orphans_takes_a_path(workspace):
    result = invoke("orphans", workspace)
    assert result.exit_code == 0, result.output
    assert "No orphan jobs found" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["workspace", "orphans"],
        ["jobs", "list"],
        ["experiments", "list"],
        ["workspace", "migrate", "v1-to-v2", "--dry-run"],
    ],
)
def test_root_workdir(workspace, args):
    result = invoke("--workdir", workspace, *args)
    assert result.exit_code == 0, result.output
    assert "not an experimaestro working directory" not in result.output


def test_group_workdir_still_accepted(workspace):
    """The remote monitor runs `experiments --workdir PATH monitor-server`"""
    result = invoke("experiments", "--workdir", workspace, "list")
    assert result.exit_code == 0, result.output
    result = invoke("experiments", "--workdir", workspace, "monitor-server", "--help")
    assert result.exit_code == 0, result.output
    result = invoke("jobs", "--workdir", workspace, "list")
    assert result.exit_code == 0, result.output


def test_bad_root_workdir(tmp_path):
    result = invoke("--workdir", tmp_path, "workspace", "orphans")
    assert result.exit_code != 0
    assert "not an experimaestro working directory" in result.output


def test_is_task_parameters(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"workspace": "/ws", "objects": []}))
    assert _is_task_parameters(str(params))

    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({"id": "my-xp", "file": "xp.py"}))
    assert not _is_task_parameters(str(experiment))

    assert not _is_task_parameters(str(tmp_path / "experiment.yaml"))
    assert not _is_task_parameters(str(tmp_path / "missing.json"))


def test_run_task_parameters(tmp_path, monkeypatch):
    """`experimaestro run PARAMS.json` still runs a single task"""
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"workspace": "/ws", "objects": []}))
    calls = []
    monkeypatch.setattr(experimaestro.run, "run", calls.append)

    result = invoke("run", params)
    assert result.exit_code == 0, result.output
    assert calls == [params]
    assert "Deprecated" in result.output


@pytest.mark.parametrize("flags,console", [(["--web"], False), (["--console"], True)])
def test_monitor_ui_flags(workspace, monkeypatch, flags, console):
    import sys

    monitor = sys.modules["experimaestro.cli.monitor"]
    calls = []
    monkeypatch.setattr(
        monitor,
        "_run_monitor_ui",
        lambda provider, workdir, ui, *a, **kw: calls.append(ui),
    )
    result = invoke("--workdir", workspace, "monitor", *flags)
    assert result.exit_code == 0, result.output
    assert calls == [console]


@pytest.mark.parametrize("flags", [[], ["--console", "--web"]])
def test_monitor_requires_one_ui(workspace, flags):
    """There is no default UI: exactly one of --console and --web"""
    result = invoke("--workdir", workspace, "monitor", *flags)
    assert result.exit_code == 2
    assert "Choose one of --console or --web" in result.output
