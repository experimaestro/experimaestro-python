"""Python environments chosen on the node that runs a job"""

import json
import subprocess
from pathlib import Path

import pytest

from experimaestro.launchers.environments import (
    ENVIRONMENTS_FILE,
    PythonEnvironment,
    python_environments,
    shell_selection,
)
from experimaestro.launchers.slurm.base import srun_command


def selected(code: str, env: dict[str, str] | None = None) -> tuple[str, str]:
    """Runs shell code ending with the selection: (selected python, stderr)"""
    out = subprocess.run(
        ["sh", "-c", code + 'printf %s "$XPM_PYTHON"'],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return out.stdout, out.stderr


ENVIRONMENTS = [
    PythonEnvironment("/envs/arm/bin/python", '[ "$ARCH" = aarch64 ]'),
    # A pipeline, with some output (hidden)
    PythonEnvironment("/envs/old gpu/bin/python", "echo $GPU | grep -E '^(6|7\\.0)'"),
    PythonEnvironment("/envs/never/bin/python", "false"),
]


@pytest.mark.parametrize(
    "variables,python",
    [
        ({"ARCH": "aarch64", "GPU": "6.1"}, "/envs/arm/bin/python"),
        ({"ARCH": "x86_64", "GPU": "6.1"}, "/envs/old gpu/bin/python"),
        ({"ARCH": "x86_64", "GPU": "7.0"}, "/envs/old gpu/bin/python"),
        ({"ARCH": "x86_64", "GPU": "8.6"}, "/envs/default/bin/python"),
        ({"ARCH": "x86_64"}, "/envs/default/bin/python"),
    ],
)
def test_first_successful_test(variables, python):
    code = shell_selection("/envs/default/bin/python", ENVIRONMENTS)
    stdout, stderr = selected(code, {"PATH": "/usr/bin:/bin", **variables})
    # Only the selection is output (on stderr): tests are silent
    assert stdout == python
    assert stderr == f"experimaestro: Python interpreter {python}\n"


def test_no_environment_file(tmp_path: Path):
    assert python_environments(tmp_path) == []
    # The job script does not change
    assert srun_command("task.py", "/usr/bin/python", []) == "srun ./task.py\n"


def test_environment_file(tmp_path: Path):
    entries = [{"python": e.python, "when": e.when} for e in ENVIRONMENTS]
    (tmp_path / ENVIRONMENTS_FILE).write_text(json.dumps({"environments": entries}))
    assert python_environments(tmp_path) == ENVIRONMENTS

    command = srun_command("'my task.py'", "/envs/default/bin/python", ENVIRONMENTS)
    assert command.endswith("""srun "$XPM_PYTHON" ./'my task.py'\n""")
    stdout, _ = selected(
        command.rsplit("srun", 1)[0], {"PATH": "/usr/bin:/bin", "ARCH": "aarch64"}
    )
    assert stdout == "/envs/arm/bin/python"


def test_invalid_environment_file(tmp_path: Path):
    (tmp_path / ENVIRONMENTS_FILE).write_text('{"environments": [{"python": "x"}]}')
    with pytest.raises(ValueError, match=ENVIRONMENTS_FILE):
        python_environments(tmp_path)


def test_cli(tmp_path: Path):
    from click.testing import CliRunner

    from experimaestro.cli import cli

    def run(*args):
        result = CliRunner().invoke(
            cli, ["tools", "environments", "--prefix", str(tmp_path), *args]
        )
        assert result.exit_code == 0, result.output
        return result.output

    assert f"No {ENVIRONMENTS_FILE} in {tmp_path}" in run()

    entries = [
        {"python": "/envs/never/bin/python", "when": "false"},
        {"python": "/envs/yes/bin/python", "when": "echo hidden && true"},
        {"python": "/envs/also/bin/python", "when": "true"},
    ]
    (tmp_path / ENVIRONMENTS_FILE).write_text(json.dumps({"environments": entries}))
    output = run()
    assert "1. /envs/never/bin/python\n   when: false" in output
    assert "[test" not in output and "would use" not in output

    output = run("--check")
    assert "1. /envs/never/bin/python [test fails]" in output
    assert "2. /envs/yes/bin/python [test succeeds]" in output
    # The output of the tests is not shown
    assert "hidden" not in output.splitlines()
    assert "jobs would use /envs/yes/bin/python" in output
