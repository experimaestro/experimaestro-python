"""Python environments chosen on the node that runs a job (opt-in)

On a cluster with heterogeneous nodes, a single Python environment may not run
everywhere: e.g. a PyTorch build only has kernels for some GPU generations, and
an x86 environment does not run on ARM nodes. The environment that launches the
experiment can then list other environments in a ``experimaestro-envs.json``
file at its root (``sys.prefix``)::

    {"environments": [
        {"python": "/path/to/.venv-cu126/bin/python",
         "when": "[ \\"$(uname -m)\\" = aarch64 ]"}
    ]}

In the job script (SLURM launcher), each ``when`` shell test is run on the node,
in order: the job runs with the Python interpreter of the first one that
succeeds, and otherwise with the current one. Without the file, job scripts do
not change.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shlex import quote as shquote

ENVIRONMENTS_FILE = "experimaestro-envs.json"

PYTHON_VARIABLE = "XPM_PYTHON"
"""Shell variable holding the selected Python interpreter in job scripts"""


@dataclass
class PythonEnvironment:
    """A Python environment, used when its shell test succeeds on the node"""

    python: str
    """Path to the Python interpreter"""

    when: str
    """Shell command (e.g. a test on the GPU or the CPU architecture) whose
    success selects this environment"""


def python_environments(prefix: Path | None = None) -> list[PythonEnvironment]:
    """The environments listed by an environment (empty if there is no file)

    :param prefix: root of the environment (default: the current one,
        ``sys.prefix``)
    """
    path = Path(sys.prefix if prefix is None else prefix) / ENVIRONMENTS_FILE
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())["environments"]
        return [PythonEnvironment(e["python"], e["when"]) for e in entries]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"Invalid environment file {path}: {e!r}") from e


def shell_selection(default: str, environments: list[PythonEnvironment]) -> str:
    """Shell code setting :data:`PYTHON_VARIABLE` to the Python interpreter of
    the first environment whose test succeeds (``default`` otherwise), and
    reporting it on the standard error"""
    lines = [f"{PYTHON_VARIABLE}={shquote(default)}"]
    for i, env in enumerate(environments):
        # Tests run in a subshell, with no output
        keyword = "if" if i == 0 else "elif"
        lines.append(f"{keyword} ( {env.when}\n) >/dev/null 2>&1; then")
        lines.append(f"    {PYTHON_VARIABLE}={shquote(env.python)}")
    if environments:
        lines.append("fi")
    lines.append(f'echo "experimaestro: Python interpreter ${PYTHON_VARIABLE}" >&2')
    return "\n".join(lines) + "\n"


def succeeds(environment: PythonEnvironment) -> bool:
    """Runs the shell test of an environment, as in job scripts"""
    command = f"( {environment.when}\n) >/dev/null 2>&1"
    return subprocess.run(["sh", "-c", command]).returncode == 0
