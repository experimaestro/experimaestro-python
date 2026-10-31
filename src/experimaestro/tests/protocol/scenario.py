"""Runs an experiment in its own process, for the filesystem protocol tests

``python -m experimaestro.tests.protocol.scenario WORKDIR NAME SPEC``

where ``SPEC`` is a JSON object:

- ``tasks``: list of ``{"cls": <name in tasks.TASKS>, "params": {...}}``;
- ``token``: optional ``{"name", "path", "count"}`` — every task then takes
  one token.

The crash points (``XPM_CRASHPOINTS``, ``XPM_CRASHPOINTS_DIR``) apply to this
process and are passed on to the job processes.
"""

import json
import os
import sys
from pathlib import Path

from experimaestro import experiment
from experimaestro.tests.protocol.tasks import TASKS
from experimaestro.tokens import CounterToken
from experimaestro.utils.crashpoints import ENV_CRASHPOINTS, ENV_CRASHPOINTS_DIR


def main(workdir: str, name: str, spec: dict):
    with experiment(workdir, name, no_environmental_impact=True) as xp:
        launcher = xp.workspace.launcher
        launcher.setenv("PYTHONPATH", str(Path(__file__).parents[3]))
        for var in (ENV_CRASHPOINTS, ENV_CRASHPOINTS_DIR):
            if var in os.environ:
                launcher.setenv(var, os.environ[var])

        token = None
        if token_spec := spec.get("token"):
            token = CounterToken(
                token_spec["name"], Path(token_spec["path"]), token_spec["count"]
            )

        for task_spec in spec.get("tasks", []):
            task = TASKS[task_spec["cls"]].C(**task_spec.get("params", {}))
            if token is not None:
                task.add_dependencies(token.dependency(1))
            task.submit()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
