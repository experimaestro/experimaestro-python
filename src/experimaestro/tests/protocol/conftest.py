"""Harness for the filesystem protocol tests

Experiments run in their own processes (``scenario.py``) so that they can be
killed or paused at named crash points (``utils/crashpoints.py``) without
affecting the test process, which plays the other actors (monitor, cleanup,
CLI, another rank, …).
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from experimaestro import Config
from experimaestro.locking import create_file_lock
from experimaestro.tests.utils import get_test_timeout
from experimaestro.utils.crashpoints import ENV_CRASHPOINTS, ENV_CRASHPOINTS_DIR

from .tasks import ProtocolTask

import filelock

XP_NAME = "protocol"


def pytest_collection_modifyitems(items):
    for item in items:
        if "/protocol/" in str(item.fspath):
            item.add_marker(pytest.mark.protocol)


def wait_for(predicate: Callable[[], bool], what: str, multiplier: float = 5.0):
    deadline = time.time() + get_test_timeout(multiplier)
    while not predicate():
        if time.time() > deadline:
            raise TimeoutError(f"Timeout while waiting for {what}")
        time.sleep(0.02)


@dataclass
class JobFiles:
    """Paths of a job, computed from its configuration"""

    path: Path
    name: str

    @staticmethod
    def of(workdir: Path, config: Config) -> "JobFiles":
        task_id = str(config.__xpmtype__.identifier)
        return JobFiles(
            workdir / "jobs" / task_id / config.__xpm__.identifier.all.hex(),
            task_id.rsplit(".", 1)[-1],
        )

    def __getattr__(self, suffix: str) -> Path:
        # done, failed, pid, lock -> <name>.<suffix>
        if suffix in ("done", "failed", "pid", "lock", "py"):
            return self.path / f"{self.name}.{suffix}"
        raise AttributeError(suffix)

    @property
    def runs(self) -> int:
        path = self.path / "runs"
        return len(path.read_text().splitlines()) if path.exists() else 0

    def touch_path(self) -> Path:
        return self.path / "touch"

    def release(self):
        """Lets a ``Controllable`` task finish"""
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "wait").write_text("go")

    def lock_is_free(self) -> bool:
        """Whether nobody holds the job lock (checked from this process)"""
        lock = create_file_lock(self.lock, timeout=0)
        try:
            lock.acquire()
        except filelock.Timeout:
            return False
        lock.release()
        return True

    def process_pid(self) -> int:
        return json.loads(self.pid.read_text())["pid"]


class Actors:
    def __init__(self, root: Path):
        self.workdir = root / "workspace"
        self.workdir.mkdir()
        self.crashdir = root / "crashpoints"
        self.crashdir.mkdir()
        self.processes: list[subprocess.Popen] = []
        #: Job processes to kill at the end (their pid file may be deleted)
        self.job_pids: set[int] = set()

    def experiment(
        self,
        *tasks: ProtocolTask,
        crashpoints: str = "",
        token: dict | None = None,
        name: str = XP_NAME,
    ) -> subprocess.Popen:
        """Starts an experiment process submitting ``tasks``"""
        spec: dict[str, Any] = {
            "tasks": [
                {
                    "cls": task.__xpmtype__.value_type.__name__,
                    "params": {"x": task.x},
                }
                for task in tasks
            ],
        }
        if token is not None:
            spec["token"] = token
        env = {
            **os.environ,
            ENV_CRASHPOINTS: crashpoints,
            ENV_CRASHPOINTS_DIR: str(self.crashdir),
        }
        log = (self.crashdir / f"xp-{len(self.processes)}.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experimaestro.tests.protocol.scenario",
                str(self.workdir),
                name,
                json.dumps(spec),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.processes.append(process)
        return process

    def logs(self) -> str:
        return "\n".join(
            f"--- {path.name}\n{path.read_text()[-4000:]}"
            for path in sorted(self.crashdir.glob("xp-*.log"))
        )

    def wait_reached(self, point: str, process: subprocess.Popen | None = None):
        """Waits until a process is paused at ``point``"""
        path = self.crashdir / f"{point}.reached"

        def reached():
            if process is not None and process.poll() is not None:
                if not path.exists():
                    raise AssertionError(
                        f"Process ended ({process.returncode}) before reaching "
                        f"{point}\n{self.logs()}"
                    )
            return path.exists()

        wait_for(reached, f"crash point {point}")

    def resume(self, point: str):
        (self.crashdir / f"{point}.resume").write_text("")

    def wait_exit(self, process: subprocess.Popen, multiplier: float = 10.0) -> int:
        try:
            return process.wait(get_test_timeout(multiplier))
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Experiment did not finish\n{self.logs()}")

    def teardown(self):
        # Unpause everything, let controllable tasks end, then kill leftovers
        for reached in self.crashdir.glob("*.reached"):
            self.resume(reached.name.removesuffix(".reached"))
        for touch in self.workdir.glob("jobs/*/*/touch"):
            (touch.parent / "wait").write_text("go")
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for pidfile in self.workdir.glob("jobs/*/*/*.pid"):
            try:
                self.job_pids.add(json.loads(pidfile.read_text())["pid"])
            except (OSError, ValueError, KeyError):
                pass
        for pid in self.job_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture
def actors(tmp_path):
    actors = Actors(tmp_path)
    try:
        yield actors
    finally:
        actors.teardown()
