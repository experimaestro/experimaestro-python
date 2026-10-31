"""Tasks used by the filesystem protocol tests

Every task appends a line to ``runs`` (in its job directory) when it starts,
so that a test can count how many times a job has been executed.
"""

import os
import sys
import time
from pathlib import Path

from experimaestro import Meta, Param, PathGenerator, Task, TaskCancelled, field


def _record_run(path: Path):
    with path.open("a") as out:
        out.write(f"{os.getpid()}\n")


class Quick(Task):
    """Ends immediately"""

    x: Param[int] = field(default=0, ignore_default=True)
    runs: Meta[Path] = field(default_factory=PathGenerator("runs"))

    def execute(self):
        _record_run(self.runs)


class Failing(Task):
    """Raises an exception"""

    x: Param[int] = field(default=0, ignore_default=True)
    runs: Meta[Path] = field(default_factory=PathGenerator("runs"))

    def execute(self):
        _record_run(self.runs)
        raise RuntimeError("failing on purpose")


class Controllable(Task):
    """Signals that it started (``touch``), then waits for ``wait``

    Processes of rank > 0 (``RANK``) end immediately.
    """

    x: Param[int] = field(default=0, ignore_default=True)
    runs: Meta[Path] = field(default_factory=PathGenerator("runs"))
    touch: Meta[Path] = field(default_factory=PathGenerator("touch"))
    wait: Meta[Path] = field(default_factory=PathGenerator("wait"))

    def execute(self):
        if int(os.environ.get("RANK", "0")) > 0:
            return
        _record_run(self.runs)
        self.touch.write_text("started")
        while not self.wait.is_file():
            time.sleep(0.05)


class CatchCancel(Task):
    """Catches the cancellation and exits with code 0"""

    x: Param[int] = field(default=0, ignore_default=True)
    runs: Meta[Path] = field(default_factory=PathGenerator("runs"))
    touch: Meta[Path] = field(default_factory=PathGenerator("touch"))

    def execute(self):
        _record_run(self.runs)
        self.touch.write_text("started")
        try:
            while True:
                time.sleep(0.05)
        except TaskCancelled:
            sys.exit(0)


#: Any of the tasks above
ProtocolTask = Quick | Failing | Controllable | CatchCancel

TASKS: dict[str, type[Task]] = {
    cls.__name__: cls for cls in (Quick, Failing, Controllable, CatchCancel)
}
