from functools import cached_property
from pathlib import Path
from typing import Optional


class Env:
    _instance = None

    # The working directory path
    wspath: Optional[Path] = None

    # The current task path
    taskpath: Optional[Path] = None

    # Set to True when multi-processing when
    # in slave mode:
    # - no progress report
    slave: bool = False

    @cached_property
    def xpm_path(self):
        path = self.taskpath / ".experimaestro"
        path.mkdir(exist_ok=True)
        return path

    @staticmethod
    def instance():
        if Env._instance is None:
            Env._instance = Env()
        return Env._instance
