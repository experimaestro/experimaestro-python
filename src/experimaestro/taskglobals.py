from pathlib import Path
from typing import Optional
import os
import logging


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

    @staticmethod
    def instance():
        if Env._instance is None:
            Env._instance = Env()
        return Env._instance
