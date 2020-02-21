from pathlib import Path
from typing import Union
from .scheduler import Job


class PathGenerator:
    def __init__(self, path: Union[str, Path]):
        self.path = path

    def __call__(self, jobcontext: Job):
        raise NotImplementedError()
