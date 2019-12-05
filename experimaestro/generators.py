from pathlib import Path
from typing import Union

class PathGenerator():
    def __init__(self, path: Union[str, Path]):
        self.path = path
