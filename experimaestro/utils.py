from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger("xpm")


def aspath(path: Union[str, Path]):
    if isinstance(path, Path):
        return path
    return Path(path)
