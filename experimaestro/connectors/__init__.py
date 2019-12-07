"""Connectors module

This module contains :

- connectors
- process builders
- launchers

"""

from typing import Dict
from pathlib import Path, PosixPath

class ProcessBuilder: pass

class Connector(): 
    def processbuilder(self) -> ProcessBuilder:
        raise NotImplementedError()

