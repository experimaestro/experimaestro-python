from pathlib import Path, PosixPath
from typing import Dict, Optional
from experimaestro.connectors.local import Connector, LocalConnector


class Launcher:
    def __init__(self, connector: "experimaestro.connectors.Connector"):
        self.connector = connector
        self.environ: Dict[str, str] = {}
        self.notificationURL: Optional[str] = None

    def setenv(self, key: str, value: str):
        self.environ[key] = value

    def setNotificationURL(self, url: str):
        self.notificationURL = url

    @staticmethod
    def get(path: Path):
        """Get a default launcher for a given path"""
        if isinstance(path, PosixPath):
            from .unix import UnixLauncher

            return UnixLauncher(LocalConnector())
        raise ValueError("Cannot create a default launcher for %s", type(path))
