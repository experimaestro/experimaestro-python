from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from experimaestro.utils.yaml import YAMLDataClass
from .specs import HostRequirement

if TYPE_CHECKING:
    from experimaestro.launchers import Launcher
    from experimaestro.connectors import Connector
    from experimaestro.tokens import Token
    from .registry import LauncherRegistry


class LauncherConfiguration:
    tags: List[str]
    weight: int

    """Generic class for a launcher configuration"""

    def get(
        self, registry: "LauncherRegistry", requirement: HostRequirement
    ) -> Optional["Launcher"]:
        raise NotImplementedError(f"For {self.__class__}")


class ConnectorConfiguration:
    def create(self, registry: "LauncherRegistry") -> "Connector":
        raise NotImplementedError(f"For {self.__class__}")


class TokenConfiguration(YAMLDataClass):
    def create(self, registry: "LauncherRegistry", identifier: str) -> "Token":
        raise NotImplementedError(f"For {self.__class__}")
