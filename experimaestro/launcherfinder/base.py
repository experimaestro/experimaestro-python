from typing import TYPE_CHECKING, Optional
from .specs import HostRequirement

if TYPE_CHECKING:
    from experimaestro.launchers import Launcher
    from experimaestro.connectors import Connector
    from experimaestro.tokens import Token
    from .registry import LauncherRegistry


class LauncherConfiguration:
    """Generic class for a launcher configuration"""

    def get(
        self, registry: "LauncherRegistry", requirement: HostRequirement
    ) -> Optional["Launcher"]:
        raise NotImplementedError(f"For {self.__class__}")


class ConnectorConfiguration:
    def create(self, registry: "LauncherRegistry") -> "Connector":
        raise NotImplementedError(f"For {self.__class__}")


class TokenConfiguration:
    def create(self, registry: "LauncherRegistry") -> "Token":
        raise NotImplementedError(f"For {self.__class__}")
