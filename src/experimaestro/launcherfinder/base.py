from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from experimaestro.connectors import Connector
    from experimaestro.tokens import Token
    from .registry import LauncherRegistry


class ConnectorConfiguration:
    def create(self, registry: "LauncherRegistry") -> "Connector":
        raise NotImplementedError(f"For {self.__class__}")


class TokenConfiguration:
    def create(self, registry: "LauncherRegistry", identifier: str) -> "Token":
        raise NotImplementedError(f"For {self.__class__}")
