import os
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, List


@dataclass
class ServerSettings:
    port: Optional[int] = None
    """Port for the server"""

    host: Optional[str] = None
    """Hostname for the server"""

    autohost: Optional[str] = None
    """Automatic hostname: values can be `fqdn` or `name`"""

    token: Optional[str] = None
    """Token for the server"""


@dataclass
class WorkspaceSettings:
    id: str
    """The workspace identifier"""

    path: Path
    """The workspace path"""


@dataclass
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    workspaces: List[WorkspaceSettings] = field(default_factory=list)


@lru_cache()
def get_settings(path: Optional[Path] = None) -> Settings:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return Settings()
    else:
        schema = OmegaConf.structured(Settings)

        path = path or Path("~/.config/experimaestro/settings.yaml").expanduser()
        if not path.is_file():
            return schema

        conf = OmegaConf.load(path)
        return OmegaConf.merge(schema, conf)


def get_workspace(id: Optional[str]) -> WorkspaceSettings:
    """Return the workspace settings given an id (or None for the default one)"""
    workspaces = get_settings().workspaces
    if workspaces:
        if id is None:
            return workspaces[0]
        for workspace in workspaces:
            if id == workspace.id:
                return workspace

    return None
