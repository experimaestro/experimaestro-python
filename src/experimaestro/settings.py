import os
from omegaconf import OmegaConf, SCMode
from dataclasses import field, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, List
import logging


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
    """Defines the workspace"""

    id: str
    """The workspace identifier"""

    path: Path = field()
    """The workspace path"""

    env: Dict[str, str] = field(default_factory=dict)
    """Workspace specific environment variables"""

    alt_workspaces: List[str] = field(default_factory=list)
    """Alternative workspaces to find jobs or experiments"""
    
    def __post_init__(self):
        self.path = self.path.expanduser().resolve()


@dataclass
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    workspaces: List[WorkspaceSettings] = field(default_factory=list)

    env: Dict[str, str] = field(default_factory=dict)
    """Default environment variables"""


@lru_cache()
def get_settings(path: Optional[Path] = None) -> Settings:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return Settings()
    else:
        schema = OmegaConf.structured(Settings)

        path = path or Path("~/.config/experimaestro/settings.yaml").expanduser()
        if not path.is_file():
            return OmegaConf.to_container(
                schema, structured_config_mode=SCMode.INSTANTIATE
            )

        conf = OmegaConf.load(path)
        return OmegaConf.to_container(
            OmegaConf.merge(schema, conf), structured_config_mode=SCMode.INSTANTIATE
        )


def get_workspace(id: Optional[str] = None) -> Optional[WorkspaceSettings]:
    """Return the workspace settings given an id (or None for the default one)"""
    workspaces = get_settings().workspaces
    if workspaces:
        if id is None:
            return workspaces[0]
        for workspace in workspaces:
            if id == workspace.id:
                return workspace

    return None


def find_workspace(*, workspace: Optional[str] = None, workdir: Optional[Path] = None) -> WorkspaceSettings:
    """Find workspace"""
    workdir = Path(workdir) if workdir else None

    if workspace:
        ws_env = get_workspace(workspace)
        if ws_env is None:
            raise RuntimeError("No workspace named %s", workspace)

        logging.info("Using workspace %s", ws_env.id)
        if workdir:
            # Overrides working directory
            logging.info(" override working directory: %s", workdir)
            ws_env.path = workdir
    elif workdir:
        logging.info("Using workdir %s", workdir)
        ws_env = WorkspaceSettings("", workdir)
    else:
        ws_env = get_workspace()
        assert ws_env is not None, "No workdir or workspace defined, and no default"
        logging.info("Using default workspace %s", ws_env.id)

    return ws_env
