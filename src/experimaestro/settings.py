import os
from omegaconf import OmegaConf, SCMode
from dataclasses import field, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, List
import logging
import fnmatch


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
class HistorySettings:
    """Settings for experiment history cleanup.

    When an experiment ends, old runs are cleaned up according to these rules
    (applied in order):

    1. If current run succeeded, all past failed runs are removed
    2. Failed runs that occurred before the newest successful run are removed
       (since the success supersedes the earlier failures)
    3. Keep at most `max_done` successful runs (oldest removed first)
    4. Keep at most `max_failed` failed runs (oldest removed first)

    Example: With max_done=2, max_failed=1 and runs:
        - 10:00 completed
        - 11:00 failed
        - 12:00 completed
        - 13:00 failed

    Result: 11:00 failed is removed (before 12:00 success), 10:00 completed
    is removed (max_done=2), leaving: 12:00 completed, 13:00 failed.
    """

    max_done: int = 5
    """Maximum number of successful runs to keep per experiment"""

    max_failed: int = 1
    """Maximum number of failed runs to keep per experiment"""


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

    max_retries: int = 3
    """Maximum number of retries for resumable tasks that timeout (default: 3)"""

    triggers: List[str] = field(default_factory=list)
    """Glob patterns to automatically select this workspace based on experiment ID"""

    history: HistorySettings = field(default_factory=HistorySettings)
    """Settings for experiment history cleanup"""

    def __post_init__(self):
        self.path = self.path.expanduser().resolve()


@dataclass
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    workspaces: List[WorkspaceSettings] = field(default_factory=list)

    env: Dict[str, str] = field(default_factory=dict)
    """Default environment variables"""

    history: HistorySettings = field(default_factory=HistorySettings)
    """Default history settings (can be overridden per workspace)"""


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


def find_workspace(
    *,
    workspace: Optional[str] = None,
    workdir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
) -> WorkspaceSettings:
    """Find workspace

    Args:
        workspace: Explicit workspace ID to use
        workdir: Explicit working directory path
        experiment_id: Experiment ID to match against workspace triggers

    Returns:
        WorkspaceSettings object
    """
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
        # Try to match experiment_id against workspace triggers
        matched_workspace = None
        if experiment_id:
            workspaces = get_settings().workspaces
            for ws in workspaces:
                for trigger in ws.triggers:
                    if fnmatch.fnmatch(experiment_id, trigger):
                        matched_workspace = ws
                        logging.info(
                            "Auto-selected workspace %s (matched trigger '%s')",
                            ws.id,
                            trigger,
                        )
                        break
                if matched_workspace:
                    break

        if matched_workspace:
            ws_env = matched_workspace
        else:
            ws_env = get_workspace()
            assert ws_env is not None, "No workdir or workspace defined, and no default"
            logging.info("Using default workspace %s", ws_env.id)

    return ws_env
