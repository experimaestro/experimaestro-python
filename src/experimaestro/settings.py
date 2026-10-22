__lazy_modules__ = ["yaml", "pydantic"]  # PEP 810 (Python 3.15+)

import sys
import yaml
import os
import re
from enum import Enum
from pydantic import TypeAdapter
from dataclasses import field, dataclass
from functools import lru_cache
from pathlib import Path
import logging
import fnmatch


class FolderMode(str, Enum):
    """Mode for an auxiliary workspace folder.

    See :class:`FolderSettings`. **Beta**: API may change.
    """

    USE = "use"
    """Read-only: look up jobs that may already exist in this folder.
    Never copy, move, or delete anything here."""

    BACKUP = "backup"
    """Copy successful jobs here after completion. Jobs remain in the
    primary workspace; the folder is used as a recovery source."""

    MOVE = "move"
    """Move successful jobs here after completion, then leave a symlink
    in the primary workspace. Use case: fast-read, slow-write tiers."""


@dataclass
class FolderSettings:
    """Auxiliary workspace folder. **Beta**: API may change.

    Allows attaching extra job-storage locations to a workspace, for
    archival, read-through alternative workspaces, or migration.
    """

    path: Path
    """Folder path (the jobs are stored under ``<path>/jobs/<task_id>/<hash>/``)"""

    mode: FolderMode = FolderMode.USE
    """How this folder is used. See :class:`FolderMode`."""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self.path = self.path.expanduser()
        if isinstance(self.mode, str):
            self.mode = FolderMode(self.mode)


@dataclass
class ServerSettings:
    port: int | None = None
    """Port for the server"""

    host: str | None = None
    """Hostname for the server"""

    autohost: str | None = None
    """Automatic hostname: values can be `fqdn` or `name`"""

    token: str | None = None
    """Token for the server"""


@dataclass
class CarbonSettings:
    """Settings for environmental impact tracking.

    Carbon tracking monitors CO2 emissions and energy consumption during
    job execution using CodeCarbon as the backend.
    """

    enabled: bool = True
    """Whether carbon tracking is enabled (default: True).

    When True and CodeCarbon is not installed, a warning is shown.
    Set to False to completely disable tracking and warnings.
    """

    provider: str = "codecarbon"
    """Carbon tracking provider (currently only 'codecarbon' supported)."""

    country_iso_code: str | None = None
    """Override auto-detected country (ISO 3166-1 alpha-3 code, e.g., 'FRA')."""

    region: str | None = None
    """Override auto-detected region."""

    report_interval_s: float = 60.0
    """How often to emit carbon metrics events during job execution (seconds)."""

    warn_if_unavailable: bool = True
    """Show warning and pause if CodeCarbon is not installed.

    Set to False to silently disable tracking when CodeCarbon is not available.
    This is ignored if enabled=False.
    """


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
class ReconnectSettings:
    """Reconnection of the remote (SSH) workspaces when the connection drops

    The monitor (TUI or web UI) reconnects on its own, at most
    ``max_retries`` times within ``window`` seconds; it then stays
    disconnected (restart it to try again). The delay between two attempts
    doubles from ``delay`` up to ``max_delay``.
    """

    max_retries: int = 5
    """Maximum number of reconnection attempts within the time window
    (0 disables the reconnection)"""

    window: float = 300.0
    """Time window (in seconds) over which the attempts are counted"""

    delay: float = 1.0
    """Delay (in seconds) before the first attempt"""

    max_delay: float = 30.0
    """Maximum delay (in seconds) between two attempts"""


@dataclass
class SSHSettings:
    """SSH settings for remote workspace access."""

    host: str = ""
    """SSH host (e.g., 'user@cluster' or SSH config alias).
    Marks the workspace as remote."""

    shell_init: str | None = None
    """Shell commands to run before the remote experimaestro command
    (e.g., 'source /etc/profile; module load python/3.10.4')"""

    uv_offline: bool = False
    """Pass --offline to uv tool run (use cached packages, no network)"""

    python: str | None = None
    """Python interpreter for uv tool run --python"""

    xpm_path: str | None = None
    """Path to experimaestro on remote host (bypasses uv tool run entirely)"""

    options: list[str] = field(default_factory=list)
    """Additional SSH options (e.g., ['-p', '2222'])"""

    log_levels: str | None = None
    """Logging levels of the remote server, shown locally (as ``--logging``,
    e.g. ``xpm.remote.trace=DEBUG`` to trace the requests)"""

    reconnect: ReconnectSettings | None = None
    """Reconnection policy of this workspace (defaults to the global
    ``reconnect`` settings)"""

    connect_timeout: float = 60.0
    """Time (in seconds) for the remote experimaestro server to start and
    answer (Python can be slow to start on shared file systems)"""


@dataclass
class WorkspaceSettings:
    """Defines the workspace"""

    id: str
    """The workspace identifier"""

    path: Path = field()
    """The workspace path"""

    env: dict[str, str] = field(default_factory=dict)
    """Workspace specific environment variables"""

    alt_workspaces: list[str] = field(default_factory=list)
    """Alternative workspaces to find jobs or experiments.

    .. deprecated::
        Use :attr:`folders` with ``mode: use`` instead. This field is
        kept for backwards-compatibility and silently merged into
        :attr:`folders` at load time.
    """

    folders: list[FolderSettings] = field(default_factory=list)
    """**Beta**: auxiliary folders for jobs (archival / read-through /
    migration). See :class:`FolderSettings` and :class:`FolderMode`."""

    max_retries: int = 3
    """Maximum number of retries for resumable tasks that timeout (default: 3)"""

    lock_mode: str = "inherit"
    """Permissions of the lock files created within this workspace

    - ``inherit`` (default): mirror the read/write permissions of the workspace
      directory, so that a group-writable workspace yields group-writable lock
      files whatever the umask is
    - ``umask``: apply the process umask to 0o666 (a umask of 022 gives 0644)
    - an octal mode such as ``"0664"``: use that mode

    The ``XPM_LOCK_MODE`` environment variable overrides this setting. See
    :ref:`sharing-a-workspace` for shared workspaces.
    """

    triggers: list[str] = field(default_factory=list)
    """Glob patterns to automatically select this workspace based on experiment ID"""

    history: HistorySettings = field(default_factory=HistorySettings)
    """Settings for experiment history cleanup"""

    ssh: SSHSettings | None = None
    """SSH settings for remote workspace access"""

    @property
    def is_remote(self) -> bool:
        """Returns True when ssh is set and ssh.host is non-empty."""
        return self.ssh is not None and bool(self.ssh.host)

    def __post_init__(self):
        # Store the raw path string before resolution (needed for remote workspaces
        # where the path refers to a location on a different machine)
        self._raw_path = str(self.path)
        if not self.is_remote:
            self.path = self.path.expanduser().resolve()


@dataclass
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    workspaces: list[WorkspaceSettings] = field(default_factory=list)

    env: dict[str, str] = field(default_factory=dict)
    """Default environment variables"""

    history: HistorySettings = field(default_factory=HistorySettings)
    """Default history settings (can be overridden per workspace)"""

    carbon: CarbonSettings = field(default_factory=CarbonSettings)
    """Settings for environmental impact tracking"""

    reconnect: ReconnectSettings = field(default_factory=ReconnectSettings)
    """Reconnection of the remote (SSH) workspaces (can be overridden per
    workspace, in its ``ssh`` settings)"""


def _interpolate(obj, is_env=False):
    """Interpolate environment variables and coerce types if needed"""
    if isinstance(obj, dict):
        return {k: _interpolate(v, is_env or k == "env") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, is_env) for v in obj]
    if isinstance(obj, str):
        # Handle ${env:VAR}, ${env.VAR}, ${oc.env:VAR} and ${oc.env.VAR}
        pattern = re.compile(r"\$\{(?:oc\.)?env[:.]([^}]+)\}")

        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return pattern.sub(replacer, obj)
    if is_env and isinstance(obj, (int, float, bool)):
        return str(obj)
    return obj


@lru_cache()
def get_settings(path: Path | None = None) -> Settings:
    if getattr(sys, "_called_from_test", False):
        settings = Settings()
        settings.carbon.enabled = False
        return settings
    else:
        # Pydantic 2.x adapter
        adapter = TypeAdapter(Settings)

        path = path or Path("~/.config/experimaestro/settings.yaml").expanduser()
        if not path.is_file():
            return adapter.validate_python({})

        with path.open("rt") as fp:
            conf = yaml.safe_load(fp)

        # Interpolate environment variables and coerce env values
        conf = _interpolate(conf)

        return adapter.validate_python(conf)


def get_workspace(
    id: str | None = None, *, include_remote: bool = False
) -> WorkspaceSettings | None:
    """Return the workspace settings given an id (or None for the default one)

    Args:
        id: Workspace ID to look up, or None for the default workspace
        include_remote: If False (default), skip remote workspaces when
            looking for the default workspace (id=None). When id is specified,
            remote workspaces are always returned.
    """
    workspaces = get_settings().workspaces
    if workspaces:
        if id is None:
            for workspace in workspaces:
                if include_remote or not workspace.is_remote:
                    return workspace
            return None
        for workspace in workspaces:
            if id == workspace.id:
                return workspace

    return None


def get_workspace_by_path(path: Path) -> WorkspaceSettings | None:
    """Return the settings of the (local) workspace located at a given path"""
    path = Path(path).expanduser().resolve()

    try:
        workspaces = get_settings().workspaces
    except Exception as e:
        logging.debug("Could not read the settings: %s", e)
        return None

    for workspace in workspaces:
        if not workspace.is_remote and Path(workspace.path) == path:
            return workspace

    return None


def find_workspace(
    *,
    workspace: str | None = None,
    workdir: Path | None = None,
    experiment_id: str | None = None,
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
        # (skip remote workspaces - they can't run experiments locally)
        matched_workspace = None
        if experiment_id:
            workspaces = get_settings().workspaces
            for ws in workspaces:
                if ws.is_remote:
                    continue
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
