# flake8: noqa: T201
"""The ``experimaestro`` command line

Commands are grouped by what they act on (``experiments``, ``jobs``,
``workspace``) plus ``tools`` for development helpers; ``run`` and ``monitor``
are the two everyday entry points. The workspace is chosen once, with the root
``--workspace``/``--workdir``/``--ssh`` options, and read by the subcommands
through :class:`RunConfig`. A remote workspace is reached through its state
provider, or by running the command remotely (see :mod:`.remote`).

Old command names are kept as hidden aliases (see :func:`deprecated_alias`).
``rpyc-server`` and ``experiments --workdir X monitor-server`` are called by
other experimaestro processes, possibly of another version over SSH: they are
hidden but must never be renamed.
"""

import copy
import json
import sys
import click

import logging
from functools import cached_property, update_wrapper
from pathlib import Path
from typing import TYPE_CHECKING
from termcolor import cprint
from importlib.metadata import entry_points

from experimaestro.experiments.cli import experiments_cli
from experimaestro.settings import WorkspaceSettings, find_workspace, get_workspace

from .remote import RemoteWorkspace, SSHOptions, ssh_options

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import StateProvider

# --- Command line main options
# Only configure logging if not running under pytest (pytest has its own config)
if not hasattr(sys, "_called_from_test"):
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def check_xp_path(ctx, self, path: Path):
    if not (path / ".__experimaestro__").is_file():
        cprint(f"{path} is not an experimaestro working directory", "red")
        for path in path.parents:
            if (path / ".__experimaestro__").is_file():
                cprint(f"{path} could be the folder you want", "green")
                if click.confirm("Do you want to use this folder?"):
                    return path
        sys.exit(1)

    return path


class RunConfig:
    """Options of the root command, shared by all the subcommands"""

    def __init__(self):
        self.traceback = False
        self.workdir: Path | None = None
        self.workspace_id: str | None = None
        self._workspace: WorkspaceSettings | None = None

        #: SSH options (a remote workspace given with ``--ssh``, overrides)
        self.ssh = SSHOptions()

        #: Root options passed along to the remote commands
        self.root_args: list[str] = []

        self._provider: "StateProvider | None" = None

    def set_workspace(self, workdir: Path | None, workspace: str | None):
        """Overrides the workspace (if any is given)"""
        if workdir is not None or workspace:
            self.workdir = workdir
            self.workspace_id = workspace or None
            self._workspace = None
            self._provider = None

    @property
    def workspace_spec(self) -> str | None:
        """The workspace as given on the command line (ID or path), if any"""
        if self.workspace_id:
            return self.workspace_id
        return str(self.workdir) if self.workdir is not None else None

    @property
    def remote(self) -> RemoteWorkspace | None:
        """The selected workspace, if it is remote"""
        settings = None
        if self.workspace_id and self.workdir is None:
            settings = get_workspace(self.workspace_id, include_remote=True)
        if self.ssh.is_set or (settings is not None and settings.is_remote):
            return RemoteWorkspace.resolve(self.ssh, settings)
        return None

    @property
    def workspace(self) -> WorkspaceSettings:
        """The selected (local) workspace, resolved on first use"""
        if self._workspace is None:
            if (remote := self.remote) is not None:
                command = click.get_current_context().command_path
                raise click.UsageError(
                    f"`{command}` does not work with a remote workspace ({remote})"
                )
            ws = find_workspace(workdir=self.workdir, workspace=self.workspace_id)
            path = check_xp_path(None, None, ws.path)
            if path != ws.path:
                ws = find_workspace(workdir=path)
            self._workspace = ws
        return self._workspace

    def state_provider(self) -> "StateProvider":
        """The state provider of the selected workspace (local or remote)

        It is closed when the command line ends.
        """
        if self._provider is None:
            if (remote := self.remote) is not None:
                provider = remote.connect()
            else:
                from experimaestro.scheduler.workspace_state_provider import (
                    WorkspaceStateProvider,
                )

                provider = WorkspaceStateProvider.get_instance(self.workspace.path)
            click.get_current_context().find_root().call_on_close(provider.close)
            self._provider = provider
        return self._provider


def root_config() -> RunConfig:
    return click.get_current_context().find_root().obj


def pass_workdir(f):
    """Pass the path of the selected workspace as first argument"""

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, root_config().workspace.path, *args, **kwargs)

    return update_wrapper(new_func, f)


def hidden_alias(command: click.Command) -> click.Command:
    """A copy of ``command`` that is not shown in the help"""
    alias = copy.copy(command)
    alias.hidden = True
    return alias


def deprecated_alias(command: click.Command, new_name: str) -> click.Command:
    """A hidden copy of ``command`` that warns it should be called ``new_name``"""
    alias = hidden_alias(command)
    callback = command.callback

    def warn(*args, **kwargs):
        cprint(
            f"Deprecated: use `experimaestro {new_name}` instead",
            "yellow",
            file=sys.stderr,
        )
        return callback(*args, **kwargs)

    alias.callback = update_wrapper(warn, callback)
    return alias


@click.group()
@click.option("--quiet", is_flag=True, help="Be quiet")
@click.option("--debug", is_flag=True, help="Be even more verbose (implies traceback)")
@click.option(
    "--logging",
    "log_levels",
    help="Set logging levels (e.g., --logging xpm.state=DEBUG,xpm.webui=INFO)",
)
@click.option(
    "--traceback", is_flag=True, help="Display traceback if an exception occurs"
)
@click.option(
    "--workspace",
    "-w",
    type=str,
    default=None,
    help="Workspace ID (from settings.yaml; default: the first one)",
)
@click.option(
    "--workdir",
    type=Path,
    default=None,
    help="Workspace directory (instead of a workspace ID)",
)
@click.option(
    "--ssh",
    "ssh_target",
    metavar="HOST:PATH",
    default=None,
    help="Remote workspace PATH on HOST, reached through SSH (e.g. user@server:/path)",
)
@ssh_options
@click.pass_context
def cli(
    ctx,
    quiet,
    debug,
    log_levels: str | None,
    traceback,
    workspace: str | None,
    workdir: Path | None,
    ssh_target: str | None,
    **ssh_kwargs,
):
    if quiet:
        logging.getLogger().setLevel(logging.WARN)
    elif debug:
        # Set DEBUG level for all xpm.* loggers
        logging.getLogger("xpm").setLevel(logging.DEBUG)

    # Parse and apply custom log levels (e.g., "xpm.workspace_state=DEBUG,xpm.webui=INFO")
    if log_levels:
        for item in log_levels.split(","):
            item = item.strip()
            if "=" in item:
                logger_name, level_str = item.split("=", 1)
                level = getattr(logging, level_str.upper(), None)
                if level is not None:
                    logging.getLogger(logger_name.strip()).setLevel(level)
                else:
                    click.echo(f"Warning: Unknown log level '{level_str}'", err=True)

    if ssh_target is not None and workdir is not None:
        raise click.UsageError("--ssh and --workdir cannot be used together")

    ctx.obj = RunConfig()
    ctx.obj.traceback = traceback
    ctx.obj.set_workspace(workdir, workspace)
    ctx.obj.ssh.set_target(ssh_target)
    ctx.obj.ssh.update(**ssh_kwargs)
    ctx.obj.root_args = [
        *(["--quiet"] if quiet else []),
        *(["--debug"] if debug else []),
        *(["--logging", log_levels] if log_levels else []),
        *(["--traceback"] if traceback else []),
    ]


@cli.command(help="Get version")
def version():
    # Not through ``experimaestro``, which imports the whole framework
    try:
        from experimaestro.version import __version__
    except ImportError:
        __version__ = "0.0.0"
    print(__version__)


# --- Run an experiment


def _is_task_parameters(path: str) -> bool:
    """Whether ``path`` is the ``params.json`` of a job (old ``run`` command)"""
    if not path.endswith(".json"):
        return False
    try:
        with open(path) as fp:
            content = json.load(fp)
    except (OSError, ValueError):
        return False
    return isinstance(content, dict) and "objects" in content


def _run(**kwargs):
    if (remote := root_config().remote) is not None:
        raise click.UsageError(
            f"Experiments cannot be run on a remote workspace ({remote})"
        )
    if not kwargs["args"] and _is_task_parameters(kwargs["yaml_file"]):
        # Before 2.x, `experimaestro run` ran a single task
        from experimaestro.run import run as run_task

        cprint(
            "Deprecated: `experimaestro run PARAMS.json` runs a single task;"
            " `run` now runs an experiment",
            "yellow",
            file=sys.stderr,
        )
        return run_task(Path(kwargs["yaml_file"]))

    return experiments_cli.callback(**kwargs)


run = copy.copy(experiments_cli)
run.callback = update_wrapper(_run, experiments_cli.callback)
cli.add_command(run, "run")
cli.add_command(deprecated_alias(run, "run"), "run-experiment")


@click.option(
    "--clean", is_flag=True, help="Remove the socket file and its enclosing directory"
)
@click.argument("unix-path", type=Path)
@cli.command(hidden=True)
def rpyc_server(unix_path, clean):
    """Start an rPyC server (used by the SSH connector)"""
    from experimaestro.rpyc import start_server

    start_server(unix_path, clean=clean)


class Launchers(click.Group):
    """Dynamic command group for entry point discovery.

    Loads subcommands from entry points in the `experimaestro.{name}` group.
    Each entry point should have a `get_cli()` static method that returns
    a click Command or Group.
    """

    @cached_property
    def _ep_commands(self):
        """Load commands from entry points (cached)."""
        cmds = {}
        for ep in entry_points(group=f"experimaestro.{self.name}"):
            if get_cli := getattr(ep.load(), "get_cli", None):
                cmds[ep.name] = get_cli()
        return cmds

    def list_commands(self, ctx):
        return list(self._ep_commands.keys())

    def get_command(self, ctx, name):
        return self._ep_commands.get(name)


cli.add_command(Launchers("launchers", help="Launcher specific commands"))
cli.add_command(Launchers("connectors", help="Connector specific commands"))
cli.add_command(Launchers("tokens", help="Token specific commands"))


# --- Command groups (imported last: they use the helpers above)

from .monitor import monitor  # noqa: E402
from .experiments import experiments  # noqa: E402
from .jobs import jobs  # noqa: E402
from .workspace import workspace  # noqa: E402
from .tools import tools  # noqa: E402

cli.add_command(monitor)
cli.add_command(experiments)
cli.add_command(jobs)
cli.add_command(workspace)
cli.add_command(tools)

# Short names
cli.add_command(hidden_alias(experiments), "xp")
cli.add_command(hidden_alias(workspace), "ws")
cli.add_command(hidden_alias(jobs.commands["inspect"]), "inspect")

# Old names
for old_name, group, name in [
    ("dag", experiments, "dag"),
    ("parameters-difference", jobs, "diff"),
    ("workspaces", workspace, None),
    ("cleanup", workspace, "cleanup"),
    ("migrate", workspace, "migrate"),
    ("check-documentation", tools, "check-documentation"),
    ("find-launchers", tools, "find-launchers"),
    ("install-skill", tools, "install-skill"),
    ("refactor", tools, "refactor"),
    ("deprecated", tools, "deprecated"),
]:
    command = group.commands[name] if name else group
    new_name = f"{group.name} {name}" if name else group.name
    cli.add_command(deprecated_alias(command, new_name), old_name)

from .workspace import orphans_alias  # noqa: E402

cli.add_command(orphans_alias, "orphans")
jobs.add_command(
    deprecated_alias(workspace.commands["stray"], "workspace stray"), "stray"
)
