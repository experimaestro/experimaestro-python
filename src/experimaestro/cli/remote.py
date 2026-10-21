"""Remote (SSH) workspaces on the command line

A workspace is remote when it has SSH settings in ``settings.yaml``, or when it
is given with the root ``--ssh HOST:PATH`` option. The other root SSH options
override the workspace settings.

Commands reach a remote workspace in one of two ways:

- through the remote state provider (:meth:`RunConfig.state_provider`), for
  the commands that only query or act on the workspace state;
- by running the same command on the remote host (:class:`RemoteCommand`),
  for the commands that read or modify the workspace files directly.
"""

import subprocess
import sys
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import click
from termcolor import cprint

from experimaestro.settings import ReconnectSettings, WorkspaceSettings, get_settings

if TYPE_CHECKING:
    from experimaestro.scheduler.remote.client import SSHStateProviderClient

#: Key of the raw command line arguments in the click context metadata
RAW_ARGS = "experimaestro.raw_args"


@dataclass
class SSHOptions:
    """SSH options given on the command line (they override the settings)"""

    host: str | None = None
    path: str | None = None
    remote_xpm: str | None = None
    ssh_option: tuple[str, ...] = ()
    remote_shell_init: str | None = None
    uv_offline: bool = False
    remote_python: str | None = None
    force_version: bool = False
    request_timeout: float = 60.0
    remote_log_levels: str | None = None

    @property
    def is_set(self) -> bool:
        """Whether a remote workspace was given on the command line"""
        return self.host is not None

    def set_target(self, target: str | None):
        """Sets the remote workspace from ``HOST:PATH``"""
        if target is None:
            return
        host, _, path = target.partition(":")
        if not host or not path:
            raise click.UsageError("--ssh expects HOST:PATH")
        self.host, self.path = host, path

    def update(self, **kwargs):
        """Overrides the options that are not set to their default value"""
        defaults = {f.name: f.default for f in fields(self)}
        for key, value in kwargs.items():
            if value != defaults[key]:
                setattr(self, key, value)


def ssh_options(f=None, *, hidden: bool = False):
    """Adds the SSH options (``SSHOptions`` fields, except host and path)"""
    options = [
        click.option(
            "--remote-xpm",
            type=str,
            default=None,
            hidden=hidden,
            help="Path to experimaestro on remote host (default: use 'uv tool run')",
        ),
        click.option(
            "--ssh-option",
            "-o",
            multiple=True,
            hidden=hidden,
            help="Additional SSH options (can be repeated, e.g., -o '-p 2222')",
        ),
        click.option(
            "--remote-shell-init",
            type=str,
            default=None,
            hidden=hidden,
            help="Shell commands to run before remote experimaestro "
            "(e.g., 'source /etc/profile; module load python/3.10')",
        ),
        click.option(
            "--uv-offline",
            is_flag=True,
            default=False,
            hidden=hidden,
            help="Pass --offline to uv tool run (use cached packages, no network)",
        ),
        click.option(
            "--remote-python",
            type=str,
            default=None,
            hidden=hidden,
            help="Python interpreter for uv tool run --python on the remote host",
        ),
        click.option(
            "--force-version",
            is_flag=True,
            default=False,
            hidden=hidden,
            help="Pin exact experimaestro version (==X.Y.Z) instead of minor (>=X.Y.0)",
        ),
        click.option(
            "--request-timeout",
            type=float,
            default=60.0,
            hidden=hidden,
            help="Timeout in seconds for remote RPC requests (default: 60)",
        ),
        click.option(
            "--remote-logging",
            "remote_log_levels",
            type=str,
            default=None,
            hidden=hidden,
            help="Logging levels of the remote server, shown locally (as --logging, "
            "e.g. xpm.remote.trace=DEBUG to trace the requests)",
        ),
    ]

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    return decorator if f is None else decorator(f)


@dataclass
class RemoteWorkspace:
    """A workspace on a remote host, reached through SSH"""

    host: str
    path: str
    ssh_options: list[str] = field(default_factory=list)
    xpm_path: str | None = None
    shell_init: str | None = None
    uv_offline: bool = False
    python: str | None = None
    force_version: bool = False
    request_timeout: float = 60.0
    log_levels: str | None = None
    reconnect: ReconnectSettings = field(default_factory=ReconnectSettings)

    @staticmethod
    def resolve(
        options: SSHOptions, settings: WorkspaceSettings | None
    ) -> "RemoteWorkspace":
        """Merges the command line options with the workspace SSH settings"""
        ssh = settings.ssh if settings is not None else None
        host = options.host or (ssh.host if ssh else None)
        path = options.path or (settings._raw_path if ssh else None)
        if not host:
            raise click.UsageError(
                "The remote host is required "
                "(use --ssh HOST:PATH or a workspace with SSH settings)"
            )
        if not path:
            raise click.UsageError(
                "The remote workspace path is required "
                "(use --ssh HOST:PATH or a workspace with SSH settings)"
            )

        return RemoteWorkspace(
            host=host,
            path=path,
            ssh_options=list(options.ssh_option) or (ssh.options if ssh else []),
            xpm_path=options.remote_xpm or (ssh.xpm_path if ssh else None),
            shell_init=options.remote_shell_init or (ssh.shell_init if ssh else None),
            uv_offline=options.uv_offline or (ssh.uv_offline if ssh else False),
            python=options.remote_python or (ssh.python if ssh else None),
            force_version=options.force_version,
            request_timeout=options.request_timeout,
            log_levels=options.remote_log_levels or (ssh.log_levels if ssh else None),
            reconnect=(ssh.reconnect if ssh else None) or get_settings().reconnect,
        )

    def __str__(self):
        return f"{self.host}:{self.path}"

    def command(self, args: list[str], *, tty: bool = False) -> list[str]:
        """The SSH command running ``experimaestro --workdir PATH ARGS`` remotely"""
        from experimaestro.scheduler.remote.client import (
            remote_shell_command,
            remote_xpm_command,
        )

        remote = remote_xpm_command(
            self.xpm_path,
            uv_offline=self.uv_offline,
            remote_python=self.python,
            force_version=self.force_version,
        )
        remote.extend(["--workdir", self.path, *args])
        return [
            "ssh",
            *(["-t"] if tty else []),
            *self.ssh_options,
            self.host,
            remote_shell_command(remote, self.shell_init),
        ]

    def run(self, args: list[str]) -> int:
        """Runs ``experimaestro ARGS`` on the remote workspace

        The command is connected to the terminal (with a remote terminal when
        the standard input is one, for prompts and pagers).
        """
        try:
            return subprocess.run(self.command(args, tty=sys.stdin.isatty())).returncode
        except KeyboardInterrupt:
            return 130

    def connect(self) -> "SSHStateProviderClient":
        """Connects to the remote state provider"""
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        cprint(f"Connecting to {self.host}...", "yellow", file=sys.stderr)
        provider = SSHStateProviderClient(
            host=self.host,
            remote_workspace=self.path,
            ssh_options=self.ssh_options or None,
            remote_xpm_path=self.xpm_path,
            remote_shell_init=self.shell_init,
            uv_offline=self.uv_offline,
            remote_python=self.python,
            force_version=self.force_version,
            request_timeout=self.request_timeout,
            remote_log_levels=self.log_levels,
            reconnect=self.reconnect,
        )
        try:
            provider.connect()
        except Exception as e:
            cprint(f"Failed to connect to {self.host}: {e}", "red", file=sys.stderr)
            raise click.Abort()
        cprint(f"Connected to {self.host}", "green", file=sys.stderr)
        return provider


class RemoteCommand(click.Command):
    """A command that runs on the remote host when the workspace is remote

    The command is run as ``experimaestro [ROOT OPTIONS] --workdir PATH
    COMMAND PATH ARGS``, with the arguments given locally (the options of the
    intermediate groups have already been taken into account).
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta[RAW_ARGS] = list(args)
        return super().parse_args(ctx, args)

    def invoke(self, ctx: click.Context):
        cfg = ctx.find_root().obj
        if (remote := cfg.remote) is None:
            return super().invoke(ctx)

        names = []
        c = ctx
        while c.parent is not None:
            names.append(c.info_name)
            c = c.parent
        args = [*cfg.root_args, *reversed(names), *ctx.meta[RAW_ARGS]]
        ctx.exit(remote.run(args))
