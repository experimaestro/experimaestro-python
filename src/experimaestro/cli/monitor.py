# flake8: noqa: T201
"""``experimaestro monitor``: web UI, console TUI or event stream"""

from pathlib import Path

import click
from termcolor import cprint

from experimaestro.settings import WorkspaceSettings, get_settings, get_workspace

from . import deprecated_alias, root_config


def _run_monitor_ui(
    state_provider,
    workdir: Path,
    console: bool,
    port: int | None,
    title: str = "",
    events_viewer: bool = False,
    events_format: str = "text",
    events_show_progress: bool = True,
):
    """Shared code for running monitor UI (TUI, web, or events viewer)

    Args:
        state_provider: StateProvider instance (local or remote)
        workdir: Local workspace/cache directory
        console: If True, use TUI; otherwise use web UI
        port: Port for web server
        title: Optional title for status messages
        events_viewer: If True, stream events to console instead of UI
        events_format: Event format ('json' or 'text')
        events_show_progress: Whether to show progress events
    """
    try:
        if events_viewer:
            # Stream events to console
            from experimaestro.scheduler.event_viewer import run_event_viewer

            run_event_viewer(
                state_provider,
                format=events_format,
                show_progress=events_show_progress,
            )
        elif console:
            # Use Textual TUI
            from experimaestro.tui import ExperimentTUI

            app = ExperimentTUI(
                workdir, state_provider=state_provider, watch=True, show_logs=True
            )
            app.run()
        else:
            # Use React web server
            from experimaestro.webui import WebUIServer

            settings = get_settings().server
            if port is not None:
                settings.port = port
            elif settings.port is None:
                settings.port = 12345

            if title:
                cprint(
                    f"Starting experiment monitor for {title} on http://localhost:{settings.port}",
                    "green",
                )
            else:
                cprint(
                    f"Starting experiment monitor on http://localhost:{settings.port}",
                    "green",
                )
            cprint("Press Ctrl+C to stop", "yellow")

            server = WebUIServer.instance(settings, state_provider=state_provider)
            server.start()

            try:
                import time

                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    finally:
        cprint("\nShutting down...", "yellow")
        if state_provider:
            state_provider.close()


def _watcher_options(f):
    f = click.option(
        "--polling-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (only for --watcher=polling)",
    )(f)
    return click.option(
        "--watcher",
        type=click.Choice(
            ["auto", "polling", "inotify", "fsevents", "kqueue", "windows"]
        ),
        default="auto",
        help="Filesystem watcher type (auto=platform default, polling=network mounts)",
    )(f)


def _events_options(f):
    f = click.option(
        "--no-progress",
        is_flag=True,
        help="Hide progress events in events viewer (reduces noise)",
    )(f)
    f = click.option(
        "--events-format",
        type=click.Choice(["text", "json"]),
        default="text",
        help="Event output format (default: text, use json for tooling)",
    )(f)
    return click.option(
        "--events-viewer",
        is_flag=True,
        help="Stream events to console (for log aggregation)",
    )(f)


def _ssh_options(f):
    options = [
        click.option(
            "--remote-xpm",
            type=str,
            default=None,
            help="Path to experimaestro on remote host (default: use 'uv tool run')",
        ),
        click.option(
            "--ssh-option",
            "-o",
            multiple=True,
            help="Additional SSH options (can be repeated, e.g., -o '-p 2222')",
        ),
        click.option(
            "--remote-shell-init",
            type=str,
            default=None,
            help="Shell commands to run before remote experimaestro "
            "(e.g., 'source /etc/profile; module load python/3.10')",
        ),
        click.option(
            "--uv-offline",
            is_flag=True,
            default=False,
            help="Pass --offline to uv tool run (use cached packages, no network)",
        ),
        click.option(
            "--remote-python",
            type=str,
            default=None,
            help="Python interpreter for uv tool run --python on the remote host",
        ),
        click.option(
            "--force-version",
            is_flag=True,
            default=False,
            help="Pin exact experimaestro version (==X.Y.Z) instead of minor (>=X.Y.0)",
        ),
        click.option(
            "--request-timeout",
            type=float,
            default=60.0,
            help="Timeout in seconds for remote RPC requests (default: 60)",
        ),
        click.option(
            "--remote-logging",
            "remote_log_levels",
            type=str,
            default=None,
            help="Logging levels of the remote server, shown locally (as --logging, "
            "e.g. xpm.remote.trace=DEBUG to trace the requests)",
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f


def _configure_watcher(watcher: str, polling_interval: float):
    from experimaestro.filewatcher import FileWatcherService, WatcherType

    if watcher != "auto":
        FileWatcherService.configure(WatcherType(watcher), polling_interval)
    elif polling_interval != 1.0:
        FileWatcherService.configure(WatcherType.POLLING, polling_interval)


def _connect_ssh(
    host: str | None,
    remote_workdir: str | None,
    ws_settings: WorkspaceSettings | None,
    *,
    remote_xpm: str | None,
    ssh_option: tuple,
    remote_shell_init: str | None,
    uv_offline: bool,
    remote_python: str | None,
    force_version: bool,
    request_timeout: float,
    remote_log_levels: str | None,
):
    """Connects to a remote workspace (CLI flags override the workspace settings)"""
    effective_host = host
    effective_workdir = remote_workdir
    effective_xpm = remote_xpm
    effective_shell_init = remote_shell_init
    effective_uv_offline = uv_offline
    effective_python = remote_python
    effective_ssh_options = list(ssh_option) if ssh_option else []

    ws_ssh = ws_settings.ssh if ws_settings else None
    if ws_ssh:
        if not effective_host and ws_ssh.host:
            effective_host = ws_ssh.host
        if not effective_workdir:
            effective_workdir = ws_settings._raw_path
        if not effective_xpm and ws_ssh.xpm_path:
            effective_xpm = ws_ssh.xpm_path
        if not effective_shell_init and ws_ssh.shell_init:
            effective_shell_init = ws_ssh.shell_init
        if not effective_uv_offline and ws_ssh.uv_offline:
            effective_uv_offline = True
        if not effective_python and ws_ssh.python:
            effective_python = ws_ssh.python
        if not effective_ssh_options and ws_ssh.options:
            effective_ssh_options = ws_ssh.options
        if not remote_log_levels and ws_ssh.log_levels:
            remote_log_levels = ws_ssh.log_levels

    if not effective_host:
        cprint(
            "Error: HOST is required (provide it or use a workspace with SSH settings)",
            "red",
        )
        raise click.Abort()
    if not effective_workdir:
        cprint(
            "Error: the remote workspace path is required "
            "(provide it or use a workspace with SSH settings)",
            "red",
        )
        raise click.Abort()

    from experimaestro.scheduler.remote.client import SSHStateProviderClient

    cprint(f"Connecting to {effective_host}...", "yellow")
    state_provider = SSHStateProviderClient(
        host=effective_host,
        remote_workspace=effective_workdir,
        ssh_options=effective_ssh_options or None,
        remote_xpm_path=effective_xpm,
        remote_shell_init=effective_shell_init,
        uv_offline=effective_uv_offline,
        remote_python=effective_python,
        force_version=force_version,
        request_timeout=request_timeout,
        remote_log_levels=remote_log_levels,
    )
    try:
        state_provider.connect()
        cprint(f"Connected to {effective_host}", "green")
    except Exception as e:
        cprint(f"Failed to connect: {e}", "red")
        raise click.Abort()

    return effective_host, state_provider


@click.command()
@click.option(
    "--ssh",
    "ssh_target",
    metavar="HOST:PATH",
    default=None,
    help="Monitor the workspace PATH on a remote HOST (e.g. user@server:/path)",
)
@click.option("--console", is_flag=True, help="Use the terminal (Textual) UI")
@click.option("--web", is_flag=True, help="Use the web UI")
@_events_options
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port for web server (default: 12345 or from settings.yaml)",
)
@_watcher_options
@_ssh_options
@click.option(
    "--sync",
    is_flag=True,
    hidden=True,
    help="Deprecated: no longer needed (filesystem state is always current)",
)
def monitor(
    ssh_target: str | None,
    console: bool,
    web: bool,
    events_viewer: bool,
    events_format: str,
    no_progress: bool,
    port: int | None,
    watcher: str,
    polling_interval: float,
    sync: bool,
    **ssh_kwargs,
):
    """Monitor the experiments of a workspace (--console or --web)

    The workspace is the one selected with `experimaestro -w ID` or
    `experimaestro --workdir PATH`; a workspace with SSH settings is
    monitored remotely, as is any workspace given with --ssh.

    \b
    Examples:
      experimaestro monitor --web              # default workspace, web UI
      experimaestro -w my-cluster monitor --console  # remote workspace
      experimaestro monitor --ssh host:/path/ws --web  # ad-hoc remote
      experimaestro monitor --console             # terminal UI
    """
    if sync:
        cprint(
            "Note: --sync is deprecated and no longer needed "
            "(filesystem state is always current)",
            "yellow",
        )

    if console == web and not events_viewer:
        raise click.UsageError("Choose one of --console or --web")
    cfg = root_config()
    _configure_watcher(watcher, polling_interval)

    ws_settings = None
    if ssh_target is None and cfg.workspace_id and cfg.workdir is None:
        ws_settings = get_workspace(cfg.workspace_id, include_remote=True)

    title = ""
    if ssh_target is not None or (ws_settings is not None and ws_settings.is_remote):
        host, remote_workdir = None, None
        if ssh_target is not None:
            host, _, remote_workdir = ssh_target.partition(":")
            if not remote_workdir:
                raise click.UsageError("--ssh expects HOST:PATH")
        title, state_provider = _connect_ssh(
            host, remote_workdir, ws_settings, **ssh_kwargs
        )
        workdir = state_provider.local_cache_dir
    else:
        from experimaestro.scheduler.workspace_state_provider import (
            WorkspaceStateProvider,
        )

        workdir = cfg.workspace.path
        state_provider = WorkspaceStateProvider.get_instance(workdir)

    _run_monitor_ui(
        state_provider,
        workdir,
        console,
        port,
        title=title,
        events_viewer=events_viewer,
        events_format=events_format,
        events_show_progress=not no_progress,
    )


@click.command("ssh-monitor", hidden=True)
@click.argument("host", type=str, required=False, default=None)
@click.argument("remote_workdir", type=str, required=False, default=None)
@click.option(
    "--web", is_flag=True, help="Use web UI instead of console TUI (default is console)"
)
@_events_options
@click.option(
    "--port", type=int, default=12345, help="Port for web server (default: 12345)"
)
@_watcher_options
@_ssh_options
@click.option(
    "--workspace",
    "ssh_workspace",
    type=str,
    default=None,
    help="Load SSH settings from a workspace in settings.yaml",
)
def ssh_monitor(
    host: str | None,
    remote_workdir: str | None,
    web: bool,
    events_viewer: bool,
    events_format: str,
    no_progress: bool,
    port: int,
    watcher: str,
    polling_interval: float,
    ssh_workspace: str | None,
    **ssh_kwargs,
):
    """Monitor experiments on a remote server via SSH (deprecated)"""
    cprint(
        "Deprecated: use `experimaestro -w WORKSPACE monitor`"
        " or `experimaestro monitor --ssh HOST:PATH` instead",
        "yellow",
    )
    ws_settings = None
    if ssh_workspace:
        ws_settings = get_workspace(ssh_workspace, include_remote=True)
        if ws_settings is None:
            cprint(f"Workspace '{ssh_workspace}' not found in settings", "red")
            raise click.Abort()

    _configure_watcher(watcher, polling_interval)
    title, state_provider = _connect_ssh(
        host, remote_workdir, ws_settings, **ssh_kwargs
    )
    _run_monitor_ui(
        state_provider,
        state_provider.local_cache_dir,
        not web,
        port,
        title=title,
        events_viewer=events_viewer,
        events_format=events_format,
        events_show_progress=not no_progress,
    )


#: Old ``experiments monitor`` (it reads the workspace from the group options)
experiments_monitor = deprecated_alias(monitor, "monitor")
