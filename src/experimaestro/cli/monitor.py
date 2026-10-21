# flake8: noqa: T201
"""``experimaestro monitor``: web UI, console TUI or event stream"""

from pathlib import Path

import click
from termcolor import cprint

from experimaestro.settings import get_settings, get_workspace

from . import deprecated_alias, root_config
from .remote import RemoteWorkspace, SSHOptions, ssh_options


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


def _configure_watcher(watcher: str, polling_interval: float):
    from experimaestro.filewatcher import FileWatcherService, WatcherType

    if watcher != "auto":
        FileWatcherService.configure(WatcherType(watcher), polling_interval)
    elif polling_interval != 1.0:
        FileWatcherService.configure(WatcherType.POLLING, polling_interval)


@click.command()
@click.option(
    "--ssh",
    "ssh_target",
    metavar="HOST:PATH",
    default=None,
    hidden=True,
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
@ssh_options(hidden=True)
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

    The workspace is the one selected with `experimaestro -w ID`,
    `experimaestro --workdir PATH` or `experimaestro --ssh HOST:PATH`; a
    workspace with SSH settings is monitored remotely.

    \b
    Examples:
      experimaestro monitor --web              # default workspace, web UI
      experimaestro -w my-cluster monitor --console  # remote workspace
      experimaestro --ssh host:/path/ws monitor --web  # ad-hoc remote
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
    # Kept for compatibility: the SSH options used to be given here
    cfg.ssh.set_target(ssh_target)
    cfg.ssh.update(**ssh_kwargs)
    _configure_watcher(watcher, polling_interval)

    state_provider = cfg.state_provider()
    if (remote := cfg.remote) is not None:
        title = remote.host
        workdir = state_provider.local_cache_dir
    else:
        title = ""
        workdir = cfg.workspace.path

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
@ssh_options
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

    options = SSHOptions(host=host, path=remote_workdir)
    options.update(**ssh_kwargs)
    remote = RemoteWorkspace.resolve(options, ws_settings)

    _configure_watcher(watcher, polling_interval)
    state_provider = remote.connect()
    _run_monitor_ui(
        state_provider,
        state_provider.local_cache_dir,
        not web,
        port,
        title=remote.host,
        events_viewer=events_viewer,
        events_format=events_format,
        events_show_progress=not no_progress,
    )


#: Old ``experiments monitor`` (it reads the workspace from the group options)
experiments_monitor = deprecated_alias(monitor, "monitor")
