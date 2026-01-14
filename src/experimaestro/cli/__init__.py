# flake8: noqa: T201
import sys
from typing import Set, Optional
from shutil import rmtree
import click
import logging
from functools import cached_property, update_wrapper
from pathlib import Path
import subprocess
from termcolor import cprint
from importlib.metadata import entry_points

import experimaestro
from experimaestro.experiments.cli import experiments_cli
import experimaestro.launcherfinder.registry as launcher_registry
from experimaestro.settings import ServerSettings, find_workspace

# --- Command line main options
# Only configure logging if not running under pytest (pytest has its own config)
if not hasattr(sys, "_called_from_test"):
    logging.basicConfig(level=logging.INFO)


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
    def __init__(self):
        self.traceback = False


def pass_cfg(f):
    """Pass configuration information"""

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj, *args, **kwargs)

    return update_wrapper(new_func, f)


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
@click.pass_context
def cli(ctx, quiet, debug, log_levels: str | None, traceback):
    if quiet:
        logging.getLogger().setLevel(logging.WARN)
    elif debug:
        logging.getLogger().setLevel(logging.DEBUG)

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

    ctx.obj = RunConfig()
    ctx.obj.traceback = traceback


# Adds the run-experiment command
cli.add_command(experiments_cli, "run-experiment")


@cli.command(help="Get version")
def version():
    print(experimaestro.__version__)


@click.argument("parameters", type=Path)
@cli.command(context_settings={"allow_extra_args": True})
def run(parameters):
    """Run a task"""

    from experimaestro.run import run as do_run

    do_run(parameters)


@click.argument("path2", type=Path)
@click.argument("path1", type=Path)
@cli.command(context_settings={"allow_extra_args": True})
def parameters_difference(path1, path2):
    """Compute the difference between two configurations"""

    from experimaestro.tools.diff import diff

    diff(path1, path2)


@click.option(
    "--clean", is_flag=True, help="Remove the socket file and its enclosing directory"
)
@click.argument("unix-path", type=Path)
@cli.command()
def rpyc_server(unix_path, clean):
    """Start an rPyC server"""
    from experimaestro.rpyc import start_server

    start_server(unix_path, clean=clean)


@cli.group()
def deprecated():
    """Manage identifier changes"""
    pass


@click.option("--fix", is_flag=True, help="Generate links to new IDs")
@click.option("--cleanup", is_flag=True, help="Remove symbolic links and move folders")
@click.argument("path", type=Path, callback=check_xp_path)
@deprecated.command(name="list")
def deprecated_list(path: Path, fix: bool, cleanup: bool):
    """List deprecated jobs"""
    from experimaestro.tools.jobs import fix_deprecated

    if cleanup and not fix:
        logging.warning("Ignoring --cleanup since we are not fixing old IDs")
    fix_deprecated(path, fix, cleanup)


@click.argument("path", type=Path, callback=check_xp_path)
@deprecated.command()
def diff(path: Path):
    """Show the reason of the identifier change for a job"""
    from experimaestro.tools.jobs import load_job
    from experimaestro import Config

    _, job = load_job(path / "params.json", discard_id=False)
    _, new_job = load_job(path / "params.json")

    def check(path: str, value, new_value, done: Set[int]):
        if isinstance(value, Config):
            if id(value) in done:
                return
            done.add(id(value))

            old_id = value.__xpm__.identifier.all.hex()
            new_id = new_value.__xpm__.identifier.all.hex()

            if new_id != old_id:
                print(f"{path} differ: {new_id} vs {old_id}")

                for arg in value.__xpmtype__.arguments.values():
                    arg_value = getattr(value, arg.name)
                    arg_newvalue = getattr(new_value, arg.name)
                    check(f"{path}/{arg.name}", arg_value, arg_newvalue, done)

        elif isinstance(value, list):
            for ix, (array_value, array_newvalue) in enumerate(zip(value, new_value)):
                check(f"{path}.{ix}", array_value, array_newvalue, done)

        elif isinstance(value, dict):
            for key, dict_value in value.items():
                check(f"{path}.{key}", dict_value, new_value[key], done)

    check(".", job, new_job, set())


@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.argument("path", type=Path, callback=check_xp_path)
@cli.command()
def orphans(path: Path, clean: bool, size: bool):
    """Check for tasks that are not part of an experimental plan

    Uses the same orphan detection as the TUI (WorkspaceStateProvider.get_orphan_jobs).
    """
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    def show(job, prefix=""):
        key = f"{job.task_id}/{job.identifier}"
        if size:
            print(
                prefix,
                subprocess.check_output(["du", "-hs", str(job.path)])
                .decode("utf-8")
                .strip(),
                sep=None,
            )
        else:
            print(prefix, key, sep=None)

    # Use WorkspaceStateProvider.get_orphan_jobs() - same as TUI
    provider = WorkspaceStateProvider.get_instance(path)
    orphan_jobs = provider.get_orphan_jobs()

    if not orphan_jobs:
        print("No orphan jobs found.")
        return

    print(f"Found {len(orphan_jobs)} orphan job(s):")
    for job in orphan_jobs:
        show(job)
        if clean:
            logging.info("Removing data in %s", job.path)
            if job.path and job.path.exists():
                rmtree(job.path)


def arg_split(ctx, param, value):
    # split columns by ',' and remove whitespace
    return set(c.strip() for c in value.split(","))


@click.option("--skip", default=set(), callback=arg_split)
@click.argument("package", type=str)
@click.argument("objects", type=Path)
@cli.command()
def check_documentation(objects, package, skip):
    """Check that all the configuration and tasks are documented within a
    package, relying on the sphinx objects.inv file"""
    from experimaestro.tools.documentation import documented_from_objects, undocumented

    documented = documented_from_objects(objects)
    errors, configs = undocumented([package], documented, skip)
    for config in configs:
        cprint(f"{config.__module__}.{config.__qualname__}", "red")

    if errors > 0 or configs:
        sys.exit(1)


@click.option("--config", type=Path, help="Show size of each folder")
@click.argument("spec", type=str)
@cli.command()
def find_launchers(config: Optional[Path], spec: str):
    """Find launchers matching a specification"""
    if config is not None:
        launcher_registry.LauncherRegistry.set_config_dir(config)

    print(launcher_registry.find_launcher(spec))


class Launchers(click.Group):
    """Dynamic command group for entry point discovery"""

    @cached_property
    def commands(self):
        map = {}
        for ep in entry_points(group=f"experimaestro.{self.name}"):
            if get_cli := getattr(ep.load(), "get_cli", None):
                map[ep.name] = get_cli()
        return map

    def list_commands(self, ctx):
        return self.commands.keys()

    def get_command(self, ctx, name):
        return self.commands[name]


cli.add_command(Launchers("launchers", help="Launcher specific commands"))
cli.add_command(Launchers("connectors", help="Connector specific commands"))
cli.add_command(Launchers("tokens", help="Token specific commands"))

# Import and add jobs commands
from .jobs import jobs as jobs_cli  # noqa: E402

cli.add_command(jobs_cli)

# Import and add refactor commands
from .refactor import refactor as refactor_cli  # noqa: E402

cli.add_command(refactor_cli)


@cli.group()
def migrate():
    """Migration commands for experimaestro workspace upgrades"""
    pass


@migrate.command("v1-to-v2")
@click.argument("workdir", type=Path, callback=check_xp_path)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
def migrate_v1_to_v2(workdir: Path, dry_run: bool):
    """Migrate workspace from v1 (xp/) to v2 (experiments/) layout

    This command migrates experiment directories from the old layout:
        workdir/xp/{experiment-id}/
    to the new layout:
        workdir/experiments/{experiment-id}/{run-id}/

    Each old experiment directory becomes a single run directory with the
    run ID based on its modification time.
    """
    from datetime import datetime

    old_xp_dir = workdir / "xp"
    new_experiments_dir = workdir / "experiments"

    if not old_xp_dir.exists():
        cprint(f"No old 'xp' directory found at {old_xp_dir}", "yellow")
        return

    # List all experiments in the old directory
    old_experiments = [d for d in old_xp_dir.iterdir() if d.is_dir()]

    if not old_experiments:
        cprint("No experiments found in xp/ directory", "yellow")
        return

    cprint(f"Found {len(old_experiments)} experiment(s) to migrate:", "cyan")
    for exp_dir in old_experiments:
        cprint(f"  - {exp_dir.name}", "white")

    if dry_run:
        cprint("\nDRY RUN MODE - showing what would be done:", "yellow")

    migrated = 0
    for exp_dir in old_experiments:
        exp_id = exp_dir.name

        # Generate run_id from directory modification time
        mtime = exp_dir.stat().st_mtime
        mtime_dt = datetime.fromtimestamp(mtime)
        run_id = mtime_dt.strftime("%Y%m%d_%H%M%S")

        # Target path
        new_exp_base = new_experiments_dir / exp_id
        new_run_dir = new_exp_base / run_id

        # Handle collision
        suffix = 1
        while new_run_dir.exists():
            run_id = f"{mtime_dt.strftime('%Y%m%d_%H%M%S')}.{suffix}"
            new_run_dir = new_exp_base / run_id
            suffix += 1

        if dry_run:
            cprint(f"  {exp_dir} -> {new_run_dir}", "white")
        else:
            # Create the parent directory
            new_exp_base.mkdir(parents=True, exist_ok=True)

            # Move the experiment directory
            import shutil

            try:
                shutil.move(str(exp_dir), str(new_run_dir))
                cprint(
                    f"  Migrated: {exp_id} -> {new_run_dir.relative_to(workdir)}",
                    "green",
                )
                migrated += 1
            except Exception as e:
                cprint(f"  Failed to migrate {exp_id}: {e}", "red")

    if not dry_run:
        cprint(f"\nMigrated {migrated}/{len(old_experiments)} experiment(s)", "cyan")

        # Handle old xp directory
        remaining = list(old_xp_dir.iterdir())
        if remaining:
            # Rename directory to preserve any leftover files
            renamed_xp_dir = workdir / "xp_MIGRATED_TO_V2"
            old_xp_dir.rename(renamed_xp_dir)
            cprint(
                f"Renamed 'xp' -> 'xp_MIGRATED_TO_V2' ({len(remaining)} leftover item(s))",
                "yellow",
            )
        else:
            # Empty directory - remove it
            old_xp_dir.rmdir()
            cprint("Removed empty 'xp' directory", "green")

        # Create a broken symlink to prevent v1 from recreating xp/
        # v1 will find the symlink but fail when trying to use it
        broken_link = workdir / "xp"
        if not broken_link.exists() and not broken_link.is_symlink():
            broken_link.symlink_to("/experimaestro_v2_migrated_workspace_do_not_use_v1")
            cprint(
                "Created broken 'xp' symlink to prevent experimaestro v1 usage", "green"
            )


@cli.group()
@click.option("--workdir", type=Path, default=None)
@click.option("--workspace", type=str, default=None)
@click.pass_context
def experiments(ctx, workdir, workspace):
    """Manage experiments"""
    ws = find_workspace(workdir=workdir, workspace=workspace)
    path = check_xp_path(None, None, ws.path)
    ctx.obj = path


@experiments.command("list")
@pass_cfg
def list_experiments(workdir: Path):
    """List experiments in the workspace"""
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    # Get experiments from state provider for detailed info
    state_provider = WorkspaceStateProvider.get_instance(workdir)
    experiments_list = state_provider.get_experiments()

    # Build lookup by experiment_id
    exp_info = {exp.experiment_id: exp for exp in experiments_list}

    # New layout: experiments/{exp-id}/{run-id}/
    experiments_dir = workdir / "experiments"
    if not experiments_dir.exists():
        cprint("No experiments found", "yellow")
        return

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_id = exp_dir.name
        exp = exp_info.get(exp_id)

        # Build display string
        display_parts = [exp_id]

        # Add current run_id if available
        if exp and getattr(exp, "current_run_id", None):
            display_parts.append(f"[run: {exp.current_run_id}]")

        # Add hostname if available
        if exp and getattr(exp, "hostname", None):
            display_parts.append(f"[{exp.hostname}]")

        # Add job stats if available
        if exp:
            display_parts.append(f"({exp.finished_jobs}/{exp.total_jobs} jobs)")

        display_str = " ".join(display_parts)
        cprint(display_str, "cyan")


def _run_monitor_ui(
    state_provider, workdir: Path, console: bool, port: int, title: str = ""
):
    """Shared code for running monitor UI (TUI or web)

    Args:
        state_provider: StateProvider instance (local or remote)
        workdir: Local workspace/cache directory
        console: If True, use TUI; otherwise use web UI
        port: Port for web server
        title: Optional title for status messages
    """
    try:
        if console:
            # Use Textual TUI
            from experimaestro.tui import ExperimentTUI

            app = ExperimentTUI(
                workdir, state_provider=state_provider, watch=True, show_logs=True
            )
            app.run()
        else:
            # Use React web server
            from experimaestro.webui import WebUIServer

            if title:
                cprint(
                    f"Starting experiment monitor for {title} on http://localhost:{port}",
                    "green",
                )
            else:
                cprint(
                    f"Starting experiment monitor on http://localhost:{port}", "green"
                )
            cprint("Press Ctrl+C to stop", "yellow")

            settings = ServerSettings()
            settings.port = port
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


@experiments.command()
@click.option("--console", is_flag=True, help="Use console TUI instead of web UI")
@click.option(
    "--port", type=int, default=12345, help="Port for web server (default: 12345)"
)
@click.option(
    "--watcher",
    type=click.Choice(["auto", "polling", "inotify", "fsevents", "kqueue", "windows"]),
    default="auto",
    help="Filesystem watcher type (auto=platform default, polling=network mounts)",
)
@click.option(
    "--polling-interval",
    type=float,
    default=1.0,
    help="Polling interval in seconds (only for --watcher=polling)",
)
@click.option(
    "--sync",
    is_flag=True,
    hidden=True,
    help="Deprecated: no longer needed (filesystem state is always current)",
)
@pass_cfg
def monitor(
    workdir: Path,
    console: bool,
    port: int,
    watcher: str,
    polling_interval: float,
    sync: bool,
):
    """Monitor local experiments with web UI or console TUI"""
    # --sync is deprecated (kept for backwards compatibility)
    if sync:
        cprint(
            "Note: --sync is deprecated and no longer needed "
            "(filesystem state is always current)",
            "yellow",
        )

    # Configure filesystem watcher type
    from experimaestro.ipc import IPCom, WatcherType

    if watcher != "auto":
        IPCom.set_watcher_type(WatcherType(watcher), polling_interval)
    elif polling_interval != 1.0:
        IPCom.set_watcher_type(WatcherType.POLLING, polling_interval)

    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    state_provider = WorkspaceStateProvider.get_instance(workdir)

    _run_monitor_ui(state_provider, workdir, console, port)


@experiments.command("ssh-monitor")
@click.argument("host", type=str)
@click.argument("remote_workdir", type=str)
@click.option("--console", is_flag=True, help="Use console TUI instead of web UI")
@click.option(
    "--port", type=int, default=12345, help="Port for web server (default: 12345)"
)
@click.option(
    "--watcher",
    type=click.Choice(["auto", "polling", "inotify", "fsevents", "kqueue", "windows"]),
    default="auto",
    help="Filesystem watcher type (auto=platform default, polling=network mounts)",
)
@click.option(
    "--polling-interval",
    type=float,
    default=1.0,
    help="Polling interval in seconds (only for --watcher=polling)",
)
@click.option(
    "--remote-xpm",
    type=str,
    default=None,
    help="Path to experimaestro on remote host (default: use 'uv tool run')",
)
@click.option(
    "--ssh-option",
    "-o",
    multiple=True,
    help="Additional SSH options (can be repeated, e.g., -o '-p 2222')",
)
def ssh_monitor(
    host: str,
    remote_workdir: str,
    console: bool,
    port: int,
    watcher: str,
    polling_interval: float,
    remote_xpm: str,
    ssh_option: tuple,
):
    """Monitor experiments on a remote server via SSH

    HOST is the SSH host (e.g., user@server)
    REMOTE_WORKDIR is the workspace path on the remote server

    Examples:
        experimaestro experiments ssh-monitor myserver /path/to/workspace
        experimaestro experiments ssh-monitor user@host /workspace --console
        experimaestro experiments ssh-monitor host /workspace --remote-xpm /opt/xpm/bin/experimaestro
    """
    # Configure filesystem watcher type
    from experimaestro.ipc import IPCom, WatcherType

    if watcher != "auto":
        IPCom.set_watcher_type(WatcherType(watcher), polling_interval)
    elif polling_interval != 1.0:
        IPCom.set_watcher_type(WatcherType.POLLING, polling_interval)

    from experimaestro.scheduler.remote.client import SSHStateProviderClient

    cprint(f"Connecting to {host}...", "yellow")
    state_provider = SSHStateProviderClient(
        host=host,
        remote_workspace=remote_workdir,
        ssh_options=list(ssh_option) if ssh_option else None,
        remote_xpm_path=remote_xpm,
    )
    try:
        state_provider.connect()
        cprint(f"Connected to {host}", "green")
    except Exception as e:
        cprint(f"Failed to connect: {e}", "red")
        raise click.Abort()

    _run_monitor_ui(
        state_provider,
        state_provider.local_cache_dir,
        console,
        port,
        title=host,
    )


@experiments.command("monitor-server")
@pass_cfg
def monitor_server(workdir: Path):
    """Start monitoring server for SSH connections (JSON-RPC over stdio)

    This command is intended to be run over SSH to provide remote monitoring.
    Communication is via JSON-RPC over stdin/stdout.

    Example:
        ssh host 'experimaestro experiments --workdir /path monitor-server'
    """
    from experimaestro.scheduler.remote.server import SSHStateProviderServer

    server = SSHStateProviderServer(workdir)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


@experiments.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="[DEPRECATED] No longer needed with filesystem-based state tracking",
)
@click.option(
    "--force",
    is_flag=True,
    help="[DEPRECATED] No longer needed with filesystem-based state tracking",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="[DEPRECATED] No longer needed with filesystem-based state tracking",
)
@pass_cfg
def sync(workdir: Path, dry_run: bool, force: bool, no_wait: bool):
    """[DEPRECATED] Synchronize workspace database from disk state

    This command is deprecated. With the new filesystem-based state tracking,
    state is read directly from status.json and events files. No synchronization
    is needed.
    """
    cprint(
        "Warning: 'sync' command is deprecated. "
        "State is now tracked via filesystem (status.json) - no sync needed.",
        "yellow",
    )
