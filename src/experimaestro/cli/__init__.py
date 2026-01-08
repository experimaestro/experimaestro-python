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
    "--traceback", is_flag=True, help="Display traceback if an exception occurs"
)
@click.pass_context
def cli(ctx, quiet, debug, traceback):
    if quiet:
        logging.getLogger().setLevel(logging.WARN)
    elif debug:
        logging.getLogger().setLevel(logging.DEBUG)

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


@click.option("--show-all", is_flag=True, help="Show even not orphans")
@click.option(
    "--ignore-old", is_flag=True, help="Ignore old jobs for unfinished experiments"
)
@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.argument("path", type=Path, callback=check_xp_path)
@cli.command()
def orphans(path: Path, clean: bool, size: bool, show_all: bool, ignore_old: bool):
    """Check for tasks that are not part of an experimental plan"""

    jobspath = path / "jobs"

    def getjobs(path: Path):
        return ((str(p.relative_to(path)), p) for p in path.glob("*/*") if p.is_dir())

    def show(key: str, prefix=""):
        if size:
            print(
                prefix,
                subprocess.check_output(["du", "-hs", key], cwd=jobspath)
                .decode("utf-8")
                .strip(),
                sep=None,
            )
        else:
            print(prefix, key, sep=None)

    # New layout: experiments/{exp-id}/{run-id}/jobs
    # Retrieve the jobs within experiments
    xpjobs = set()
    paths = (path / "experiments").glob("*/*/jobs")

    for p in paths:
        if p.is_dir():
            for relpath, path in getjobs(p):
                xpjobs.add(relpath)

    # Now, look at stored jobs
    found = 0
    for key, jobpath in getjobs(jobspath):
        if key not in xpjobs:
            show(key)
            if clean:
                logging.info("Removing data in %s", jobpath)
                rmtree(jobpath)
        else:
            if show_all:
                show(key, prefix="[not orphan] ")
            found += 1

    print(f"{found} jobs are not orphans")


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

# Import and add progress commands
from .progress import progress as progress_cli  # noqa: E402

cli.add_command(progress_cli)

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
@click.option(
    "--keep-old", is_flag=True, help="Keep the old xp directory after migration"
)
def migrate_v1_to_v2(workdir: Path, dry_run: bool, keep_old: bool):
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
            if keep_old:
                # Keep remaining files, rename directory
                renamed_xp_dir = workdir / "xp_MIGRATED_TO_V2"
                old_xp_dir.rename(renamed_xp_dir)
                cprint(
                    f"Renamed 'xp' -> 'xp_MIGRATED_TO_V2' ({len(remaining)} item(s))",
                    "yellow",
                )
            else:
                cprint(
                    f"'xp' directory still contains {len(remaining)} item(s), not removing",
                    "yellow",
                )
                cprint("Remove manually or use --keep-old to rename", "yellow")
                return
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


@experiments.command()
@pass_cfg
def list(workdir: Path):
    """List experiments in the workspace"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    # Get experiments from state provider for detailed info
    state_provider = DbStateProvider.get_instance(
        workdir, read_only=True, sync_on_start=True
    )
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
    "--sync", is_flag=True, help="Force sync from disk before starting monitor"
)
@pass_cfg
def monitor(workdir: Path, console: bool, port: int, sync: bool):
    """Monitor local experiments with web UI or console TUI"""
    # Force sync from disk if requested
    if sync:
        from experimaestro.scheduler.state_sync import sync_workspace_from_disk

        cprint("Syncing workspace from disk...", "yellow")
        sync_workspace_from_disk(workdir, write_mode=True, force=True)
        cprint("Sync complete", "green")

    from experimaestro.scheduler.db_state_provider import DbStateProvider

    state_provider = DbStateProvider.get_instance(
        workdir,
        sync_on_start=not sync,  # Skip auto-sync if we just did a forced one
    )

    _run_monitor_ui(state_provider, workdir, console, port)


@experiments.command("ssh-monitor")
@click.argument("host", type=str)
@click.argument("remote_workdir", type=str)
@click.option("--console", is_flag=True, help="Use console TUI instead of web UI")
@click.option(
    "--port", type=int, default=12345, help="Port for web server (default: 12345)"
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
    help="Don't write to database, only show what would be synced",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force sync even if recently synced (bypasses time throttling)",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Don't wait for lock, fail immediately if unavailable",
)
@pass_cfg
def sync(workdir: Path, dry_run: bool, force: bool, no_wait: bool):
    """Synchronize workspace database from disk state

    Scans experiment directories and job marker files to update the workspace
    database. Uses exclusive locking to prevent conflicts with running experiments.
    """
    from experimaestro.scheduler.state_sync import sync_workspace_from_disk
    from experimaestro.scheduler.workspace import Workspace
    from experimaestro.settings import Settings

    # Get settings and workspace settings
    settings = Settings.instance()
    ws_settings = find_workspace(workdir=workdir)

    # Create workspace instance (manages database lifecycle)
    workspace = Workspace(
        settings=settings,
        workspace_settings=ws_settings,
        sync_on_init=False,  # Don't sync on init since we're explicitly syncing
    )

    try:
        # Enter workspace context to initialize database
        with workspace:
            cprint(f"Syncing workspace: {workspace.path}", "cyan")
            if dry_run:
                cprint("DRY RUN MODE: No changes will be written", "yellow")
            if force:
                cprint("FORCE MODE: Bypassing time throttling", "yellow")

            # Run sync
            sync_workspace_from_disk(
                workspace=workspace,
                write_mode=not dry_run,
                force=force,
                blocking=not no_wait,
            )

            cprint("Sync completed successfully", "green")

    except RuntimeError as e:
        cprint(f"Sync failed: {e}", "red")
        sys.exit(1)
    except Exception as e:
        cprint(f"Unexpected error during sync: {e}", "red")
        import traceback

        traceback.print_exc()
        sys.exit(1)
