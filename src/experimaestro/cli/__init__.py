# flake8: noqa: T201
import sys
from typing import Set, Optional, TYPE_CHECKING
from shutil import rmtree
import click

if TYPE_CHECKING:
    from experimaestro.locking import OrphanedEventsError
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
    def __init__(self):
        self.traceback = False


def pass_cfg(f):
    """Pass configuration information"""

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj, *args, **kwargs)

    return update_wrapper(new_func, f)


def check_and_warn_stale_tokens(workdir: Path, min_age_seconds: int = 3600) -> None:
    """Check for stale token locks and print a warning if found.

    Args:
        workdir: Workspace directory
        min_age_seconds: Minimum age in seconds for a lock to be considered stale
    """
    try:
        from experimaestro.tokens import CounterToken
        import json

        tokens_dir = workdir / "tokens"
        if not tokens_dir.exists():
            return

        stale_count = 0

        for token_dir in tokens_dir.iterdir():
            if not token_dir.is_dir() or not token_dir.name.endswith(".counter"):
                continue

            token_name = token_dir.name[:-8]  # Remove .counter suffix

            try:
                # Read token total from informations.json
                info_path = token_dir / "informations.json"
                if not info_path.exists():
                    continue

                with info_path.open() as f:
                    data = json.load(f)
                    total = data.get("total", 1)

                # Create token instance to check stale locks
                token = CounterToken(token_name, token_dir, total, force=False)
                stale_locks = token.get_stale_lock_files(
                    min_age_seconds=min_age_seconds
                )
                stale_count += len(stale_locks)

            except Exception:
                # Silently ignore errors in stale token detection
                continue

        if stale_count > 0:
            cprint("\n⚠️  Warning: Found stale token locks in workspace", "yellow")
            print(
                f"  {stale_count} stale lock(s) detected (processes no longer running)"
            )
            cprint(
                "  Run 'experimaestro experiments cleanup' to remove them\n",
                "yellow",
            )

    except Exception:
        # Silently ignore errors - this is just a helpful warning
        pass


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

    # Check for stale tokens and warn (skip for cleanup command)
    if ctx.invoked_subcommand != "cleanup":
        check_and_warn_stale_tokens(path)


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
    state_provider,
    workdir: Path,
    console: bool,
    port: int,
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
    "--events-viewer",
    is_flag=True,
    help="Stream events to console (for log aggregation)",
)
@click.option(
    "--events-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Event output format (default: text, use json for tooling)",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Hide progress events in events viewer (reduces noise)",
)
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
    events_viewer: bool,
    events_format: str,
    no_progress: bool,
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

    _run_monitor_ui(
        state_provider,
        workdir,
        console,
        port,
        events_viewer=events_viewer,
        events_format=events_format,
        events_show_progress=not no_progress,
    )


@experiments.command("ssh-monitor")
@click.argument("host", type=str)
@click.argument("remote_workdir", type=str)
@click.option("--console", is_flag=True, help="Use console TUI instead of web UI")
@click.option(
    "--events-viewer",
    is_flag=True,
    help="Stream events to console (for log aggregation)",
)
@click.option(
    "--events-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Event output format (default: text, use json for tooling)",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Hide progress events in events viewer (reduces noise)",
)
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
    events_viewer: bool,
    events_format: str,
    no_progress: bool,
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
        events_viewer=events_viewer,
        events_format=events_format,
        events_show_progress=not no_progress,
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


# === History and tagging commands ===


def _get_experiment_base(workdir: Path, experiment_id: str) -> Path:
    """Get the experiment base directory, checking it exists."""
    experiment_base = workdir / "experiments" / experiment_id
    if not experiment_base.exists():
        cprint(f"Experiment '{experiment_id}' not found", "red")
        sys.exit(1)
    return experiment_base


def _read_run_status(run_dir: Path) -> dict:
    """Read status.json from a run directory."""
    import json

    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            with status_path.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_run_status(run_dir: Path, status: dict) -> None:
    """Write status.json to a run directory."""
    import json

    status_path = run_dir / "status.json"
    with status_path.open("w") as f:
        json.dump(status, f, indent=2)


def _get_run_info(run_dir: Path) -> dict:
    """Get run information from status.json and environment.json."""
    from experimaestro.utils.environment import ExperimentEnvironment
    from experimaestro.scheduler.experiment import get_run_status

    status_data = _read_run_status(run_dir)

    info = {
        "run_id": run_dir.name,
        "status": get_run_status(run_dir),
        "hostname": status_data.get("hostname"),
        "started_at": status_data.get("started_at"),
        "ended_at": status_data.get("ended_at"),
        "jobs_count": 0,
        "run_tags": status_data.get("run_tags", []),
    }

    # Get run info from environment.json if not in status
    if not info["hostname"] or not info["started_at"]:
        env_path = run_dir / "environment.json"
        if env_path.exists():
            env = ExperimentEnvironment.load(env_path)
            if env.run:
                info["hostname"] = info["hostname"] or env.run.hostname
                info["started_at"] = info["started_at"] or env.run.started_at
                info["ended_at"] = info["ended_at"] or env.run.ended_at

    # Count jobs from jobs.jsonl
    jobs_jsonl = run_dir / "jobs.jsonl"
    if jobs_jsonl.exists():
        with jobs_jsonl.open() as f:
            info["jobs_count"] = sum(1 for _ in f)

    return info


def _find_run_with_tag(experiment_base: Path, tag_name: str) -> Path | None:
    """Find the run directory that has a specific tag."""
    for run_dir in experiment_base.iterdir():
        if not run_dir.is_dir():
            continue
        status = _read_run_status(run_dir)
        if tag_name in status.get("run_tags", []):
            return run_dir
    return None


@experiments.command("history")
@click.argument("experiment_id", type=str)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_cfg
def history(workdir: Path, experiment_id: str, as_json: bool):
    """Show history of runs for an experiment

    Lists all runs with their status, start/end times, and tags.
    """
    import json

    experiment_base = _get_experiment_base(workdir, experiment_id)

    # Get all run directories sorted by name (oldest first)
    run_dirs = sorted(
        [d for d in experiment_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not run_dirs:
        cprint(f"No runs found for experiment '{experiment_id}'", "yellow")
        return

    runs = []
    for run_dir in run_dirs:
        info = _get_run_info(run_dir)
        runs.append(info)

    if as_json:
        print(json.dumps(runs, indent=2))
        return

    # Pretty print
    cprint(f"Experiment: {experiment_id}", "cyan", attrs=["bold"])
    cprint(f"{'Run ID':<20} {'Status':<12} {'Jobs':<6} {'Tags':<20} Started", "white")
    cprint("-" * 80, "white")

    for run in runs:
        status = run["status"] or "unknown"
        status_color = (
            "green"
            if status == "completed"
            else "red"
            if status == "failed"
            else "yellow"
        )
        tags_str = ", ".join(run["run_tags"]) if run["run_tags"] else ""
        started = run["started_at"][:19] if run["started_at"] else "N/A"

        print(f"{run['run_id']:<20} ", end="")
        cprint(f"{status:<12}", status_color, end=" ")
        print(f"{run['jobs_count']:<6} {tags_str:<20} {started}")


@experiments.command("history-prune")
@click.argument("experiment_id", type=str)
@click.option(
    "--before", type=str, help="Remove runs before this timestamp (YYYYMMDD_HHMMSS)"
)
@click.option(
    "--after", type=str, help="Remove runs after this timestamp (YYYYMMDD_HHMMSS)"
)
@click.option(
    "--status",
    type=click.Choice(["completed", "failed"]),
    help="Only remove runs with this status",
)
@click.option("--keep-tagged", is_flag=True, help="Keep runs that have tags")
@click.option(
    "--keep",
    type=int,
    default=None,
    help="Keep at least N most recent runs (per status if --status used)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be removed without actually removing",
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@pass_cfg
def history_prune(
    workdir: Path,
    experiment_id: str,
    before: str | None,
    after: str | None,
    status: str | None,
    keep_tagged: bool,
    keep: int | None,
    dry_run: bool,
    force: bool,
):
    """Prune (remove) old runs from an experiment's history

    Examples:
        # Remove all runs before a specific date
        experimaestro experiments history-prune my-exp --before 20250101_000000

        # Remove all failed runs, keeping 2 most recent
        experimaestro experiments history-prune my-exp --status failed --keep 2

        # Remove runs between two dates (dry run)
        experimaestro experiments history-prune my-exp --before 20250201 --after 20250101 --dry-run
    """
    from experimaestro.scheduler.experiment import get_run_status

    experiment_base = _get_experiment_base(workdir, experiment_id)

    # Get all run directories
    run_dirs = sorted(
        [d for d in experiment_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # Filter runs to remove
    to_remove = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        run_status = get_run_status(run_dir)

        # Check status filter
        if status and run_status != status:
            continue

        # Check timestamp filters (compare as strings since format is consistent)
        if before and run_id >= before:
            continue
        if after and run_id <= after:
            continue

        # Check tag filter
        if keep_tagged:
            status_data = _read_run_status(run_dir)
            if status_data.get("run_tags", []):
                continue

        to_remove.append(run_dir)

    # Apply --keep filter (keep N most recent)
    if keep is not None and len(to_remove) > 0:
        # Sort by name descending (newest first)
        to_remove_sorted = sorted(to_remove, key=lambda d: d.name, reverse=True)
        # Keep the first N (newest)
        to_remove = to_remove_sorted[keep:]

    if not to_remove:
        cprint("No runs match the criteria for removal", "yellow")
        return

    # Show what will be removed
    cprint(f"Runs to remove from '{experiment_id}':", "cyan")
    for run_dir in to_remove:
        run_status = get_run_status(run_dir) or "unknown"
        status_data = _read_run_status(run_dir)
        run_tags = status_data.get("run_tags", [])
        tags_str = f" [tags: {', '.join(run_tags)}]" if run_tags else ""
        cprint(f"  {run_dir.name} ({run_status}){tags_str}", "white")

    if dry_run:
        cprint(f"\nDry run: would remove {len(to_remove)} run(s)", "yellow")
        return

    # Confirm unless forced
    if not force:
        if not click.confirm(f"\nRemove {len(to_remove)} run(s)?"):
            cprint("Aborted", "yellow")
            return

    # Remove runs
    removed = 0
    for run_dir in to_remove:
        try:
            rmtree(run_dir)
            removed += 1
            cprint(f"  Removed: {run_dir.name}", "green")
        except Exception as e:
            cprint(f"  Failed to remove {run_dir.name}: {e}", "red")

    cprint(f"\nRemoved {removed}/{len(to_remove)} run(s)", "cyan")


@experiments.command("tag")
@click.argument("experiment_id", type=str)
@click.argument("run_id", type=str)
@click.argument("tag_name", type=str)
@click.option("--force", is_flag=True, help="Overwrite existing tag")
@pass_cfg
def tag(workdir: Path, experiment_id: str, run_id: str, tag_name: str, force: bool):
    """Tag an experiment run

    Tags provide human-readable names for specific runs. A tag name can only
    point to one run at a time.

    Examples:
        # Tag a specific run as 'baseline'
        experimaestro experiments tag my-exp 20250115_103045 baseline

        # Tag the latest run as 'best'
        experimaestro experiments tag my-exp latest best

    Special run IDs:
        latest  - The most recent run
        oldest  - The oldest run
    """
    experiment_base = _get_experiment_base(workdir, experiment_id)

    # Handle special run IDs
    if run_id == "latest":
        run_dirs = sorted(
            [d for d in experiment_base.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            cprint(f"No runs found for experiment '{experiment_id}'", "red")
            sys.exit(1)
        run_id = run_dirs[0].name
        cprint(f"Resolved 'latest' to {run_id}", "white")
    elif run_id == "oldest":
        run_dirs = sorted(
            [d for d in experiment_base.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not run_dirs:
            cprint(f"No runs found for experiment '{experiment_id}'", "red")
            sys.exit(1)
        run_id = run_dirs[0].name
        cprint(f"Resolved 'oldest' to {run_id}", "white")

    # Verify run exists
    run_dir = experiment_base / run_id
    if not run_dir.exists():
        cprint(f"Run '{run_id}' not found", "red")
        sys.exit(1)

    # Check if tag already exists on another run
    existing_run = _find_run_with_tag(experiment_base, tag_name)
    if existing_run:
        if existing_run == run_dir:
            cprint(f"Tag '{tag_name}' already on {run_id}", "yellow")
            return
        if not force:
            cprint(
                f"Tag '{tag_name}' already exists on {existing_run.name}. "
                "Use --force to move it.",
                "red",
            )
            sys.exit(1)
        # Remove tag from existing run
        old_status = _read_run_status(existing_run)
        old_tags = old_status.get("run_tags", [])
        if tag_name in old_tags:
            old_tags.remove(tag_name)
            old_status["run_tags"] = old_tags
            _write_run_status(existing_run, old_status)
        cprint(f"Moved tag '{tag_name}' from {existing_run.name}", "yellow")

    # Add tag to target run
    status = _read_run_status(run_dir)
    tags = status.get("run_tags", [])
    if tag_name not in tags:
        tags.append(tag_name)
        status["run_tags"] = tags
        _write_run_status(run_dir, status)
    cprint(f"Tagged {run_id} as '{tag_name}'", "green")


@experiments.command("untag")
@click.argument("experiment_id", type=str)
@click.argument("tag_name", type=str)
@pass_cfg
def untag(workdir: Path, experiment_id: str, tag_name: str):
    """Remove a tag from an experiment

    Examples:
        experimaestro experiments untag my-exp baseline
    """
    experiment_base = _get_experiment_base(workdir, experiment_id)

    # Find run with this tag
    run_dir = _find_run_with_tag(experiment_base, tag_name)
    if not run_dir:
        cprint(f"Tag '{tag_name}' not found", "red")
        sys.exit(1)

    # Remove tag
    status = _read_run_status(run_dir)
    tags = status.get("run_tags", [])
    if tag_name in tags:
        tags.remove(tag_name)
        status["run_tags"] = tags
        _write_run_status(run_dir, status)
    cprint(f"Removed tag '{tag_name}' (was on {run_dir.name})", "green")


@experiments.command("tags")
@click.argument("experiment_id", type=str)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_cfg
def tags_list(workdir: Path, experiment_id: str, as_json: bool):
    """List all tags for an experiment

    Examples:
        experimaestro experiments tags my-exp
    """
    import json

    experiment_base = _get_experiment_base(workdir, experiment_id)

    # Collect all tags from all runs
    tags_map = {}  # tag_name -> run_id
    for run_dir in experiment_base.iterdir():
        if not run_dir.is_dir():
            continue
        status = _read_run_status(run_dir)
        for tag_name in status.get("run_tags", []):
            tags_map[tag_name] = run_dir.name

    if not tags_map:
        cprint(f"No tags found for experiment '{experiment_id}'", "yellow")
        return

    if as_json:
        print(json.dumps(tags_map, indent=2))
        return

    cprint(f"Tags for experiment '{experiment_id}':", "cyan")
    for tag_name, run_id in sorted(tags_map.items()):
        print(f"  {tag_name:<20} -> {run_id}")


def _handle_orphaned_events_in_cli(error: "OrphanedEventsError", force: bool) -> None:
    """Handle OrphanedEventsError in CLI by prompting user for action.

    Args:
        error: The OrphanedEventsError exception
        force: If True, automatically choose "skip" action
    """
    cprint("\n⚠️  Orphaned Events Detected", "yellow", attrs=["bold"])
    print()
    print(error.description)
    print()

    # If force mode, auto-select "skip" option
    if force:
        cprint("Force mode: automatically skipping orphaned events", "yellow")
        action_key = "skip"
    else:
        # Prompt user for action
        cprint("Available actions:", "cyan")
        for key, label in error.actions.items():
            print(f"  [{key}] {label}")
        print()

        # Get user choice
        while True:
            action_key = click.prompt(
                "Select action",
                type=click.Choice(list(error.actions.keys())),
                default="skip",
            )
            break

    # Execute the chosen callback
    callback = error.callbacks.get(action_key)
    if callback:
        try:
            callback()
            cprint(f"✓ Action '{action_key}' completed successfully", "green")
        except Exception as e:
            cprint(f"✗ Action '{action_key}' failed: {e}", "red")
            raise
    else:
        cprint(f"No callback for action '{action_key}'", "yellow")


@cli.command("cleanup")
@click.option("--workdir", type=Path, default=None)
@click.option("--workspace", type=str, default=None)
def cleanup(workdir: Path | None, workspace: str | None):
    """Perform comprehensive workspace cleanup

    Scans the workspace for cleanup issues and presents them interactively:
    - Orphaned experiment events (from crashes)
    - Orphaned job events (from crashes)
    - Stray jobs (running jobs not in latest run)
    - Orphan jobs (finished jobs not in any run)
    - Orphan partials (unused checkpoint directories)

    Safe issues (orphaned events with events_count) are fixed automatically.
    Issues requiring user confirmation are presented interactively.

    Running experiments are silently skipped (not reported).

    Examples:

    \b
    # Run comprehensive workspace cleanup
    experimaestro cleanup
    """
    from experimaestro.scheduler.cleanup import perform_workspace_cleanup
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    # Resolve workspace
    ws = find_workspace(workdir=workdir, workspace=workspace)
    workdir = check_xp_path(None, None, ws.path)

    cprint("=== Workspace Cleanup ===", "cyan", attrs=["bold"])
    print()

    # Get or create workspace state provider (without automatic cleanup)
    cprint("Initializing workspace...", "cyan")
    provider = WorkspaceStateProvider.get_instance(workdir, no_cleanup=True)

    # Perform cleanup using the unified cleanup function
    cprint("Scanning workspace for cleanup issues...", "cyan")
    warnings, callbacks = perform_workspace_cleanup(
        workdir, auto_fix=True, provider=provider
    )

    if not warnings:
        cprint("✓ No cleanup issues found.", "green")
        return

    # Present warnings interactively
    cprint(
        f"\nFound {len(warnings)} issue(s) requiring confirmation:\n",
        "yellow",
        attrs=["bold"],
    )

    for idx, warning in enumerate(warnings, 1):
        title = warning.context.get("title", "Warning")
        cprint(f"[{idx}] {title}", "yellow", attrs=["bold"])
        print()
        print(warning.description)
        print()

        # Show actions
        if not warning.actions:
            continue

        # Get callbacks for this warning
        warning_callbacks = callbacks.get(warning.warning_key, {})

        # Ask user which action to take
        action_keys = list(warning.actions.keys())
        action_labels = list(warning.actions.values())

        print("  Choose an action:")
        for i, (action_key, action_label) in enumerate(
            zip(action_keys, action_labels), 1
        ):
            print(f"    [{i}] {action_label}")
        print()

        # Get user choice
        while True:
            try:
                choice_str = input(f"  Enter choice [1-{len(action_keys)}]: ").strip()
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(action_keys):
                    break
                else:
                    cprint(
                        f"  Invalid choice. Please enter 1-{len(action_keys)}.", "red"
                    )
            except (ValueError, EOFError, KeyboardInterrupt):
                cprint("\n  Skipping this warning.", "yellow")
                choice_idx = None
                break

        if choice_idx is not None:
            action_key = action_keys[choice_idx]
            action_label = action_labels[choice_idx]
            callback = warning_callbacks.get(action_key)

            if callback is not None:
                try:
                    cprint(f"\n  Executing: {action_label}...", "cyan")
                    callback()
                    cprint(f"  ✓ {action_label} completed successfully", "green")
                except Exception as e:
                    cprint(f"  ✗ {action_label} failed: {e}", "red")
            else:
                cprint(f"  Skipping (no callback for {action_key})", "yellow")

        print()

    cprint("Cleanup completed.", "green")


# === Carbon tracking commands ===


@experiments.group()
def carbon():
    """Carbon tracking commands"""
    pass


@carbon.command("summary")
@click.option(
    "--name",
    "-n",
    "name_pattern",
    type=str,
    default=None,
    help="Filter experiments by name (regex pattern)",
)
@click.option(
    "--tag",
    "-t",
    "tags",
    multiple=True,
    help="Filter experiments by run tag (can be repeated)",
)
@click.option(
    "--latest-only",
    is_flag=True,
    help="Only count the latest run of each job (avoids counting retries)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def _collect_carbon_job_ids(
    experiments_dir: Path,
    name_re,
    tags_set: set | None,
    verbose: bool,
) -> tuple[set[str], dict[str, set[str]]]:
    """Collect job IDs from experiments matching filters.

    Returns:
        Tuple of (all_job_ids, experiment_jobs dict for verbose output)
    """
    all_job_ids: set[str] = set()
    experiment_jobs: dict[str, set[str]] = {}

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_id = exp_dir.name
        if name_re and not name_re.search(exp_id):
            continue

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            if tags_set:
                status = _read_run_status(run_dir)
                run_tags = set(status.get("run_tags", []))
                if not tags_set.intersection(run_tags):
                    continue

            jobs_path = run_dir / "jobs.jsonl"
            if not jobs_path.exists():
                continue

            _read_job_ids_from_jsonl(
                jobs_path, exp_id, all_job_ids, experiment_jobs, verbose
            )

    return all_job_ids, experiment_jobs


def _read_job_ids_from_jsonl(
    jobs_path: Path,
    exp_id: str,
    all_job_ids: set[str],
    experiment_jobs: dict[str, set[str]],
    verbose: bool,
) -> None:
    """Read job IDs from a jobs.jsonl file."""
    import json

    try:
        with jobs_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    job_data = json.loads(line)
                    job_id = job_data.get("identifier") or job_data.get("job_id")
                    if job_id:
                        all_job_ids.add(job_id)
                        if verbose:
                            experiment_jobs.setdefault(exp_id, set()).add(job_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug("Failed to read jobs from %s: %s", jobs_path, e)


def _print_carbon_summary(
    totals: dict,
    all_job_ids: set[str],
    name_pattern: str | None,
    tags: tuple,
    latest_only: bool,
    experiment_breakdown: dict,
    verbose: bool,
) -> None:
    """Pretty print carbon summary."""
    from experimaestro.carbon.utils import format_co2_kg, format_energy_kwh

    cprint("Carbon Impact Summary", "cyan", attrs=["bold"])
    cprint("-" * 50, "white")

    if name_pattern:
        print(f"  Name filter: {name_pattern}")
    if tags:
        print(f"  Tag filter: {', '.join(tags)}")
    if latest_only:
        print("  Mode: Latest run only (excluding retries)")
    else:
        print("  Mode: All runs (including retries)")
    print()

    co2_str = format_co2_kg(totals["co2_kg"])
    energy_str = format_energy_kwh(totals["energy_kwh"])
    duration_h = totals["duration_s"] / 3600

    cprint("Total Emissions:", "green", attrs=["bold"])
    print(f"  CO2:      {co2_str}")
    print(f"  Energy:   {energy_str}")
    print(f"  Duration: {duration_h:.1f} hours")
    print(f"  Jobs:     {totals['job_count']} (unique: {len(all_job_ids)})")

    if verbose and experiment_breakdown:
        print()
        cprint("Per-Experiment Breakdown:", "cyan")
        for exp_id, exp_totals in sorted(experiment_breakdown.items()):
            if exp_totals["job_count"] > 0:
                exp_co2 = format_co2_kg(exp_totals["co2_kg"])
                print(f"  {exp_id}: {exp_co2} ({exp_totals['job_count']} jobs)")


@click.option("--verbose", "-v", is_flag=True, help="Show per-experiment breakdown")
@pass_cfg
def carbon_summary(
    workdir: Path,
    name_pattern: str | None,
    tags: tuple,
    latest_only: bool,
    as_json: bool,
    verbose: bool,
):
    """Aggregate CO2 emissions across experiments

    This command aggregates carbon metrics from the carbon storage.
    Jobs are counted only once even if they appear in multiple experiments.

    Examples:
        # Total CO2 for all experiments
        experimaestro experiments carbon summary

        # Filter by experiment name pattern
        experimaestro experiments carbon summary --name "^train-.*"

        # Filter by run tags
        experimaestro experiments carbon summary --tag baseline --tag production

        # Only count latest job runs (exclude retries)
        experimaestro experiments carbon summary --latest-only

        # JSON output with per-experiment breakdown
        experimaestro experiments carbon summary --json --verbose
    """
    import re
    import json

    from experimaestro.carbon.storage import CarbonStorage

    experiments_dir = workdir / "experiments"
    if not experiments_dir.exists():
        cprint("No experiments found", "yellow")
        return

    name_re = re.compile(name_pattern) if name_pattern else None
    tags_set = set(tags) if tags else None

    all_job_ids, experiment_jobs = _collect_carbon_job_ids(
        experiments_dir, name_re, tags_set, verbose
    )

    if not all_job_ids:
        cprint("No jobs found matching the filters", "yellow")
        return

    storage = CarbonStorage(workdir)
    totals = storage.aggregate_for_jobs(all_job_ids, use_latest_only=latest_only)

    experiment_breakdown = {}
    if verbose:
        for exp_id, job_ids in experiment_jobs.items():
            experiment_breakdown[exp_id] = storage.aggregate_for_jobs(
                job_ids, use_latest_only=latest_only
            )

    if as_json:
        output = {
            "totals": totals,
            "filters": {
                "name_pattern": name_pattern,
                "tags": list(tags),
                "latest_only": latest_only,
            },
            "unique_jobs": len(all_job_ids),
        }
        if verbose:
            output["experiments"] = experiment_breakdown
        print(json.dumps(output, indent=2))
        return

    _print_carbon_summary(
        totals,
        all_job_ids,
        name_pattern,
        tags,
        latest_only,
        experiment_breakdown,
        verbose,
    )


@carbon.command("stats")
@pass_cfg
def carbon_stats(workdir: Path):
    """Show carbon storage statistics

    Displays information about the carbon storage including
    total records, file count, and storage size.
    """
    from experimaestro.carbon.storage import CarbonStorage

    storage = CarbonStorage(workdir)
    stats = storage.get_stats()

    if stats["total_records"] == 0:
        cprint("No carbon records found", "yellow")
        return

    cprint("Carbon Storage Statistics", "cyan", attrs=["bold"])
    cprint("-" * 40, "white")
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  File count:    {stats['file_count']}")

    # Format size
    size_bytes = stats["size_bytes"]
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    print(f"  Storage size:  {size_str}")
