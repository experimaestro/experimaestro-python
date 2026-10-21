# flake8: noqa: T201
"""``experimaestro experiments``: experiments, their runs, services and actions"""

import logging
import sys
from pathlib import Path
from shutil import rmtree

import click
from termcolor import cprint

from experimaestro.settings import get_settings

from . import RunConfig, deprecated_alias, pass_workdir
from .dag import dag
from .monitor import experiments_monitor, ssh_monitor

logger = logging.getLogger(__name__)


@click.group()
@click.option("--workdir", type=Path, default=None, hidden=True)
@click.option("--workspace", type=str, default=None, hidden=True)
@click.pass_obj
def experiments(cfg: RunConfig, workdir: Path | None, workspace: str | None):
    """Manage experiments, their runs, services and actions (alias: xp)"""
    # Kept for compatibility: the remote monitor calls
    # `experiments --workdir PATH monitor-server`
    cfg.set_workspace(workdir, workspace)


experiments.add_command(dag, "dag")


@experiments.command("list")
@pass_workdir
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


@experiments.command("monitor-server", hidden=True)
@pass_workdir
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


# === Runs: history and tags ===


@experiments.group()
def runs():
    """Runs of an experiment: history, pruning and tags"""


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
    from experimaestro.utils.atomic import atomic_write_json

    atomic_write_json(run_dir / "status.json", status, indent=2)


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


@runs.command("list")
@click.argument("experiment_id", type=str)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_workdir
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


# === Services commands ===


def _service_state_name(service) -> str:
    state = getattr(service, "state", None)
    return state.name if hasattr(state, "name") else str(state)


def _as_live_service(service):
    """Best-effort reconstruction of a live service from a (mock) service."""
    return service.to_service() if hasattr(service, "to_service") else service


def _load_services_by_id(workdir: Path, experiment_id, run_id) -> dict:
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    provider = WorkspaceStateProvider.get_instance(workdir)
    services = provider.get_services(experiment_id, run_id)
    return {service.id: service for service in services}


def _iter_service_rows(services_by_id: dict):
    """Yield ``(id, description, state, url)`` rows, expanding sub-services.

    Each sub-service is addressed as ``<service>/<subservice>``, except the
    sub-service that *is* the service itself (a service is its own sub-service by
    default), which keeps the bare ``<service>`` id. So a plain service yields a
    single ``<service>`` row, while a composite that also offers, say, a wandb
    sub-service yields ``<service>`` plus ``<service>/wandb``.
    """
    for sid in sorted(services_by_id):
        live = _as_live_service(services_by_id[sid])
        subs = live.subservices() if hasattr(live, "subservices") else [live]
        for sub in subs:
            addr = sid if sub is live else f"{sid}/{sub.id}"
            yield (
                addr,
                sub.description(),
                _service_state_name(sub),
                getattr(sub, "url", None),
            )


@experiments.group("services")
def services_group():
    """Inspect and start experiment services (and their sub-services)."""


@services_group.command("list")
@click.argument("experiment_id", required=False, default=None)
@click.option("--run", "run_id", default=None, help="Run id (default: current run)")
@pass_workdir
def services_list(workdir: Path, experiment_id, run_id):
    """List services and sub-services with their state."""
    rows = list(
        _iter_service_rows(_load_services_by_id(workdir, experiment_id, run_id))
    )
    if not rows:
        cprint("No services found", "yellow")
        return

    cprint(f"{'SERVICE':<32} {'STATE':<10} {'DESCRIPTION':<28} URL", "white")
    cprint("-" * 90, "white")
    for sid, desc, state, url in rows:
        print(f"{sid:<32} {state:<10} {(desc or ''):<28} {url or ''}")


@services_group.command("start")
@click.argument("service_id")
@click.argument("experiment_id", required=False, default=None)
@click.option("--run", "run_id", default=None, help="Run id (default: current run)")
@pass_workdir
def services_start(workdir: Path, service_id: str, experiment_id, run_id):
    """Start a service or sub-service, e.g. ``monitoring/wandb``.

    Blocks while the service runs (Ctrl-C to stop). A service that completes on
    its own (e.g. a one-shot W&B sync of a finished experiment) returns when done.
    """
    import time
    from experimaestro.scheduler.services import ServiceState

    services_by_id = _load_services_by_id(workdir, experiment_id, run_id)
    parent_id, _, sub_id = service_id.partition("/")
    if parent_id not in services_by_id:
        cprint(f"Service '{parent_id}' not found", "red")
        sys.exit(1)

    live = _as_live_service(services_by_id[parent_id])
    try:
        target = live.get_subservice(sub_id) if sub_id else live
    except (KeyError, AttributeError):
        cprint(f"Sub-service '{sub_id}' not found in '{parent_id}'", "red")
        sys.exit(1)

    if not hasattr(target, "get_url"):
        cprint(f"Service '{service_id}' cannot be started", "red")
        sys.exit(1)

    # Services normally get their log directory from set_experiment(). When
    # starting a reconstructed service offline — or one persisted before
    # log_directory was stored (older experiments) — provide a temporary one so
    # it can write its stdout/stderr.
    if getattr(target, "log_directory", None) is None:
        import tempfile

        target.log_directory = Path(
            tempfile.mkdtemp(prefix=f"xpm-service-{target.id}-")
        )

    cprint(f"Starting {service_id}...", "cyan")
    url = target.get_url()
    if url:
        cprint(f"  URL: {url}", "green")

    try:
        while target.state in (ServiceState.STARTING, ServiceState.RUNNING):
            time.sleep(0.5)
    except KeyboardInterrupt:
        cprint("\nStopping...", "yellow")
    finally:
        if hasattr(target, "stop") and target.state != ServiceState.STOPPED:
            target.stop()


@runs.command("prune")
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
@pass_workdir
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
        experimaestro experiments runs prune my-exp --before 20250101_000000

        # Remove all failed runs, keeping 2 most recent
        experimaestro experiments runs prune my-exp --status failed --keep 2

        # Remove runs between two dates (dry run)
        experimaestro experiments runs prune my-exp --before 20250201 --after 20250101 --dry-run
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


@runs.command("tag")
@click.argument("experiment_id", type=str)
@click.argument("run_id", type=str)
@click.argument("tag_name", type=str)
@click.option("--force", is_flag=True, help="Overwrite existing tag")
@pass_workdir
def tag(workdir: Path, experiment_id: str, run_id: str, tag_name: str, force: bool):
    """Tag an experiment run

    Tags provide human-readable names for specific runs. A tag name can only
    point to one run at a time.

    Examples:
        # Tag a specific run as 'baseline'
        experimaestro experiments runs tag my-exp 20250115_103045 baseline

        # Tag the latest run as 'best'
        experimaestro experiments runs tag my-exp latest best

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


@runs.command("untag")
@click.argument("experiment_id", type=str)
@click.argument("tag_name", type=str)
@pass_workdir
def untag(workdir: Path, experiment_id: str, tag_name: str):
    """Remove a tag from an experiment

    Examples:
        experimaestro experiments runs untag my-exp baseline
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


@runs.command("tags")
@click.argument("experiment_id", type=str)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_workdir
def tags_list(workdir: Path, experiment_id: str, as_json: bool):
    """List all tags for an experiment

    Examples:
        experimaestro experiments runs tags my-exp
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


# === Carbon tracking commands ===


@experiments.group()
def carbon():
    """Carbon tracking commands"""
    pass


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
@click.option("--verbose", "-v", is_flag=True, help="Show per-experiment breakdown")
@pass_workdir
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
@pass_workdir
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


# === Copy command ===


@experiments.command("copy")
@click.argument("experiment_id", type=str, required=False, default=None)
@click.argument("run_id", type=str, required=False, default=None)
@click.option(
    "--from-workspace", "from_ws", type=str, default=None, help="Source workspace ID"
)
@click.option(
    "--to-workspace", "to_ws", type=str, default=None, help="Destination workspace ID"
)
@click.option("--yes", "-y", is_flag=True, help="Actually copy (default is dry-run)")
def copy_experiment_cmd(experiment_id, run_id, from_ws, to_ws, yes):
    """Copy an experiment between workspaces.

    Copies the experiment run directory and all referenced job directories
    from one workspace to another. Supports local and remote (SSH) workspaces.

    If EXPERIMENT_ID or RUN_ID are not given, prompts for interactive selection.
    """
    from rich.console import Console
    from rich.prompt import IntPrompt

    from experimaestro.cli.copy_experiment import (
        copy_experiment,
        format_run_id,
        list_experiments as list_ws_experiments,
        list_runs,
        resolve_current_run,
    )
    from experimaestro.settings import get_workspace

    console = Console()
    settings = get_settings()
    workspaces = settings.workspaces

    if not workspaces:
        cprint("No workspaces configured in settings.yaml", "red")
        sys.exit(1)

    # --- Resolve source workspace ---
    if from_ws:
        src_ws = get_workspace(from_ws, include_remote=True)
        if src_ws is None:
            cprint(f"Source workspace '{from_ws}' not found", "red")
            sys.exit(1)
    else:
        console.print("\n[bold]Select source workspace:[/bold]")
        for i, ws in enumerate(workspaces, 1):
            remote_tag = " [dim](remote)[/dim]" if ws.is_remote else ""
            console.print(
                f"  [cyan]{i}[/cyan]) {ws.id}{remote_tag}  [dim]{ws._raw_path}[/dim]"
            )
        idx = IntPrompt.ask(
            "\n[dim]Select[/dim]",
            choices=[str(i) for i in range(1, len(workspaces) + 1)],
            show_choices=False,
        )
        src_ws = workspaces[idx - 1]

    # --- Resolve destination workspace ---
    if to_ws:
        dst_ws = get_workspace(to_ws, include_remote=True)
        if dst_ws is None:
            cprint(f"Destination workspace '{to_ws}' not found", "red")
            sys.exit(1)
    else:
        # Filter out source workspace
        dst_candidates = [ws for ws in workspaces if ws.id != src_ws.id]
        if not dst_candidates:
            cprint("No other workspaces available as destination", "red")
            sys.exit(1)
        console.print("\n[bold]Select destination workspace:[/bold]")
        for i, ws in enumerate(dst_candidates, 1):
            remote_tag = " [dim](remote)[/dim]" if ws.is_remote else ""
            console.print(
                f"  [cyan]{i}[/cyan]) {ws.id}{remote_tag}  [dim]{ws._raw_path}[/dim]"
            )
        idx = IntPrompt.ask(
            "\n[dim]Select[/dim]",
            choices=[str(i) for i in range(1, len(dst_candidates) + 1)],
            show_choices=False,
        )
        dst_ws = dst_candidates[idx - 1]

    # --- Resolve experiment ID ---
    if experiment_id is None:
        exp_list = list_ws_experiments(src_ws)
        if not exp_list:
            cprint("No experiments found in source workspace", "yellow")
            sys.exit(1)
        console.print("\n[bold]Select experiment:[/bold]")
        for i, eid in enumerate(exp_list, 1):
            console.print(f"  [cyan]{i}[/cyan]) {eid}")
        idx = IntPrompt.ask(
            "\n[dim]Select[/dim]",
            choices=[str(i) for i in range(1, len(exp_list) + 1)],
            show_choices=False,
        )
        experiment_id = exp_list[idx - 1]

    # --- Resolve run ID ---
    if run_id is None:
        current = resolve_current_run(src_ws, experiment_id)
        runs = list_runs(src_ws, experiment_id)
        if not runs:
            if current:
                run_id = current
                console.print(
                    f"Using current run: [cyan]{format_run_id(run_id)}[/cyan]"
                )
            else:
                cprint(f"No runs found for experiment '{experiment_id}'", "yellow")
                sys.exit(1)
        elif len(runs) == 1:
            run_id = runs[0]
            console.print(
                f"Using only available run: [cyan]{format_run_id(run_id)}[/cyan]"
            )
        else:
            console.print("\n[bold]Select run:[/bold]")
            for i, rid in enumerate(runs, 1):
                label = format_run_id(rid)
                current_tag = " [green](current)[/green]" if rid == current else ""
                console.print(
                    f"  [cyan]{i}[/cyan]) {label}{current_tag}  [dim]{rid}[/dim]"
                )
            idx = IntPrompt.ask(
                "\n[dim]Select[/dim]",
                choices=[str(i) for i in range(1, len(runs) + 1)],
                show_choices=False,
            )
            run_id = runs[idx - 1]

    # --- Execute copy ---
    dry_run = not yes
    console.print(
        f"\nCopying experiment [bold]{experiment_id}[/bold] run [bold]{run_id}[/bold]"
        f"\n  from [cyan]{src_ws.id}[/cyan] ({src_ws._raw_path})"
        f"\n  to   [cyan]{dst_ws.id}[/cyan] ({dst_ws._raw_path})"
    )
    if dry_run:
        console.print("[yellow]  (dry run — use --yes to actually copy)[/yellow]")

    result = copy_experiment(src_ws, dst_ws, experiment_id, run_id, dry_run=dry_run)

    # --- Print summary ---
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Jobs copied:  {result.jobs_copied}")
    console.print(f"  Jobs skipped: {result.jobs_skipped}")
    if result.errors:
        console.print(f"  [red]Errors: {len(result.errors)}[/red]")
        for err in result.errors:
            console.print(f"    [red]- {err}[/red]")
        sys.exit(1)
    else:
        console.print("[green]Copy completed successfully.[/green]")


# === Action commands (alpha) ===


@experiments.group("actions")
def actions_group():
    """Manage experiment actions (alpha feature)"""
    pass


def _select_experiment(state_provider, experiment_id: str | None):
    """Select an experiment, prompting interactively if not specified.

    Returns the selected experiment or None if not found/cancelled.
    """
    from rich.console import Console
    from rich.prompt import IntPrompt

    experiments_list = list(state_provider.get_experiments())
    if not experiments_list:
        cprint("No experiments found in workspace", "yellow")
        return None

    if experiment_id is not None:
        matching = [
            exp for exp in experiments_list if exp.experiment_id == experiment_id
        ]
        if not matching:
            cprint(f"Experiment '{experiment_id}' not found", "red")
            return None
        return matching[0]

    # Interactive selection
    console = Console()
    console.print("\n[bold]Select an experiment:[/bold]")
    for i, exp in enumerate(experiments_list, 1):
        status = exp.status.value if hasattr(exp.status, "value") else "?"
        console.print(f"  [cyan]{i}[/cyan]) {exp.experiment_id}  [{status}]")

    idx = IntPrompt.ask(
        "\n[dim]Select[/dim]",
        choices=[str(i) for i in range(1, len(experiments_list) + 1)],
        show_choices=False,
    )
    return experiments_list[idx - 1]


@actions_group.command("list")
@click.argument("experiment_id", type=str, required=False, default=None)
@pass_workdir
def list_actions(workdir: Path, experiment_id: str | None):
    """List actions for an experiment.

    If EXPERIMENT_ID is not given, prompts for interactive selection.
    """
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    state_provider = WorkspaceStateProvider(workdir, no_cleanup=True)
    try:
        exp = _select_experiment(state_provider, experiment_id)
        if exp is None:
            return

        actions = exp.actions
        if not actions:
            cprint(
                f"No actions registered for experiment '{exp.experiment_id}'",
                "yellow",
            )
            return

        cprint(f"Actions for experiment '{exp.experiment_id}':", "cyan")
        for action_id, action in actions.items():
            desc = action.description() if hasattr(action, "description") else ""
            action_class = getattr(action, "action_class", "")
            print(f"  {action_id:<30} {desc:<40} [{action_class}]")
    finally:
        state_provider.close()


def _select_action(state_provider, exp, action_id: str | None):
    """Select an action from an experiment, prompting interactively if not specified.

    Returns the action_id or None if not found/cancelled.
    """
    from rich.console import Console
    from rich.prompt import IntPrompt

    actions = exp.actions
    if not actions:
        cprint(
            f"No actions registered for experiment '{exp.experiment_id}'",
            "yellow",
        )
        return None

    if action_id is not None:
        if action_id not in actions:
            cprint(f"Action '{action_id}' not found in experiment", "red")
            available = list(actions.keys())
            if available:
                cprint(f"Available actions: {', '.join(available)}", "yellow")
            return None
        return action_id

    # Interactive selection
    action_list = list(actions.items())
    console = Console()
    console.print("\n[bold]Select an action:[/bold]")
    for i, (aid, action) in enumerate(action_list, 1):
        desc = action.description() if hasattr(action, "description") else ""
        console.print(f"  [cyan]{i}[/cyan]) {desc}  [dim]({aid})[/dim]")

    idx = IntPrompt.ask(
        "\n[dim]Select[/dim]",
        choices=[str(i) for i in range(1, len(action_list) + 1)],
        show_choices=False,
    )
    return action_list[idx - 1][0]


@actions_group.command("run")
@click.argument("experiment_id", type=str, required=False, default=None)
@click.argument("action_id", type=str, required=False, default=None)
@click.option(
    "--set",
    "prefill_args",
    multiple=True,
    help="Pre-fill answer: key=value (can be repeated)",
)
@pass_workdir
def run_action(
    workdir: Path,
    experiment_id: str | None,
    action_id: str | None,
    prefill_args: tuple[str, ...],
):
    """Execute an experiment action.

    If EXPERIMENT_ID or ACTION_ID are not given, prompts for interactive selection.
    """
    from experimaestro.actions import CLIInteraction
    from experimaestro.core.serialization import load_xp_info
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    state_provider = WorkspaceStateProvider(workdir, no_cleanup=True)
    try:
        exp = _select_experiment(state_provider, experiment_id)
        if exp is None:
            return

        selected_action_id = _select_action(state_provider, exp, action_id)
        if selected_action_id is None:
            return

        # Load full action from objects.jsonl
        xp_info = load_xp_info(exp.workdir)
        if selected_action_id not in xp_info.actions:
            cprint(
                f"Action '{selected_action_id}' not found in objects.jsonl",
                "red",
            )
            return

        action = xp_info.actions[selected_action_id]

        # Parse pre-fill args
        prefill: dict[str, str] = {}
        for arg in prefill_args:
            if "=" not in arg:
                cprint(f"Invalid --set format: '{arg}' (expected key=value)", "red")
                return
            key, value = arg.split("=", 1)
            prefill[key] = value

        # Execute with CLI interaction
        interaction = CLIInteraction(prefill=prefill if prefill else None)
        cprint(f"Running action: {action.describe()}", "cyan")
        action.execute(interaction)
        cprint("Action completed successfully", "green")

    finally:
        state_provider.close()


# === Old names ===

experiments.add_command(experiments_monitor, "monitor")
experiments.add_command(ssh_monitor, "ssh-monitor")
for old_name, name in [
    ("history", "list"),
    ("history-prune", "prune"),
    ("tag", "tag"),
    ("untag", "untag"),
    ("tags", "tags"),
]:
    experiments.add_command(
        deprecated_alias(runs.commands[name], f"experiments runs {name}"), old_name
    )
