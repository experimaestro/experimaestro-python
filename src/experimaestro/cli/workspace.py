# flake8: noqa: T201
"""``experimaestro workspace``: list, clean up and migrate workspaces"""

__lazy_modules__ = ["yaml", "experimaestro.scheduler"]  # PEP 810 (Python 3.15+)

import logging
from dataclasses import asdict
from pathlib import Path
from shutil import rmtree

import click
import yaml
from termcolor import colored, cprint

from experimaestro.scheduler import job_operations
from experimaestro.settings import get_settings
from experimaestro.utils.disk_usage import folder_size, format_size

from . import RunConfig, check_xp_path, root_config
from .remote import RemoteCommand


@click.group()
def workspace():
    """Workspaces: list, clean up, migrate (alias: ws)"""


@workspace.command(name="list")
@click.option(
    "-v", "--verbose", is_flag=True, help="Display all parameters for each workspace"
)
def list_workspaces(verbose):
    """List workspaces configured in settings.yaml"""
    settings = get_settings()

    if not settings.workspaces:
        cprint("No workspaces configured in settings.yaml", "yellow")
        return

    cprint("Experimaestro workspaces:", "white", attrs=["bold"])
    for ws in settings.workspaces:
        tags = []
        if ws.is_remote:
            tags.append(f"remote: {ws.ssh.host}")

        tags_str = f" [{', '.join(tags)}]" if tags else ""
        cprint(f"- {ws.id}: {ws.path}{tags_str}", "cyan")

        if verbose:
            # Print a structured representation of all workspace parameters
            ws_dict = asdict(ws)
            # Remove redundant ID and path from the verbose dump as they are already printed
            ws_dict.pop("id", None)
            ws_dict.pop("path", None)

            # path is a Path object, which might not serialize cleanly with simple yaml dump in older PyYAML, but asdict() might keep it as Path.
            # We convert Path to str just to be safe.
            def sanitize(d):
                for k, v in list(d.items()):
                    if hasattr(v, "__fspath__"):
                        d[k] = str(v)
                    elif isinstance(v, dict):
                        sanitize(v)
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            if hasattr(item, "__fspath__"):
                                v[i] = str(item)
                            elif isinstance(item, dict):
                                sanitize(item)

            sanitize(ws_dict)
            cprint(
                yaml.dump(ws_dict, sort_keys=False, default_flow_style=False), "white"
            )


def _orphans(path: Path, clean: bool, size: bool, include_folders: bool):
    """Check for tasks that are not part of an experimental plan

    Uses the same orphan detection as the TUI (WorkspaceStateProvider.get_orphan_jobs).
    """
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
    from experimaestro.utils.disk_usage import folder_size, format_size

    def show(job, prefix=""):
        key = f"{job.task_id}/{job.identifier}"
        if size:
            print(prefix, f"{format_size(folder_size(job.path))}\t{job.path}", sep=None)
        else:
            print(prefix, key, sep=None)

    # Use WorkspaceStateProvider.get_orphan_jobs() - same as TUI
    provider = WorkspaceStateProvider.get_instance(path)
    orphan_jobs = provider.get_orphan_jobs(include_folders=include_folders)

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


@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.option(
    "--include-folders/--no-folders",
    default=True,
    help=(
        "Treat jobs reachable through backup/move folders as non-orphan "
        "(default: enabled). Disable to scan only the primary workspace."
    ),
)
@workspace.command(cls=RemoteCommand)
@click.pass_obj
def orphans(cfg: RunConfig, clean: bool, size: bool, include_folders: bool):
    """Check for jobs that are not part of any experiment

    Uses the same orphan detection as the TUI (WorkspaceStateProvider.get_orphan_jobs).
    """
    _orphans(cfg.workspace.path, clean, size, include_folders)


@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.option(
    "--include-folders/--no-folders",
    default=True,
    help=(
        "Treat jobs reachable through backup/move folders as non-orphan "
        "(default: enabled). Disable to scan only the primary workspace."
    ),
)
@click.argument("path", type=Path, callback=check_xp_path)
@click.command(hidden=True)
def orphans_alias(path: Path, clean: bool, size: bool, include_folders: bool):
    """Check for jobs that are not part of any experiment (deprecated)"""
    cprint("Deprecated: use `experimaestro workspace orphans` instead", "yellow")
    _orphans(path, clean, size, include_folders)


def _check_xp_path(ctx, param, path: Path | None):
    return None if path is None else check_xp_path(ctx, param, path)


@workspace.group()
def migrate():
    """Migration commands for experimaestro workspace upgrades"""
    pass


@migrate.command("v1-to-v2", cls=RemoteCommand)
@click.argument("workdir", type=Path, required=False, callback=_check_xp_path)
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

    if workdir is None:
        workdir = root_config().workspace.path

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


@workspace.command("cleanup", cls=RemoteCommand)
@click.option("--workdir", type=Path, default=None, hidden=True)
@click.option("--workspace", "workspace_id", type=str, default=None, hidden=True)
@click.pass_obj
def cleanup(cfg: RunConfig, workdir: Path | None, workspace_id: str | None):
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
    experimaestro workspace cleanup
    """
    from experimaestro.scheduler.cleanup import perform_workspace_cleanup
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    # Resolve workspace
    cfg.set_workspace(workdir, workspace_id)
    workdir = cfg.workspace.path

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


@click.option(
    "--kill", is_flag=True, help="Kill running stray jobs (requires --perform)"
)
@click.option(
    "--delete", is_flag=True, help="Delete non-running stray jobs (requires --perform)"
)
@click.option("--perform", is_flag=True, help="Actually perform the operation")
@click.option(
    "--force",
    is_flag=True,
    help="Bypass safety checks (e.g., when scheduler is running)",
)
@click.option("--size", is_flag=True, help="Show size of each job folder")
@click.option("--fullpath", is_flag=True, help="Show full paths")
@workspace.command(cls=RemoteCommand)
@click.pass_context
def stray(
    ctx,
    kill: bool,
    delete: bool,
    perform: bool,
    force: bool,
    size: bool,
    fullpath: bool,
):
    """Manage stray jobs (jobs not associated with any experiment)

    Stray jobs are jobs that exist on disk but are not referenced by any
    experiment. This can happen when:

    \b
    - An experiment plan changes and a job is no longer needed
    - An experiment is deleted but jobs remain on disk
    - Jobs are manually created outside of experiments

    Safety: By default, this command will warn if an experiment appears to be
    running (scheduler active). Use --force to bypass this check.

    Examples:

    \b
    # List all stray jobs
    experimaestro workspace stray

    \b
    # List stray jobs with sizes
    experimaestro workspace stray --size

    \b
    # Kill running stray jobs (dry run)
    experimaestro workspace stray --kill

    \b
    # Kill running stray jobs (for real)
    experimaestro workspace stray --kill --perform

    \b
    # Delete non-running stray jobs
    experimaestro workspace stray --delete --perform

    \b
    # Kill and delete all stray jobs (dangerous!)
    experimaestro workspace stray --kill --delete --perform --force
    """
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
    from experimaestro.scheduler import JobState

    provider = WorkspaceStateProvider.get_instance(ctx.obj.workspace.path)

    # Safety check: warn if scheduler appears to be running
    if provider.is_live and not force:
        cprint(
            "Warning: Scheduler appears to be running. Stray detection may be inaccurate.",
            "yellow",
        )
        cprint("Use --force to proceed anyway.", "yellow")
        if perform:
            cprint("Aborting due to active scheduler.", "red")
            return

    # Get stray jobs (running orphans) and all orphan jobs
    stray_jobs = provider.get_stray_jobs()
    stray_jobs = [j for j in stray_jobs if j.path and j.path.exists()]

    orphan_jobs = provider.get_orphan_jobs()
    orphan_jobs = [j for j in orphan_jobs if j.path and j.path.exists()]

    # Finished orphans = orphans that are not stray (not running)
    stray_ids = {j.identifier for j in stray_jobs}
    finished_jobs = [j for j in orphan_jobs if j.identifier not in stray_ids]

    if not stray_jobs and not finished_jobs:
        cprint("No stray or orphan jobs found.", "green")
        return

    # Print summary
    print(
        f"Found {len(stray_jobs)} stray (running) and {len(finished_jobs)} orphan (finished) jobs:"
    )
    if stray_jobs:
        cprint(f"  {len(stray_jobs)} stray (running)", "yellow")
    if finished_jobs:
        cprint(f"  {len(finished_jobs)} orphan (finished)", "cyan")
    print()

    # Combine for display (stray first, then finished orphans)
    all_jobs = stray_jobs + finished_jobs

    # Process each job
    killed_count = 0
    deleted_count = 0

    for job in all_jobs:
        job_str = str(job.path) if fullpath else f"{job.task_id}/{job.identifier}"
        state_name = job.state.name if job.state else "UNKNOWN"

        # Determine color based on state
        if job.state and job.state.running():
            state_color = "yellow"
        elif job.state == JobState.DONE:
            state_color = "green"
        elif job.state == JobState.ERROR:
            state_color = "red"
        else:
            state_color = "white"

        # Show job info
        print(colored(f"{state_name:10}{job_str}", state_color), end="")

        # Show size if requested
        if size and job.path and job.path.exists():
            print(f"  [{format_size(folder_size(job.path))}]", end="")

        print()

        # Kill running jobs if requested
        if kill and job_operations.check_kill_job(job) is None:
            if perform:
                outcome = job_operations.kill_job(provider, job)
                if outcome.ok:
                    cprint("  KILLED", "light_red")
                    killed_count += 1
                else:
                    cprint(f"  KILL FAILED: {outcome.message}", "red")
            else:
                cprint("  Would kill (dry run)", "yellow")

        # Delete finished jobs if requested (orphan partials are cleaned once,
        # at the end)
        if delete and job_operations.check_delete_job(job) is None:
            if perform:
                outcome = job_operations.delete_job(provider, job, cleanup=False)
                if outcome.ok:
                    cprint("  DELETED", "light_red")
                    deleted_count += 1
                else:
                    cprint(f"  DELETE FAILED: {outcome.message}", "red")
            else:
                cprint("  Would delete (dry run)", "yellow")

    # Summary
    print()
    if perform:
        if kill and killed_count > 0:
            cprint(f"Killed {killed_count} running job(s)", "green")
        if delete and deleted_count > 0:
            cprint(f"Deleted {deleted_count} job(s)", "green")
            # Clean up orphan partials after deleting jobs
            provider.cleanup_orphan_partials(perform=True)
    else:
        if kill or delete:
            cprint("Dry run - no changes made. Use --perform to execute.", "yellow")
