# flake8: noqa: T201
import subprocess
from typing import Optional
import click
from pathlib import Path
from termcolor import colored, cprint

from experimaestro.settings import find_workspace
from . import check_xp_path, cli


@click.option("--workspace", default="", help="Experimaestro workspace")
@click.option("--workdir", type=Path, default=None)
@cli.group()
@click.pass_context
def jobs(
    ctx,
    workdir: Optional[Path],
    workspace: Optional[str],
):
    """Job control: list, kill and clean

    The job filter is a boolean expression where tags (alphanumeric)
    and special job information (@state for job state, @name for job full
    name) can be compared to a given value (using '~' for regex matching,
    '=', 'not in', or 'in')

    For instance,

    model = "bm25" and mode in ["a", b"] and @state = "RUNNING"

    selects jobs where the tag model is "bm25", the tag mode is either
    "a" or "b", and the state is running.

    Note: Jobs are read from the workspace database. If jobs are missing,
    run 'experimaestro experiments sync' to synchronize the database
    with the filesystem.
    """
    ws = ctx.obj.workspace = find_workspace(workdir=workdir, workspace=workspace)
    check_xp_path(ctx, None, ws.path)


def process(
    workspace,
    *,
    experiment="",
    tags="",
    clean=False,
    kill=False,
    filter="",
    perform=False,
    fullpath=False,
    count=0,
):
    """Process jobs from the workspace database

    Args:
        workspace: Workspace settings
        experiment: Filter by experiment ID
        tags: Show tags in output
        clean: Clean finished jobs
        kill: Kill running jobs
        filter: Filter expression
        perform: Actually perform kill/clean (dry run if False)
        fullpath: Show full paths instead of short names
        count: Limit output to N most recent jobs (0 = no limit)
    """
    from .filter import createFilter, FilterContext
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
    from experimaestro.scheduler import JobState

    # Get state provider (read-only monitoring)
    provider = WorkspaceStateProvider.get_instance(workspace.path)

    # Get jobs from the database, optionally filtered by experiment
    if experiment:
        all_jobs = provider.get_jobs(experiment_id=experiment)
    else:
        all_jobs = provider.get_all_jobs()

    # Load tags map for the experiment (if specified)
    tags_map = {}
    if experiment:
        tags_map = provider.get_tags_map(experiment_id=experiment)

    # Create filter context with tags map
    filter_ctx = FilterContext(tags_map=tags_map)

    # Create filter function with context
    _filter = createFilter(filter, filter_ctx) if filter else None

    # Apply filter expression
    if _filter:
        all_jobs = [j for j in all_jobs if _filter(j)]

    # Sort by submission time (most recent first)
    # Jobs without submittime go to the end
    all_jobs.sort(key=lambda j: j.submittime or 0, reverse=True)

    # Limit to N most recent jobs if count is specified
    if count > 0:
        all_jobs = all_jobs[:count]

    if not all_jobs:
        cprint("No jobs found.", "yellow")
        return

    # Process each job
    for job in all_jobs:
        job_str = str(job.path) if fullpath else f"{job.task_id}/{job.identifier}"

        if job.state is None or job.state == JobState.UNSCHEDULED:
            print(colored(f"UNSCHED {job_str}", "red"), end="")
        elif job.state.running():
            if kill:
                if perform:
                    if provider.kill_job(job, perform=True):
                        cprint(f"KILLED  {job_str}", "light_red")
                    else:
                        cprint(f"KILL FAILED {job_str}", "red")
                else:
                    cprint(f"KILLING {job_str} (dry run)", "yellow")
            else:
                print(colored(f"{job.state.name:8}{job_str}", "yellow"), end="")
        elif job.state == JobState.DONE:
            print(colored(f"DONE    {job_str}", "green"), end="")
        elif job.state == JobState.ERROR:
            print(colored(f"FAIL    {job_str}", "red"), end="")
        else:
            print(colored(f"{job.state.name:8}{job_str}", "red"), end="")

        # Show tags if requested (from tags_map)
        if tags:
            job_tags = tags_map.get(job.identifier, {})
            if job_tags:
                print(f""" {" ".join(f"{k}={v}" for k, v in job_tags.items())}""")
            elif not (kill and perform):
                print()
        elif not (kill and perform):
            print()

        # Clean finished jobs
        if clean and job.state and job.state.finished():
            if perform:
                if provider.clean_job(job, perform=True):
                    cprint("  Cleaned", "red")
                else:
                    cprint("  Clean failed", "red")
            else:
                cprint("  Would clean (dry run)", "yellow")

    print()


@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--filter", default="", help="Filter expression")
@click.option("--fullpath", is_flag=True, help="Prints full paths")
@click.option("--count", "-c", default=0, type=int, help="Limit to N most recent jobs")
@jobs.command()
@click.pass_context
def list(
    ctx,
    experiment: str,
    filter: str,
    tags: bool,
    fullpath: bool,
    count: int,
):
    """List all jobs in the workspace (sorted by submission date, most recent first)"""
    process(
        ctx.obj.workspace,
        experiment=experiment,
        filter=filter,
        tags=tags,
        fullpath=fullpath,
        count=count,
    )


@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--filter", default="", help="Filter expression")
@click.option("--perform", is_flag=True, help="Really perform the killing")
@click.option("--fullpath", is_flag=True, help="Prints full paths")
@jobs.command()
@click.pass_context
def kill(
    ctx,
    experiment: str,
    filter: str,
    tags: bool,
    fullpath: bool,
    perform: bool,
):
    """Kill running jobs"""
    process(
        ctx.obj.workspace,
        experiment=experiment,
        filter=filter,
        tags=tags,
        kill=True,
        perform=perform,
        fullpath=fullpath,
    )


@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--filter", default="", help="Filter expression")
@click.option("--perform", is_flag=True, help="Really perform the cleaning")
@click.option("--fullpath", is_flag=True, help="Prints full paths")
@jobs.command()
@click.pass_context
def clean(
    ctx,
    experiment: str,
    filter: str,
    tags: bool,
    fullpath: bool,
    perform: bool,
):
    """Clean finished jobs (delete directories and DB entries)"""
    process(
        ctx.obj.workspace,
        experiment=experiment,
        filter=filter,
        tags=tags,
        clean=True,
        perform=perform,
        fullpath=fullpath,
    )


@click.argument("jobid", type=str)
@click.option(
    "--follow", "-f", help="Use tail instead of less to follow changes", is_flag=True
)
@click.option("--std", help="Follow stdout instead of stderr", is_flag=True)
@jobs.command()
@click.pass_context
def log(ctx, jobid: str, follow: bool, std: bool):
    """View job log (stderr by default, stdout with --std)

    JOBID format: task.name/hash (e.g., mymodule.MyTask/abc123)
    """
    task_name, task_hash = jobid.split("/")
    _, name = task_name.rsplit(".", 1)
    log_path = (
        ctx.obj.workspace.path
        / "jobs"
        / task_name
        / task_hash
        / f"""{name}.{"out" if std else "err"}"""
    )
    if not log_path.exists():
        cprint(f"Log file not found: {log_path}", "red")
        return
    if follow:
        subprocess.run(["tail", "-f", log_path])
    else:
        subprocess.run(["less", "-r", log_path])


@click.argument("jobid", type=str)
@jobs.command()
@click.pass_context
def path(ctx, jobid: str):
    """Print the path to a job directory

    JOBID format: task.name/hash (e.g., mymodule.MyTask/abc123)
    """
    task_name, task_hash = jobid.split("/")
    job_path = ctx.obj.workspace.path / "jobs" / task_name / task_hash
    if not job_path.exists():
        cprint(f"Job directory not found: {job_path}", "red")
        return
    print(job_path)


@click.option("--perform", is_flag=True, help="Actually delete orphan partials")
@jobs.command("cleanup-partials")
@click.pass_context
def cleanup_partials(ctx, perform: bool):
    """Clean up orphan partial directories

    Partial directories are shared checkpoint locations created by
    partial. When all jobs using a partial are deleted, the
    partial becomes orphaned and can be cleaned up.

    This command finds all orphan partials and deletes them (or shows
    what would be deleted in dry-run mode).
    """
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

    provider = WorkspaceStateProvider.get_instance(ctx.obj.workspace.path)

    try:
        orphan_paths = provider.cleanup_orphan_partials(perform=perform)

        if not orphan_paths:
            cprint("No orphan partials found.", "green")
            return

        if perform:
            cprint(f"Cleaned {len(orphan_paths)} orphan partial(s):", "green")
        else:
            cprint(f"Found {len(orphan_paths)} orphan partial(s) (dry run):", "yellow")

        for path in orphan_paths:
            if perform:
                print(colored(f"  Deleted: {path}", "red"))
            else:
                print(colored(f"  Would delete: {path}", "yellow"))

    finally:
        if perform:
            provider.close()


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
@jobs.command()
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
    experimaestro jobs stray

    \b
    # List stray jobs with sizes
    experimaestro jobs stray --size

    \b
    # Kill running stray jobs (dry run)
    experimaestro jobs stray --kill

    \b
    # Kill running stray jobs (for real)
    experimaestro jobs stray --kill --perform

    \b
    # Delete non-running stray jobs
    experimaestro jobs stray --delete --perform

    \b
    # Kill and delete all stray jobs (dangerous!)
    experimaestro jobs stray --kill --delete --perform --force
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
            try:
                result = subprocess.run(
                    ["du", "-hs", str(job.path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    size_str = result.stdout.strip().split()[0]
                    print(f"  [{size_str}]", end="")
            except (subprocess.TimeoutExpired, Exception):
                print("  [?]", end="")

        print()

        # Kill running jobs if requested
        if kill and job.state and job.state.running():
            if perform:
                if provider.kill_job(job, perform=True):
                    cprint("  KILLED", "light_red")
                    killed_count += 1
                else:
                    cprint("  KILL FAILED", "red")
            else:
                cprint("  Would kill (dry run)", "yellow")

        # Delete non-running jobs if requested
        if delete and (not job.state or not job.state.running()):
            if perform:
                success, msg = provider.delete_job_safely(job, cascade_orphans=False)
                if success:
                    cprint("  DELETED", "light_red")
                    deleted_count += 1
                else:
                    cprint(f"  DELETE FAILED: {msg}", "red")
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
