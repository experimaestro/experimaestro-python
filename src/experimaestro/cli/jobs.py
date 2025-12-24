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
    from .filter import createFilter
    from experimaestro.scheduler.state_provider import WorkspaceStateProvider
    from experimaestro.scheduler import JobState

    _filter = createFilter(filter) if filter else None

    # Get state provider (write mode for kill/clean operations)
    read_only = not (kill or clean)
    provider = WorkspaceStateProvider.get_instance(workspace.path, read_only=read_only)

    try:
        # Get all jobs from the database
        all_jobs = provider.get_all_jobs()

        # Filter by experiment if specified
        if experiment:
            all_jobs = [j for j in all_jobs if j.experiment_id == experiment]

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

            # Add experiment info
            if job.experiment_id:
                job_str += f" [{job.experiment_id}]"

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

            # Show tags if requested
            if tags and job.tags:
                print(f""" {" ".join(f"{k}={v}" for k, v in job.tags.items())}""")
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

    finally:
        # Close provider if we created it for write mode
        if not read_only:
            provider.close()


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
        / f"""{name}.{'out' if std else 'err'}"""
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
    subparameters. When all jobs using a partial are deleted, the
    partial becomes orphaned and can be cleaned up.

    This command finds all orphan partials and deletes them (or shows
    what would be deleted in dry-run mode).
    """
    from experimaestro.scheduler.state_provider import WorkspaceStateProvider

    provider = WorkspaceStateProvider.get_instance(
        ctx.obj.workspace.path, read_only=not perform
    )

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
