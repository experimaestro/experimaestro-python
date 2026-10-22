# flake8: noqa: T201
import subprocess
from typing import TYPE_CHECKING
import click
from pathlib import Path
from termcolor import colored, cprint

from experimaestro.scheduler import job_operations
from . import RunConfig
from .inspect import inspect_command
from .remote import RemoteCommand

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import StateProvider


@click.option("--workspace", default="", hidden=True, help="Experimaestro workspace")
@click.option("--workdir", type=Path, default=None, hidden=True)
@click.group()
@click.pass_obj
def jobs(
    cfg: RunConfig,
    workdir: Path | None,
    workspace: str | None,
):
    """Jobs: list, kill, clean, inspect, logs

    The job filter is a boolean expression where tags (alphanumeric)
    and special job information (@state for job state, @name for job full
    name) can be compared to a given value (using '~' for regex matching,
    '=', 'not in', or 'in')

    For instance,

    model = "bm25" and mode in ["a", b"] and @state = "RUNNING"

    selects jobs where the tag model is "bm25", the tag mode is either
    "a" or "b", and the state is running.
    """
    cfg.set_workspace(workdir, workspace)


def process(
    provider: "StateProvider",
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
        provider: State provider of the workspace
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
    from experimaestro.scheduler import JobState

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
    # Use experiment job info for submittime
    experiment_job_info = {}
    if experiment:
        experiment_job_info = provider.get_experiment_job_info(experiment_id=experiment)
    all_jobs.sort(
        key=lambda j: (
            experiment_job_info[j.identifier].timestamp
            if j.identifier in experiment_job_info
            and experiment_job_info[j.identifier].timestamp
            else 0
        ),
        reverse=True,
    )

    # Limit to N most recent jobs if count is specified
    if count > 0:
        all_jobs = all_jobs[:count]

    if not all_jobs:
        cprint("No jobs found.", "yellow")
        return

    # Process each job
    for job in all_jobs:
        job_str = str(job.path) if fullpath else f"{job.task_id}/{job.identifier}"

        if job.state is None or job.state.is_unscheduled():
            print(colored(f"UNSCHED {job_str}", "red"), end="")
        elif job.state.running():
            if kill:
                if perform:
                    outcome = job_operations.kill_job(provider, job)
                    if outcome.ok:
                        cprint(f"KILLED  {job_str}", "light_red")
                    else:
                        cprint(f"KILL FAILED {job_str}: {outcome.message}", "red")
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
        ctx.obj.state_provider(),
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
        ctx.obj.state_provider(),
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
        ctx.obj.state_provider(),
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
@jobs.command(cls=RemoteCommand)
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
@jobs.command(cls=RemoteCommand)
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


@click.argument("jobid", type=str)
@jobs.command(cls=RemoteCommand)
@click.pass_context
def recover(ctx, jobid: str):
    """Recover a job from a configured backup folder (**Beta**).

    JOBID format: task.name/hash. Copies the job back into the primary
    workspace if it is missing locally but present in any folder
    attached to the workspace (mode=backup / move / use).
    """
    from experimaestro.scheduler.folders import recover_from_folders

    task_name, task_hash = jobid.split("/")
    primary = ctx.obj.workspace.path / "jobs"
    rel = Path(task_name) / task_hash

    dest = recover_from_folders(rel, primary, ctx.obj.workspace.folders)
    if dest is None:
        cprint(f"No folder contained job {jobid}", "red")
        return
    cprint(f"Recovered {jobid} into {dest}", "green")


jobs.add_command(inspect_command, "inspect")


@click.argument("path2", type=Path)
@click.argument("path1", type=Path)
@jobs.command()
def diff(path1, path2):
    """Compute the difference between two job configurations (params.json)"""

    from experimaestro.tools.diff import diff

    diff(path1, path2)
