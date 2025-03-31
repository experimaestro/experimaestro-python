# flake8: noqa: T201
import subprocess
from typing import Optional
from shutil import rmtree
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

    """
    ws = ctx.obj.workspace = find_workspace(workdir=workdir, workspace=workspace)
    check_xp_path(ctx, None, ws.path)


def process(
    workspace,
    *,
    experiment="",
    tags="",
    ready=False,
    clean=False,
    kill=False,
    filter="",
    perform=False,
    fullpath=False,
):
    from .filter import createFilter, JobInformation
    from experimaestro.scheduler import JobState

    _filter = createFilter(filter) if filter else lambda x: True

    # Get all jobs from experiments
    job2xp = {}

    path = workspace.path
    for p in (path / "xp").glob("*"):
        for job in p.glob("jobs/*/*"):
            job_path = job.resolve()
            if job_path.is_dir():
                *_, scriptname = job_path.parent.name.rsplit(".", 1)
                job2xp.setdefault(scriptname, set()).add(p.name)

        if (p / "jobs.bak").is_dir():
            cprint(f"  Experiment {p.name} has not finished yet", "red")
            if (not perform) and (kill or clean):
                cprint("  Preventing kill/clean (use --perform if you want to)", "yellow")
                kill = False
                clean = False

    # Now, process jobs
    for job in path.glob("jobs/*/*"):
        info = None
        p = job.resolve()
        if p.is_dir():
            *_, scriptname = p.parent.name.rsplit(".", 1)
            xps = job2xp.get(scriptname, set())
            if experiment and experiment not in xps:
                continue

            info = JobInformation(p, scriptname)
            job_str = (
                (str(job.resolve()) if fullpath else f"{job.parent.name}/{job.name}")
                + " "
                + ",".join(xps)
            )

            if filter:
                if not _filter(info):
                    continue

            if info.state is None:
                print(colored(f"NODIR   {job_str}", "red"), end="")
            elif info.state.running():
                if kill:
                    if perform:
                        process = info.getprocess()
                        if process is None:
                            cprint(
                                "internal error – no process could be retrieved",
                                "red",
                            )
                        else:
                            cprint(f"KILLING {process}", "light_red")
                            process.kill()
                    else:
                        print("KILLING (not performing)", process)
                print(
                    colored(f"{info.state.name:8}{job_str}", "yellow"),
                    end="",
                )
            elif info.state == JobState.DONE:
                print(
                    colored(f"DONE    {job_str}", "green"),
                    end="",
                )
            elif info.state == JobState.ERROR:
                print(colored(f"FAIL    {job_str}", "red"), end="")
            else:
                print(
                    colored(f"{info.state.name:8}{job_str}", "red"),
                    end="",
                )

        else:
            if not ready:
                continue
            print(colored(f"READY {job_path}", "yellow"), end="")

        if tags:
            print(f""" {" ".join(f"{k}={v}" for k, v in info.tags.items())}""")
        else:
            print()

        if clean and info.state and info.state.finished():
            if perform:
                cprint("Cleaning...", "red")
                rmtree(p)
            else:
                cprint("Cleaning... (not performed)", "red")
    print()


@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--ready", is_flag=True, help="Include tasks which are not yet scheduled")
@click.option("--filter", default="", help="Filter expression")
@click.option("--fullpath", is_flag=True, help="Prints full paths")
@jobs.command()
@click.pass_context
def list(
    ctx,
    experiment: str,
    filter: str,
    tags: bool,
    ready: bool,
    fullpath: bool,
):
    process(
        ctx.obj.workspace,
        experiment=experiment,
        filter=filter,
        tags=tags,
        ready=ready,
        fullpath=fullpath,
    )


@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--ready", is_flag=True, help="Include tasks which are not yet scheduled")
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
    ready: bool,
    fullpath: bool,
    perform: bool,
):
    process(
        ctx.obj.workspace,
        experiment=experiment,
        filter=filter,
        tags=tags,
        ready=ready,
        kill=True,
        perform=perform,
        fullpath=fullpath,
    )


@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--ready", is_flag=True, help="Include tasks which are not yet scheduled")
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
    ready: bool,
    fullpath: bool,
    perform: bool,
):
    process(
        ctx.obj.workspace,
        experiment=experiment,
        filter=filter,
        tags=tags,
        ready=ready,
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
    task_name, task_hash = jobid.split("/")
    _, name = task_name.rsplit(".", 1)
    path = (
        ctx.obj.workspace.path
        / "jobs"
        / task_name
        / task_hash
        / f"""{name}.{'out' if std else 'err'}"""
    )
    if follow:
        subprocess.run(["tail", "-f", path])
    else:
        subprocess.run(["less", "-r", path])


@click.argument("jobid", type=str)
@jobs.command()
@click.pass_context
def path(ctx, jobid: str):
    task_name, task_hash = jobid.split("/")
    path = ctx.obj.workspace.path / "jobs" / task_name / task_hash
    print(path)
