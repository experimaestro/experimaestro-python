import pkg_resources
from itertools import chain
from shutil import rmtree
import click
import logging
from functools import cached_property, update_wrapper
from pathlib import Path
import subprocess
from termcolor import colored, cprint

import experimaestro

# --- Command line main options
logging.basicConfig(level=logging.INFO)


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


@cli.command(help="Get version")
def version():
    print(experimaestro.__version__)


@click.argument("parameters", type=Path)
@cli.command(context_settings={"allow_extra_args": True})
def run(parameters):
    """Run a task"""

    from experimaestro.run import run as do_run

    do_run(parameters)


@click.option(
    "--clean", is_flag=True, help="Remove the socket file and its enclosing directory"
)
@click.argument("unix-path", type=Path)
@cli.command()
def rpyc_server(unix_path, clean):
    """Start an rPyC server"""
    from experimaestro.rpyc import start_server

    start_server(unix_path, clean=clean)


@click.argument("path", type=Path)
@click.option("--fix", is_flag=True, help="Do fix deprecated jobs")
@cli.command()
def deprecated(path: Path, fix: bool):
    """List deprecated jobs and allows fixing (i.e. linking) them"""
    from experimaestro.tools.jobs import fix_deprecated

    fix_deprecated(path, fix)


@click.argument("path", type=Path)
@click.option("--experiment", default=None, help="Restrict to this experiment")
@click.option("--tags", is_flag=True, help="Show tags")
@click.option("--ready", is_flag=True, help="Include tasks which are not yet scheduled")
@click.option("--filter", default="", help="Filter expression")
@click.option(
    "--force",
    is_flag=True,
    help="Force operation even if experiment has not been completed yet",
)
@click.option("--kill", is_flag=True, help="Kill filtered tasks (when/before running)")
@click.option("--clean", is_flag=True, help="Remove finished tasks directories")
@cli.command()
def jobs(
    path: Path,
    experiment: str,
    filter: str,
    tags: bool,
    ready: bool,
    kill: bool,
    clean: bool,
    force: bool,
):
    """Job control: list, kill and clean"""
    for p in (path / "xp").glob("*"):
        if experiment and p.name != experiment:
            continue

        from .filter import createFilter, JobInformation
        from experimaestro.scheduler import JobState

        _filter = createFilter(filter) if filter else lambda x: True

        print(f"* Experiment {p.name}")
        if (p / "jobs.bak").is_dir():
            cprint("  Experiment has not finished yet", "red")
            if not force and (kill or clean):
                cprint("  Preventing kill/clean (use --force if you want to)", "yellow")
                kill = False
                clean = False
        print()

        for job in p.glob("jobs/*/*"):
            info = None
            p = job.resolve()
            if p.is_dir():
                *_, scriptname = p.parent.name.rsplit(".", 1)

                info = JobInformation(p, scriptname)
                if filter:
                    if not _filter(info):
                        continue

                if info.state is None:
                    print(
                        colored(f"NODIR   {job.parent.name}/{job.name}", "red"), end=""
                    )
                elif info.state.running():
                    if kill:
                        process = info.getprocess()
                        print("KILLING", process)
                        process.kill()
                    print(
                        colored(
                            f"{info.state.name:8}{job.parent.name}/{job.name}", "yellow"
                        ),
                        end="",
                    )
                elif info.state == JobState.DONE:
                    print(
                        colored(f"DONE    {job.parent.name}/{job.name}", "green"),
                        end="",
                    )
                elif info.state == JobState.ERROR:
                    print(
                        colored(f"FAIL    {job.parent.name}/{job.name}", "red"), end=""
                    )
                else:
                    print(
                        colored(
                            f"{info.state.name:8}{job.parent.name}/{job.name}", "red"
                        ),
                        end="",
                    )

            else:
                if not ready:
                    continue
                print(colored(f"READY {job.parent.name}/{job.name}", "yellow"), end="")

            if tags:
                print(f""" {" ".join(f"{k}={v}" for k, v in info.tags.items())}""")
            else:
                print()

            if clean and info.state and info.state.finished():
                cprint("Cleaning...", "red")
                rmtree(p)
        print()


@click.option("--show-all", is_flag=True, help="Show even not orphans")
@click.option(
    "--ignore-old", is_flag=True, help="Ignore old jobs for unfinished experiments"
)
@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.argument("path", type=Path)
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

    for p in (path / "xp").glob("*/jobs.bak"):
        logging.warning("Experiment %s has not completed successfully", p.parent.name)

    # Retrieve the jobs within expedriments (jobs and jobs.bak folder within experiments)
    xpjobs = set()
    if ignore_old:
        paths = (path / "xp").glob("*/jobs")
    else:
        paths = chain((path / "xp").glob("*/jobs"), (path / "xp").glob("*/jobs.bak"))

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


class Launchers(click.MultiCommand):
    """Connectors commands"""

    @cached_property
    def commands(self):
        map = {}
        for ep in pkg_resources.iter_entry_points(f"experimaestro.{self.name}"):
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


def main():
    cli(obj=None)


if __name__ == "__main__":
    main()
