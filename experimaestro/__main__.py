from itertools import chain
from shutil import rmtree
import click
import logging
from functools import update_wrapper
from pathlib import Path
import subprocess
import json

from experimaestro.core.objects import Config, ConfigInformation
import experimaestro
import experimaestro.taskglobals as taskglobals

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

    with open(parameters, "r") as fp:
        params = json.load(fp)
        taskglobals.wspath = Path(params["workspace"])

        task = ConfigInformation.fromParameters(params["objects"])
        task.__taskdir__ = Path.cwd()

        # Set the tags
        task.__tags__ = params["tags"]

        # If we have sub-parameters, we set the __maintaskdir__ folder
        if params["has_subparam"]:
            task.__maintaskdir__ = Path.cwd().parents[1]

        task.execute()


@click.option(
    "--clean", is_flag=True, help="Remove the socket file and its enclosing directory"
)
@click.argument("unix-path", type=Path)
@cli.command()
def rpyc_server(unix_path, clean):
    """Start an rPyC server"""
    from experimaestro.rpyc import start_server

    start_server(unix_path, clean=clean)


@click.option("--show-all", is_flag=True, help="Show even not orphans")
@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.argument("path", type=Path)
@cli.command()
def orphans(path: Path, clean: bool, size: bool, show_all: bool):
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

    xpjobs = set(
        key
        for key, value in chain(
            *[getjobs(p) for p in (path / "xp").glob("*/jobs") if p.is_dir()]
        )
    )

    found = 0
    for key, job in getjobs(jobspath):
        if key not in xpjobs:
            show(jobspath, key)
            if clean:
                rmtree(job)
        else:
            if show_all:
                show(key, "[not orphan] ")
            found += 1

    print(f"{found} jobs are not orphans")


def main():
    cli(obj=None)


if __name__ == "__main__":
    main()
