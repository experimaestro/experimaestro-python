import click
import logging
from functools import update_wrapper
from pathlib import Path
import importlib
import importlib.machinery
import json

from .utils import logger
from experimaestro.core.types import ObjectType
from experimaestro.core.objects import Config, ConfigInformation
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

    Config.TASKMODE = True

    with open(parameters, "r") as fp:
        params = json.load(fp)
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


def main():
    cli(obj=None)


if __name__ == "__main__":
    main()
