import click
import logging
from functools import update_wrapper
from pathlib import Path
import importlib
import importlib.machinery
import json

from .utils import logger
from experimaestro.api import ObjectType, Config, TypeInformation
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
@click.option("--traceback", is_flag=True, help="Display traceback if an exception occurs")
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
@click.argument("taskid", type=str)
@click.argument("path", type=str)
@click.option("--file", is_flag=True, help="The path is not a module but a python file")
@cli.command(context_settings={"allow_extra_args": True})
def run(file, path, taskid, parameters):
    """Run a task"""

    if file:
        loader = importlib.machinery.SourceFileLoader(path, path)
        mod = loader.load_module()
    else:
        logging.debug("Importing module %s", path)
        importlib.import_module(path)

    tasktype = ObjectType.REGISTERED[taskid]

    with open(parameters, "r") as fp:
        Config.TASKMODE = True
        
        params = json.load(fp)
        TypeInformation.LOADING = True
        task = tasktype(**params)
        TypeInformation.LOADING = False
        
        task.execute()

def main():
    cli(obj=None)

if __name__ == "__main__":
    main()


