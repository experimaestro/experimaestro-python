# flake8: noqa: T201
import sys
from typing import Set, Optional
import pkg_resources
from itertools import chain
from shutil import rmtree
import click
import logging
from functools import cached_property, update_wrapper
from pathlib import Path
import subprocess
from termcolor import cprint

import experimaestro
from experimaestro.experiments.cli import experiments_cli
import experimaestro.launcherfinder.registry as launcher_registry
from experimaestro.settings import find_workspace

# --- Command line main options
logging.basicConfig(level=logging.INFO)


def pass_cfg(f):
    """Pass configuration information"""

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj, *args, **kwargs)

    return update_wrapper(new_func, f)


def check_xp_path(ctx, self, path: Path):
    if not (path / ".__experimaestro__").is_file():
        cprint(f"{path} is not an experimaestro working directory", "red")
        for path in path.parents:
            if (path / ".__experimaestro__").is_file():
                cprint(f"{path} could be the folder you want", "green")
                if click.confirm("Do you want to use this folder?"):
                    return path
        sys.exit(1)

    return path


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


# Adds the run-experiment command
cli.add_command(experiments_cli, "run-experiment")


@cli.command(help="Get version")
def version():
    print(experimaestro.__version__)


@click.argument("parameters", type=Path)
@cli.command(context_settings={"allow_extra_args": True})
def run(parameters):
    """Run a task"""

    from experimaestro.run import run as do_run

    do_run(parameters)


@click.argument("path2", type=Path)
@click.argument("path1", type=Path)
@cli.command(context_settings={"allow_extra_args": True})
def parameters_difference(path1, path2):
    """Compute the difference between two configurations"""

    from experimaestro.tools.diff import diff

    diff(path1, path2)


@click.option(
    "--clean", is_flag=True, help="Remove the socket file and its enclosing directory"
)
@click.argument("unix-path", type=Path)
@cli.command()
def rpyc_server(unix_path, clean):
    """Start an rPyC server"""
    from experimaestro.rpyc import start_server

    start_server(unix_path, clean=clean)


@cli.group()
def deprecated():
    """Manage identifier changes"""
    pass


@click.option("--fix", is_flag=True, help="Generate links to new IDs")
@click.option("--cleanup", is_flag=True, help="Remove symbolic links and move folders")
@click.argument("path", type=Path, callback=check_xp_path)
@deprecated.command(name="list")
def deprecated_list(path: Path, fix: bool, cleanup: bool):
    """List deprecated jobs"""
    from experimaestro.tools.jobs import fix_deprecated

    if cleanup and not fix:
        logging.warning("Ignoring --cleanup since we are not fixing old IDs")
    fix_deprecated(path, fix, cleanup)


@click.argument("path", type=Path, callback=check_xp_path)
@deprecated.command()
def diff(path: Path):
    """Show the reason of the identifier change for a job"""
    from experimaestro.tools.jobs import load_job
    from experimaestro import Config
    from experimaestro.core.objects import ConfigWalkContext

    _, job = load_job(path / "params.json", discard_id=False)
    _, new_job = load_job(path / "params.json")

    def check(path: str, value, new_value, done: Set[int]):
        if isinstance(value, Config):
            if id(value) in done:
                return
            done.add(id(value))

            old_id = value.__xpm__.identifier.all.hex()
            new_id = new_value.__xpm__.identifier.all.hex()

            if new_id != old_id:
                print(f"{path} differ: {new_id} vs {old_id}")

                for arg in value.__xpmtype__.arguments.values():
                    arg_value = getattr(value, arg.name)
                    arg_newvalue = getattr(new_value, arg.name)
                    check(f"{path}/{arg.name}", arg_value, arg_newvalue, done)

        elif isinstance(value, list):
            for ix, (array_value, array_newvalue) in enumerate(zip(value, new_value)):
                check(f"{path}.{ix}", array_value, array_newvalue, done)

        elif isinstance(value, dict):
            for key, dict_value in value.items():
                check(f"{path}.{key}", dict_value, new_value[key], done)

    check(".", job, new_job, set())


@click.option("--show-all", is_flag=True, help="Show even not orphans")
@click.option(
    "--ignore-old", is_flag=True, help="Ignore old jobs for unfinished experiments"
)
@click.option("--clean", is_flag=True, help="Prune the orphan folders")
@click.option("--size", is_flag=True, help="Show size of each folder")
@click.argument("path", type=Path, callback=check_xp_path)
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


def arg_split(ctx, param, value):
    # split columns by ',' and remove whitespace
    return set(c.strip() for c in value.split(","))


@click.option("--skip", default=set(), callback=arg_split)
@click.argument("package", type=str)
@click.argument("objects", type=Path)
@cli.command()
def check_documentation(objects, package, skip):
    """Check that all the configuration and tasks are documented within a
    package, relying on the sphinx objects.inv file"""
    from experimaestro.tools.documentation import documented_from_objects, undocumented

    documented = documented_from_objects(objects)
    errors, configs = undocumented([package], documented, skip)
    for config in configs:
        cprint(f"{config.__module__}.{config.__qualname__}", "red")

    if errors > 0 or configs:
        sys.exit(1)


@click.option("--config", type=Path, help="Show size of each folder")
@click.argument("spec", type=str)
@cli.command()
def find_launchers(config: Optional[Path], spec: str):
    """Find launchers matching a specification"""
    if config is not None:
        launcher_registry.LauncherRegistry.set_config_dir(config)

    print(launcher_registry.find_launcher(spec))


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


@cli.group()
@click.option("--workdir", type=Path, default=None)
@click.option("--workspace", type=str, default=None)
@click.pass_context
def experiments(ctx, workdir, workspace):
    """Manage experiments"""
    ws = find_workspace(workdir=workdir, workspace=workspace)
    path = check_xp_path(None, None, ws.path)
    ctx.obj = path


@experiments.command()
@pass_cfg
def list(workdir: Path):
    for p in (workdir / "xp").iterdir():
        if (p / "jobs.bak").exists():
            cprint(f"[unfinished] {p.name}", "yellow")
        else:
            cprint(p.name, "cyan")
