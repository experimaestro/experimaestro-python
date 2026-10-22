# flake8: noqa: T201
"""``experimaestro tools``: helpers for developers of experimaestro-based code"""

import logging
import sys
from pathlib import Path

import click
from termcolor import cprint

import experimaestro.launcherfinder.registry as launcher_registry

from . import check_xp_path
from .refactor import refactor
from .skill import install_skill


@click.group()
def tools():
    """Developer tools: documentation, launchers, refactoring, identifiers"""


tools.add_command(refactor)
tools.add_command(install_skill)


@tools.group()
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

    _, job = load_job(path / "params.json", discard_id=False)
    _, new_job = load_job(path / "params.json")

    def check(path: str, value, new_value, done: set[int]):
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


def arg_split(ctx, param, value):
    # split columns by ',' and remove whitespace
    return set(c.strip() for c in value.split(","))


@click.option("--skip", default=set(), callback=arg_split)
@click.argument("package", type=str)
@click.argument("objects", type=Path)
@tools.command()
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


@click.option(
    "--config",
    type=Path,
    help="Configuration directory containing launchers.py"
    " (defaults to ~/.config/experimaestro)",
)
@click.argument("spec", type=str)
@tools.command()
def find_launchers(config: Path | None, spec: str):
    """Find launchers matching a specification"""
    if config is not None:
        launcher_registry.LauncherRegistry.set_config_dir(config)

    print(launcher_registry.find_launcher(spec))
