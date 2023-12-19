import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, Dict

import click
import omegaconf
import yaml
from experimaestro import LauncherRegistry, RunMode, experiment
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.settings import get_workspace
from omegaconf import OmegaConf, SCMode
from termcolor import cprint


class ExperimentHelper:
    """Helper for experiments"""

    # The experiment
    xp: experiment

    #: Run function
    callable: "ExperimentCallable"

    def __init__(self, callable: "ExperimentCallable"):
        self.callable = callable

    """Handles extra arguments"""

    def run(self, args: List[str], configuration: ConfigurationBase):
        assert len(args) == 0
        self.callable(self, configuration)

    @classmethod
    def decorator(cls, *args, **kwargs):
        """Decorator for the run(helper, configuration) method"""
        if len(args) == 1 and len(kwargs) == 0 and inspect.isfunction(args[0]):
            return cls(callable)

        def wrapper(callable):
            return cls(callable)

        return wrapper


class ExperimentCallable(Protocol):
    """Protocol for the run function"""

    def __call__(self, helper: ExperimentHelper, configuration: Any):
        ...


def load(yaml_file: Path):
    """Loads a YAML file, and parents one if they exist"""
    if not yaml_file.exists() and yaml_file.suffix != ".yaml":
        yaml_file = yaml_file.with_suffix(".yaml")

    with yaml_file.open("rt") as fp:
        _data = yaml.full_load(fp)
    data = [_data]
    if parent := _data.get("parent", None):
        data.extend(load(yaml_file.parent / parent))

    return data


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--show", is_flag=True, help="Print configuration and exits")
@click.option(
    "--env",
    help="Define one environment variable",
    type=(str, str),
    multiple=True,
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Server hostname (default to localhost,"
    " not suitable if your jobs are remote)",
)
@click.option(
    "--run-mode",
    type=click.Choice(RunMode),
    default=RunMode.NORMAL,
    help="Sets the run mode",
)
@click.option(
    "--xpm-config-dir",
    type=Path,
    default=None,
    help="Path for the experimaestro config directory "
    "(if not specified, use $HOME/.config/experimaestro)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port for monitoring (can be defined in the settings.yaml file)",
)
@click.option(
    "--file", "xp_file", help="The file containing the main experimental code"
)
@click.option(
    "--workdir",
    type=str,
    default=None,
    help="Working directory - if None, uses the default XPM " "working directory",
)
@click.option("--conf", "-c", "extra_conf", type=str, multiple=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.argument("yaml_file", metavar="YAML file", type=str)
@click.command()
def experiments_cli(
    yaml_file: str,
    xp_file: str,
    host: str,
    port: int,
    xpm_config_dir: Path,
    workdir: Optional[Path],
    env: List[Tuple[str, str]],
    run_mode: RunMode,
    extra_conf: List[str],
    args: List[str],
    show: bool,
    debug: bool,
):
    """Run an experiment"""
    # --- Set the logger
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger("xpm.hash").setLevel(logging.INFO)

    # --- Loads the YAML
    yamls = load(Path(yaml_file))

    # --- Get the XP file
    if xp_file is None:
        for data in yamls:
            if xp_file := data.get("file"):
                del data["file"]
                break

        if xp_file is None:
            raise ValueError("No experiment file given")

    # --- Set some options

    if xpm_config_dir is not None:
        assert xpm_config_dir.is_dir()
        LauncherRegistry.set_config_dir(xpm_config_dir)

    # --- Loads the XP file
    xp_file = Path(xp_file)
    if not xp_file.exists() and xp_file.suffix != ".py":
        xp_file = xp_file.with_suffix(".py")
    xp_file = Path(yaml_file).parent / xp_file

    with open(xp_file, "r") as f:
        source = f.read()
    if sys.version_info < (3, 9):
        the__file__ = str(xp_file)
    else:
        the__file__ = str(xp_file.absolute())

    code = compile(source, filename=the__file__, mode="exec")
    _locals: Dict[str, Any] = {}

    sys.path.append(str(xp_file.parent.absolute()))
    try:
        exec(code, _locals, _locals)
    finally:
        sys.path.pop()

    # --- ... and runs it
    helper = _locals.get("run", None)
    if helper is None:
        raise ValueError(f"Could not find run function in {the__file__}")

    if not isinstance(helper, ExperimentHelper):
        helper = ExperimentHelper(helper)

    parameters = inspect.signature(helper.callable).parameters
    list_parameters = list(parameters.values())
    assert len(list_parameters) == 2, (
        "Callable function should only "
        f"have two arguments (got {len(list_parameters)})"
    )

    schema = list_parameters[1].annotation
    omegaconf_schema = OmegaConf.structured(schema())

    configuration = OmegaConf.merge(*yamls)
    if extra_conf:
        configuration.merge_with(OmegaConf.from_dotlist(extra_conf))
    if omegaconf_schema is not None:
        try:
            configuration = OmegaConf.merge(omegaconf_schema, configuration)
        except omegaconf.errors.ConfigKeyError as e:
            cprint(f"Error in configuration:\n\n{e}", "red", file=sys.stderr)
            sys.exit(1)

    # Move to an object container
    configuration: schema = OmegaConf.to_container(
        configuration, structured_config_mode=SCMode.INSTANTIATE
    )

    if show:
        print(json.dumps(OmegaConf.to_container(configuration)))  # noqa: T201
        sys.exit(0)

    # Get the working directory
    if workdir is None or not Path(workdir).is_dir():
        workdir = get_workspace(workdir).path.expanduser().resolve()
        logging.info("Using working directory %s", workdir)

    # --- Runs the experiment
    with experiment(
        workdir, configuration.id, host=host, port=port, run_mode=run_mode
    ) as xp:
        # Set up the environment
        for key, value in env:
            xp.setenv(key, value)

        # Run the experiment
        helper.xp = xp
        helper.run(list(args), configuration)

        # ... and wait
        xp.wait()
