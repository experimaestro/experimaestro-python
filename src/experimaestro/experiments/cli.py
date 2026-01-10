import datetime
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

import click
import omegaconf
import yaml
from omegaconf import OmegaConf, SCMode
from termcolor import cprint

from experimaestro.exceptions import HandledException
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder.registry import LauncherRegistry
from experimaestro.scheduler.workspace import RunMode
from experimaestro.settings import find_workspace
from experimaestro.utils.logging import setup_logging

if TYPE_CHECKING:
    from experimaestro.scheduler.experiment import experiment


class ExperimentHelper:
    """Helper for experiments"""

    xp: "experiment"
    """The experiment object"""

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

    def __call__(self, helper: ExperimentHelper, configuration: Any): ...


class ConfigurationLoader:
    def __init__(self):
        self.yamls = []
        self.python_path = set()
        self.yaml_module_file: None | Path = None

    def load(self, yaml_file: Path):
        """Loads a YAML file, and parents one if they exist"""
        if not yaml_file.exists() and yaml_file.suffix != ".yaml":
            yaml_file = yaml_file.with_suffix(".yaml")

        with yaml_file.open("rt") as fp:
            _data = yaml.full_load(fp)

        if "file" in _data:
            path = Path(_data["file"])
            if not path.is_absolute():
                _data["file"] = str((yaml_file.parent / path).resolve())

        if "pre_experiment" in _data:
            path = Path(_data["pre_experiment"])
            if not path.is_absolute():
                _data["pre_experiment"] = str((yaml_file.parent / path).resolve())

        if "module" in _data:
            # Keeps track of the YAML file where the module was defined
            self.yaml_module_file = yaml_file

        if parent := _data.get("parent", None):
            self.load(yaml_file.parent / parent)

        self.yamls.append(_data)

        for path in _data.get("pythonpath", []):
            path = Path(path)
            if path.is_absolute():
                self.python_path.add(path.resolve())
            else:
                self.python_path.add((yaml_file.parent / path).resolve())


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--show", is_flag=True, help="Print configuration and exits")
@click.option(
    "--watcher",
    type=click.Choice(["auto", "polling", "inotify", "fsevents", "kqueue", "windows"]),
    default="auto",
    help="Filesystem watcher type (auto=platform default, polling=network mounts)",
)
@click.option(
    "--polling-interval",
    type=float,
    default=1.0,
    help="Polling interval in seconds (only for --watcher=polling)",
)
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
    help="[DEPRECATED] Server hostname (use --web instead)",
    hidden=True,
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
    help="Path for the experimaestro config directory (if not specified, use $HOME/.config/experimaestro)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="[DEPRECATED] Port for monitoring (use --web instead)",
    hidden=True,
)
@click.option(
    "--web",
    is_flag=True,
    help="Start web server for monitoring (use settings.yaml for host/port config)",
)
@click.option(
    "--console",
    is_flag=True,
    help="Launch Textual console UI for monitoring with logs",
)
@click.option(
    "--no-db",
    "no_db",
    is_flag=True,
    help="Disable database state tracking for this experiment",
)
@click.option(
    "--file",
    "xp_file",
    type=Path,
    help="The file containing the main experimental code",
)
@click.option(
    "--module-name", "module_name", help="Module containing the experimental code"
)
@click.option(
    "--workspace",
    type=str,
    default=None,
    help="Workspace ID (reads from settings.yaml in experimaestro config)",
)
@click.option(
    "--workdir",
    type=str,
    default=None,
    help="Working environment",
)
@click.option("--conf", "-c", "extra_conf", type=str, multiple=True)
@click.option(
    "--pre-yaml", type=str, multiple=True, help="Add YAML file after the main one"
)
@click.option(
    "--post-yaml", type=str, multiple=True, help="Add YAML file before the main one"
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.argument("yaml_file", metavar="YAML file", type=str)
@click.command()
def experiments_cli(  # noqa: C901
    yaml_file: str,
    xp_file: str,
    host: str,
    port: int,
    web: bool,
    console: bool,
    no_db: bool,
    xpm_config_dir: Path,
    workdir: Optional[Path],
    workspace: Optional[str],
    env: List[Tuple[str, str]],
    run_mode: RunMode,
    extra_conf: List[str],
    pre_yaml: List[str],
    post_yaml: List[str],
    module_name: Optional[str],
    args: List[str],
    show: bool,
    watcher: str,
    polling_interval: float,
    debug: bool,
):
    """Run an experiment"""
    import warnings

    # --- Set the logger with colors if outputting to terminal
    setup_logging(debug=debug)

    # --- Configure filesystem watcher type
    from experimaestro.ipc import IPCom, WatcherType

    if watcher != "auto":
        IPCom.set_watcher_type(WatcherType(watcher), polling_interval)
    elif polling_interval != 1.0:
        # If polling interval is specified but watcher is auto, use polling
        IPCom.set_watcher_type(WatcherType.POLLING, polling_interval)

    # --- Warn about deprecated options
    if host is not None:
        warnings.warn(
            "The '--host' option is deprecated. Use '--web' flag instead. "
            "Configure host in settings.yaml.",
            DeprecationWarning,
            stacklevel=2,
        )
    if port is not None:
        warnings.warn(
            "The '--port' option is deprecated. Use '--web' flag instead. "
            "Configure port in settings.yaml.",
            DeprecationWarning,
            stacklevel=2,
        )

    # --- Loads the YAML
    conf_loader = ConfigurationLoader()
    for y in pre_yaml:
        conf_loader.load(Path(y))
    conf_loader.load(Path(yaml_file))
    for y in post_yaml:
        conf_loader.load(Path(y))

    # --- Merge the YAMLs
    configuration = OmegaConf.merge(*conf_loader.yamls)
    if extra_conf:
        configuration.merge_with(OmegaConf.from_dotlist(extra_conf))

    # --- Get the XP file
    python_path = list(conf_loader.python_path)
    if module_name is None:
        module_name = configuration.get("module", None)

    if xp_file is None:
        xp_file = configuration.get("file", None)
        if xp_file:
            assert not module_name, (
                "Module name and experiment file are mutually exclusive options"
            )
            xp_file = Path(xp_file)
            if not python_path:
                python_path.append(xp_file.parent.absolute())
            logging.info(
                "Using python path: %s", ", ".join(str(s) for s in python_path)
            )

    assert module_name or xp_file, (
        "Either the module name or experiment file should be given"
    )

    # --- Set some options

    if xpm_config_dir is not None:
        assert xpm_config_dir.is_dir()
        LauncherRegistry.set_config_dir(xpm_config_dir)

    # --- Finds the "run" function

    # Modifies the Python path
    for path in python_path:
        sys.path.append(str(path))

    # --- Execute pre-experiment script if specified
    pre_experiment = configuration.get("pre_experiment", None)
    if pre_experiment:
        pre_exp_path = Path(pre_experiment)
        if pre_exp_path.exists():
            logging.info("Executing pre-experiment script: %s", pre_exp_path)
            try:
                spec = importlib.util.spec_from_file_location(
                    "pre_experiment", str(pre_exp_path.absolute())
                )
                pre_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pre_mod)
            except Exception as e:
                raise click.ClickException(
                    f"Failed to execute pre-experiment script '{pre_exp_path}': {e}"
                )
        else:
            raise click.ClickException(
                f"Pre-experiment script not found: {pre_exp_path}"
            )

    # --- Adds automatically the experiment module if not found
    if module_name and conf_loader.yaml_module_file:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            # Try to setup a path
            path = conf_loader.yaml_module_file.resolve()
            for _ in range(len(module_name.split("."))):
                path = path.parent

            logging.info("Appending %s to python path", path)
            sys.path.append(str(path))
            python_path.append(path)

    if xp_file:
        if not xp_file.exists() and xp_file.suffix != ".py":
            xp_file = xp_file.with_suffix(".py")
        xp_file: Path = Path(yaml_file).parent / xp_file
        module_name = xp_file.with_suffix("").name
        spec = importlib.util.spec_from_file_location(
            module_name, str(xp_file.absolute())
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        # Module
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            logging.error("Module not found: %s with python path %s", e, sys.path)
            raise

    helper = getattr(mod, "run", None)

    # --- ... and runs it
    if helper is None:
        raise click.ClickException(
            f"Could not find run function in {xp_file if xp_file else module_name}"
        )

    if not isinstance(helper, ExperimentHelper):
        helper = ExperimentHelper(helper)

    parameters = inspect.signature(helper.callable).parameters
    list_parameters = list(parameters.values())
    if len(list_parameters) != 2:
        raise click.ClickException(
            f"run in {xp_file if xp_file else module_name} function should only "
            f"have two arguments (got {len(list_parameters)}), "
        )

    schema = list_parameters[1].annotation
    if isinstance(schema, str):
        # Get the schema from the module
        schema = getattr(mod.__dict__, schema, None)
        assert schema is not None, (
            f"Could not find schema {list_parameters[1].annotation} "
            f"in module {module_name}"
        )

    omegaconf_schema = OmegaConf.structured(schema())

    if omegaconf_schema is not None:
        try:
            configuration = OmegaConf.merge(omegaconf_schema, configuration)
        except omegaconf.errors.ConfigKeyError as e:
            cprint(f"Error in configuration:\n\n{e}", "red", file=sys.stderr)
            raise click.ClickException("Error in configuration")

    if show:
        print(json.dumps(OmegaConf.to_container(configuration)))  # noqa: T201
        sys.exit(0)

    # Move to an object container
    xp_configuration: ConfigurationBase = OmegaConf.to_container(
        configuration, structured_config_mode=SCMode.INSTANTIATE
    )

    # --- Sets up the experiment ID
    if xp_configuration.add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        experiment_id = f"""{xp_configuration.id}-{timestamp}"""
    else:
        experiment_id = xp_configuration.id

    # Define the workspace (may auto-select based on experiment_id triggers)
    ws_env = find_workspace(
        workdir=workdir, workspace=workspace, experiment_id=experiment_id
    )

    workdir = ws_env.path

    logging.info(
        "Running experiment %s working directory %s",
        experiment_id,
        str(workdir.resolve()),
    )

    # Determine project path for git info
    project_paths = []
    if xp_file:
        project_paths.append(xp_file.resolve().parent)
    elif hasattr(mod, "__file__") and mod.__file__:
        project_paths.append(Path(mod.__file__).resolve().parent)

    # Define the experiment execution function
    def run_experiment_code(
        xp_holder=None, xp_ready_event=None, register_signals=True, in_thread=False
    ):
        """Run the experiment code - optionally storing xp in xp_holder"""
        try:
            from experimaestro.scheduler.experiment import experiment
            from experimaestro.settings import get_settings

            with experiment(
                ws_env,
                experiment_id,
                run_mode=run_mode,
                register_signals=register_signals,
                project_paths=project_paths,
                dirty_git=xp_configuration.dirty_git,
                no_db=no_db,
            ) as xp:
                if xp_holder is not None:
                    xp_holder["xp"] = xp

                # Start web server if requested
                if web and run_mode == RunMode.NORMAL:
                    settings = get_settings()
                    xp.scheduler.start_server(
                        settings.server,
                        workspace=xp.workspace,
                        wait_for_quit=False,
                    )
                    logging.info(
                        "Web server started at http://%s:%d",
                        settings.server.host or "localhost",
                        settings.server.port or 12345,
                    )

                if xp_ready_event is not None:
                    xp_ready_event.set()  # Signal that xp is ready

                # Test logging from experiment thread
                logging.info("Experiment started in background thread")

                # Set up the environment
                for key, value in env:
                    xp.setenv(key, value)

                # Sets the python path
                xp.workspace.python_path.extend(python_path)

                # Run the experiment
                helper.xp = xp
                helper.run(list(args), xp_configuration)

                # ... and wait
                xp.wait()

        except HandledException as e:
            if in_thread:
                # Re-raise to preserve exception info for the main thread
                raise
            cprint(f"Experiment failed: {e}", "red", file=sys.stderr)
            sys.exit(1)

    # Console mode is only available in NORMAL run mode
    use_console = console and run_mode == RunMode.NORMAL
    if console and not use_console:
        logging.warning("--console is ignored when run_mode is not NORMAL")

    if use_console:
        # Start TUI first, then run experiment in background thread
        # This ensures all logs (including startup) are captured
        import threading
        from experimaestro.tui import ExperimentTUI

        # Initialize multiprocessing resource tracker before Textual takes over
        # terminal file descriptors. This prevents "bad value(s) in fds_to_keep"
        # errors when code in the background thread (e.g., tqdm in torchvision)
        # tries to use multiprocessing.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.ensure_running()
        except Exception:
            pass  # Best effort - may not be needed on all systems

        xp_holder = {"xp": None}
        exception_holder = {"exception": None}

        # Create TUI first in deferred mode (no state_provider yet)
        # This allows capturing all logs from experiment startup
        tui_app = ExperimentTUI(show_logs=True)

        def run_experiment_in_thread():
            """Run experiment and connect state provider to TUI when ready"""
            try:
                from experimaestro.scheduler.experiment import experiment as exp_context
                from experimaestro.settings import get_settings

                with exp_context(
                    ws_env,
                    experiment_id,
                    run_mode=run_mode,
                    register_signals=False,  # TUI handles signals
                    project_paths=project_paths,
                    dirty_git=xp_configuration.dirty_git,
                    no_db=no_db,
                ) as xp:
                    xp_holder["xp"] = xp

                    # Connect TUI to the experiment's scheduler
                    tui_app.call_from_thread(tui_app.set_state_provider, xp.scheduler)

                    # Start web server if requested
                    if web:
                        settings = get_settings()
                        xp.scheduler.start_server(
                            settings.server,
                            workspace=xp.workspace,
                            wait_for_quit=False,
                        )
                        logging.info(
                            "Web server started at http://%s:%d",
                            settings.server.host or "localhost",
                            settings.server.port or 12345,
                        )

                    logging.info("Experiment started")

                    # Set up the environment
                    for key, value in env:
                        xp.setenv(key, value)

                    # Sets the python path
                    xp.workspace.python_path.extend(python_path)

                    # Run the experiment
                    helper.xp = xp
                    helper.run(list(args), xp_configuration)

                    # ... and wait
                    xp.wait()

                logging.info("Experiment thread completed")

            except BaseException as e:
                # Use BaseException to also catch SystemExit from sys.exit()
                exception_holder["exception"] = e

        # Start experiment in background thread
        exp_thread = threading.Thread(target=run_experiment_in_thread, daemon=True)
        exp_thread.start()

        try:
            # Run TUI in main thread (handles signals via Textual)
            # Textual automatically captures stdout/stderr via Print events
            tui_app.run()
        finally:
            # TUI exited (user pressed q or Ctrl+C) - stop the experiment
            if xp_holder["xp"]:
                xp_holder["xp"].stop()

            # Wait for experiment thread to finish
            exp_thread.join(timeout=5.0)

        # Handle exceptions
        if exception_holder["exception"]:
            raise exception_holder["exception"]

    else:
        # Normal mode without TUI - run directly
        run_experiment_code()
