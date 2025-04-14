# Configuration registers

from contextlib import contextmanager
from typing import ClassVar, Dict, Optional, Set, Type, Union

from pathlib import Path
import typing
from omegaconf import DictConfig, OmegaConf, SCMode
import pkg_resources
from experimaestro.utils import logger
from .base import ConnectorConfiguration, TokenConfiguration
from .specs import HostRequirement

if typing.TYPE_CHECKING:
    from experimaestro.launchers import Launcher
    from experimaestro.tokens import Token


class LauncherNotFoundError(Exception):
    pass


Connectors = Dict[str, Dict[str, ConnectorConfiguration]]
Tokens = Dict[str, Dict[str, TokenConfiguration]]


def load_yaml(schema, path: Path):
    if not path.is_file():
        return {}

    logger.debug("Loading %s", path)
    with path.open("rt") as fp:
        cfg = OmegaConf.load(fp)
        return OmegaConf.to_container(
            OmegaConf.merge(cfg, schema), structured_config_mode=SCMode.INSTANTIATE
        )


@contextmanager
def ensure_enter(fp):
    """Behaves as a resource, whether it is one or not"""
    if hasattr(fp, "__enter__"):
        with fp as _fp:
            yield _fp
    else:
        yield fp


class LauncherRegistry:
    INSTANCES: ClassVar[Dict[Path, "LauncherRegistry"]] = {}
    CURRENT_CONFIG_DIR: ClassVar[Optional[Path]] = None

    @staticmethod
    def instance():
        """Returns an instance for the current configuration directory"""
        if LauncherRegistry.CURRENT_CONFIG_DIR is None:
            LauncherRegistry.CURRENT_CONFIG_DIR = Path(
                "~/.config/experimaestro"
            ).expanduser()

        if LauncherRegistry.CURRENT_CONFIG_DIR not in LauncherRegistry.INSTANCES:
            LauncherRegistry.INSTANCES[
                LauncherRegistry.CURRENT_CONFIG_DIR
            ] = LauncherRegistry(LauncherRegistry.CURRENT_CONFIG_DIR)

        return LauncherRegistry.INSTANCES[LauncherRegistry.CURRENT_CONFIG_DIR]

    @staticmethod
    def set_config_dir(config_dir: Path):
        LauncherRegistry.CURRENT_CONFIG_DIR = config_dir

    def __init__(self, basepath: Path):
        self.connectors_schema = DictConfig({})
        self.tokens_schema = DictConfig({})
        self.find_launcher_fn = None

        # Use entry points for connectors and launchers
        for entry_point in pkg_resources.iter_entry_points("experimaestro.connectors"):
            entry_point.load().init_registry(self)

        for entry_point in pkg_resources.iter_entry_points("experimaestro.tokens"):
            entry_point.load().init_registry(self)

        # Register the find launcher function if it exists
        launchers_py = basepath / "launchers.py"
        if launchers_py.is_file():
            logger.info("Loading %s", launchers_py)

            from importlib import util

            with ensure_enter(launchers_py.__fspath__()) as fp:
                spec = util.spec_from_file_location("xpm_launchers_conf", fp)
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

            self.find_launcher_fn = getattr(module, "find_launcher", None)
            if self.find_launcher_fn is None:
                logger.warning(
                    "No find_launcher() function was found in %s", launchers_py
                )

        # Read the configuration file
        self.connectors = load_yaml(
            self.connectors_schema, basepath / "connectors.yaml"
        )
        self.tokens = load_yaml(self.tokens_schema, basepath / "tokens.yaml")

    def register_connector(self, identifier: str, cls: Type):
        self.connectors_schema.merge_with({identifier: cls})

    def register_token(self, identifier: str, cls: Type):
        self.tokens_schema.merge_with({identifier: cls})

    def getToken(self, identifier: str) -> "Token":
        for tokens in self.tokens.values():
            if identifier in tokens:
                return tokens[identifier].create(self, identifier)
        raise AssertionError(f"No token with identifier {identifier}")

    def getConnector(self, identifier: str):
        for connectors in self.connectors.values():
            if identifier in connectors:
                return connectors[identifier].create(self)

        # Default local connector
        if identifier == "local":
            from experimaestro.connectors.local import LocalConnector

            return LocalConnector.instance()

        raise AssertionError(f"No connector with identifier {identifier}")

    def find(
        self, *input_specs: Union[HostRequirement, str], tags: Set[str] = set()
    ) -> Optional["Launcher"]:
        """ "
        Arguments:
            spec: The processing requirements
            tags: Restrict the launchers to those containing one of the specified tags
        """

        if self.find_launcher_fn is None:
            logger.info("No launchers.yaml file: using local host ")
            from experimaestro.launchers.direct import DirectLauncher
            from experimaestro.connectors.local import LocalConnector

            return DirectLauncher(LocalConnector.instance())

        # Parse specs
        from .parser import parse

        specs = []
        for spec in input_specs:
            if isinstance(spec, str):
                specs.extend(parse(spec))
            else:
                specs.append(spec)

        # Use launcher function
        from experimaestro.launchers import Launcher

        if self.find_launcher_fn is not None:
            for spec in specs:
                if launcher := self.find_launcher_fn(spec, tags):
                    assert isinstance(
                        launcher, Launcher
                    ), "f{self.find_launcher_fn} did not return a Launcher but {type(launcher)}"
                    return launcher

        return None


def find_launcher(
    *specs: Union[HostRequirement, str], tags: Set[str] = set()
) -> "Launcher":
    """Find a launcher matching a given specification"""
    launcher = LauncherRegistry.instance().find(*specs, tags=tags)
    if not launcher:
        raise LauncherNotFoundError(
            f"No launcher with specification: {specs}."
            "Please refer to the documentation at the following URL: "
            "https://experimaestro-python.readthedocs.io/en/latest/launchers/"
        )
    return launcher
