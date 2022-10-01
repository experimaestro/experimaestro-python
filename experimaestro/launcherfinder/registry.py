# Launcher registry

from dataclasses import dataclass
import itertools
from types import new_class
from typing import Annotated, Any, ClassVar, Dict, List, Optional, Type, TypeVar
from pathlib import Path
import typing
from experimaestro import typingutils
import pkg_resources
import humanfriendly
import yaml
from yaml import Loader
from experimaestro.utils import logger

from .base import LauncherConfiguration, ConnectorConfiguration, TokenConfiguration
from .specs import CPUSpecification, CudaSpecification, HostRequirement

if typing.TYPE_CHECKING:
    from experimaestro.launchers import Launcher
    from experimaestro.tokens import Token


class YAMLException(Exception):
    def __init__(self, key: str, name: str, line: int, column: int, message: str):
        super().__init__(
            f"Exception while setting {key} in {name}:{line}:{column}: {message}"
        )


class LauncherNotFoundError(Exception):
    pass


def check_type(object, typehint):
    typehint = typingutils.get_type(typehint)
    if list_type := typingutils.get_list(typehint):
        assert isinstance(object, list), f"{object} is not a list"
        for el in object:
            check_type(el, list_type)
    elif dict_type := typingutils.get_dict(typehint):
        assert isinstance(object, dict)
        for key, value in object.items():
            check_type(key, dict_type[0])
            check_type(value, dict_type[1])
    else:
        assert isinstance(
            object, typehint
        ), f"{type(object)} is not a subtype of {typehint}"


def register_yaml(cls, yaml_loader: Type[Loader]):
    yaml_tag = getattr(cls, "yaml_tag", None)
    if yaml_tag is None and getattr(cls, "from_yaml", None):
        cls.yaml_tag = f"!{cls.__module__}.{cls.__qualname__}"
        logger.debug("Registering constructor %s", cls.yaml_tag)
        yaml_loader.add_constructor(cls.yaml_tag, cls.from_yaml)
        return True

    # None when no from_yaml
    return yaml_tag is not None


class YAMLDataClass:
    """Base class for dataclass driven YAML objects"""

    __dataclass_fields__: ClassVar[Dict[str, Any]]

    @classmethod
    def from_yaml(cls, loader: Loader, node):
        kwargs = {}
        for key, value in node.value:
            try:
                assert isinstance(key.value, str)
                assert (
                    key.value in cls.__dataclass_fields__
                ), f"{key.value} is not a valid field for {cls}"

                v = loader.construct_object(value, deep=True)

                fieldtype = cls.__dataclass_fields__[key.value].type
                if typingutils.is_annotated(fieldtype):
                    initializers = [
                        x
                        for x in typing.get_args(fieldtype)
                        if isinstance(x, Initialize)
                    ]
                    assert (
                        len(initializers) <= 1
                    ), "Too many initializers for this field"
                    if initializers:
                        v = initializers[0].fn(v)

                # Cast if needed
                origin = typingutils.get_type(fieldtype)
                origin = typing.get_origin(origin) or origin
                if origin and not isinstance(v, origin):
                    v = origin(v)

                check_type(v, fieldtype)
                kwargs[key.value] = v
            except YAMLException:
                raise
            except Exception as e:
                raise YAMLException(
                    key.value,
                    node.start_mark.name,
                    node.start_mark.line,
                    node.start_mark.column,
                    str(e),
                )

        return cls(**kwargs)


T = TypeVar("T")


class YAMLList(list[T]):
    @classmethod
    def from_yaml(cls, loader, node):
        array = [loader.construct_object(el, deep=True) for el in node.value]
        ctype = cls.get_ctype()
        for el in array:
            assert isinstance(el, ctype)

        return cls(array)

    @classmethod
    def get_ctype(cls):
        (list_type,) = [
            p for p in cls.__orig_bases__ if typing.get_origin(p) is YAMLList
        ]
        (list_element_type,) = typing.get_args(list_type)
        return list_element_type


class Initialize:
    def __init__(self, fn):
        self.fn = fn


def add_path_resolvers(
    loader: Type[Loader], path: List[Optional[str]], cls: typing.Type
):
    cls = typingutils.get_type(cls) or cls

    if list_type := typingutils.get_list(cls):
        add_path_resolvers(loader, path + [None], list_type)
    elif dict_type := typingutils.get_dict(cls):
        key_type, value_type = dict_type
        assert issubclass(key_type, str), f"Key of dict must be a string ({key_type})"
        add_path_resolvers(loader, path + [None], value_type)

    else:
        if register_yaml(cls, loader):
            logger.debug(f"Adding {cls.yaml_tag} -> {path}")

            if issubclass(cls, YAMLDataClass):
                # Add others
                loader.add_path_resolver(cls.yaml_tag, path, dict)
                for field in cls.__dataclass_fields__.values():
                    # logger.debug("Processing %s", field)
                    field_type = typingutils.get_type(field.type) or field.type
                    add_path_resolvers(loader, path + [field.name], field_type)

            elif issubclass(cls, YAMLList):
                loader.add_path_resolver(cls.yaml_tag, path, list)
                add_path_resolvers(loader, path + [None], cls.get_ctype())


# --- Useful dataclasses


@dataclass
class GPU(YAMLDataClass):
    """Represents a GPU"""

    model: str
    count: int
    memory: Annotated[int, Initialize(humanfriendly.parse_size)]

    def to_spec(self):
        return [CudaSpecification(self.memory, self.model) for _ in range(self.count)]


class GPUList(YAMLList[GPU]):
    """Represents a list of GPUs"""

    def __repr__(self):
        return f"GPUs({super().__repr__()})"

    def to_spec(self) -> List[CudaSpecification]:
        return list(itertools.chain(*[gpu.to_spec() for gpu in self]))


@dataclass
class CPU(YAMLDataClass):
    """Represents a CPU"""

    memory: Annotated[int, Initialize(humanfriendly.parse_size)] = 0
    cores: int = 1

    def to_spec(self):
        return CPUSpecification(self.memory, self.cores)


@dataclass
class Host(YAMLDataClass):
    name: str
    gpus: List[GPU]
    launchers: List[str]


Launchers = Dict[str, List[LauncherConfiguration]]
Connectors = Dict[str, List[ConnectorConfiguration]]
Tokens = Dict[str, List[TokenConfiguration]]


def new_loader(name: str) -> Type[Loader]:
    return new_class("LauncherLoader", (yaml.FullLoader,))  # type: ignore


def load_yaml(loader_cls: Type[Loader], path: Path):
    if not path.is_file():
        return None

    with path.open("rt") as fp:
        loader = loader_cls(fp)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


class LauncherRegistry:
    INSTANCE: ClassVar[Optional["LauncherRegistry"]] = None

    @staticmethod
    def instance():
        if LauncherRegistry.INSTANCE is None:
            LauncherRegistry.INSTANCE = LauncherRegistry()

        return LauncherRegistry.INSTANCE

    def __init__(self):
        self.LauncherLoader: Type[Loader] = new_loader("LauncherLoader")
        self.ConnectorLoader: Type[Loader] = new_loader("ConnectorLoader")
        self.TokenLoader: Type[Loader] = new_loader("TokenLoader")

        add_path_resolvers(self.LauncherLoader, [], Dict[str, LauncherConfiguration])

        # Use entry points for connectors and launchers
        for entry_point in pkg_resources.iter_entry_points("experimaestro.connectors"):
            entry_point.load().init_registry(self)

        for entry_point in pkg_resources.iter_entry_points("experimaestro.launchers"):
            entry_point.load().init_registry(self)

        # Read the configuration file
        basepath = Path("~/.config/experimaestro").expanduser()
        self.launchers: Launchers = (
            load_yaml(self.LauncherLoader, basepath / "launchers.yaml") or {}
        )
        self.connectors: Connectors = (
            load_yaml(self.LauncherLoader, basepath / "connectors.yaml") or {}
        )
        self.tokens: Tokens = (
            load_yaml(self.LauncherLoader, basepath / "tokens.yaml") or {}
        )

    def register_launcher(self, identifier: str, cls: Type[YAMLDataClass]):
        add_path_resolvers(self.LauncherLoader, [identifier, None], cls)

    def register_connector(self, identifier: str, cls: Type[YAMLDataClass]):
        add_path_resolvers(self.ConnectorLoader, ["connectors", identifier, None], cls)

    def register_token(self, identifier: str, cls: Type[YAMLDataClass]):
        add_path_resolvers(self.ConnectorLoader, ["connectors", identifier, None], cls)

    def getToken(self, identifier: str) -> "Token":
        raise NotImplementedError()

    def getConnector(self, identifier: str):
        if identifier == "local" and identifier not in self.connectors:
            from experimaestro.connectors.local import LocalConnector

            return LocalConnector.instance()
        return self.connectors[identifier].create(self)

    def find(self, spec: HostRequirement) -> Optional["Launcher"]:
        for handler in itertools.chain(*self.launchers.values()):
            if launcher := handler.get(self, spec):
                return launcher
        return None


def find_launcher(*specs: HostRequirement) -> "Launcher":
    """Find a launcher matching a given specification"""
    launcher = LauncherRegistry.instance().find(*specs)
    if not launcher:
        raise LauncherNotFoundError()
    return launcher
