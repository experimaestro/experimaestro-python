# Launcher registry

from ast import Str
from dataclasses import dataclass, field as dataclass_field
import itertools
from typing import Annotated, Any, ClassVar, Dict, List, Optional, Type, Union
from pathlib import Path
import typing
from experimaestro import typingutils
import pkg_resources
import humanfriendly
from yaml import Loader, YAMLObject
from experimaestro.utils import logger

if typing.TYPE_CHECKING:
    from experimaestro.launchers import Launcher


class LauncherSpec:
    """Specifications for a launcher"""

    cuda_memory: int
    """CUDA memory"""

    def __init__(self, cuda_memory: Optional[str] = None):
        self.cuda_memory = humanfriendly.parse_size(cuda_memory) if cuda_memory else 0


class LauncherLoader(Loader):
    def __init__(self, registry: "LauncherRegistry", stream):
        super().__init__(stream)
        self.registry = registry


class YAMLException(Exception):
    def __init__(self, key: str, name: str, line: int, column: int, message: str):
        super().__init__(f"Exception while setting {key} in {name}:{line}:{column}")


def check_type(object, typehint):
    typehint = typingutils.get_type(typehint)
    if list_type := typingutils.get_list(typehint):
        assert isinstance(object, list), f"{object} is not a list"
        for el in object:
            check_type(el, list_type)
    else:
        assert isinstance(object, typehint)


class YAMLDataClass(YAMLObject):
    """Base class for dataclass driven YAML objects"""

    yaml_loader = LauncherLoader
    yaml_tag: ClassVar[str]
    __dataclass_fields__: ClassVar[Dict[str, Any]]

    @classmethod
    def from_yaml(cls, loader, node):
        kwargs = {}
        for key, value in node.value:
            try:
                assert isinstance(key.value, str)
                assert (
                    key.value in cls.__dataclass_fields__
                ), f"{key.value} is not a valid field for {cls}"
                fieldtype = cls.__dataclass_fields__[key.value].type
                # print(key, value)
                v = None
                if list_type := typingutils.get_list(fieldtype):
                    assert isinstance(value.value, list), f"{value.value} is not a list"
                    v = []
                    for element in value.value:
                        # print("Constructing element object for list of key", key.value, list_type, element)
                        # print("Constructing object from", element)
                        v.append(loader.construct_object(element))
                        assert isinstance(
                            v[-1], list_type
                        ), f"Element of {v[-1]} is not of type {list_type}"
                elif dict_type := typingutils.get_dict(fieldtype):
                    # print(key, "--->", value.value)
                    _, value_type = dict_type
                    v = {}
                    for dict_key, dict_value in value.value:
                        o = loader.construct_object(dict_value)
                        check_type(o, value_type)
                        v[dict_key.value] = o
                        # assert isinstance(o, value_type), f"Element {o} is not of type {value_type}"
                        # print(key.value, "--->", dict_key.value, dict_value)

                elif isinstance(value.value, (str, int, float)):
                    v = value.value
                else:
                    # print("Constructing object from", value)
                    v = loader.construct_object(value)

                # assert isinstance(v, fieldtype), f"{v} is not of type {fieldtype}"
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


class Initialize:
    def __init__(self, fn):
        self.fn = fn


@dataclass
class GPU(YAMLDataClass):
    """Represents a GPU"""

    yaml_tag = "!gpu"
    type: str
    count: str
    memory: Annotated[int, Initialize(humanfriendly.parse_size)]


@dataclass
class CPU(YAMLDataClass):
    """Represents a CPU"""

    yaml_tag = "!cpu"
    count: str
    memory: Annotated[int, Initialize(humanfriendly.parse_size)]


@dataclass
class Host(YAMLDataClass):
    yaml_tag = "!host"
    name: str
    gpus: List[GPU]
    launchers: List[str]


def add_path_resolvers(path: List[Optional[str]], cls: type[YAMLDataClass]):
    # Add ourself
    cls = typingutils.get_type(cls) or cls

    if list_type := typingutils.get_list(cls):
        add_path_resolvers(path + [None], list_type)
    elif dict_type := typingutils.get_dict(cls):
        key_type, value_type = dict_type
        assert issubclass(key_type, str), f"Key of dict must be a string ({key_type})"
        add_path_resolvers(path + [None], value_type)

    elif issubclass(cls, YAMLDataClass):
        assert cls.yaml_tag is not None, f"yaml_tag not defined for {cls}"
        logger.debug(f"Adding {cls.yaml_tag} -> {path}")
        LauncherLoader.add_path_resolver(cls.yaml_tag, path, dict)

        # Add others
        for field in cls.__dataclass_fields__.values():
            # logger.debug("Processing %s", field)
            field_type = typingutils.get_type(field.type) or field.type
            add_path_resolvers(path + [field.name], field_type)


class LauncherConfiguration:
    """Generic class for a launcher configuration"""

    def fullfills_spec(self, spec: "LauncherSpec"):
        pass


class ConnectorConfiguration:
    pass


class TokenConfiguration:
    pass


@dataclass
class Configuration(YAMLDataClass):
    yaml_tag = "!configuration"
    launchers: Dict[str, List[LauncherConfiguration]]
    connectors: Dict[str, List[ConnectorConfiguration]]
    tokens: Dict[str, List[TokenConfiguration]] = dataclass_field(
        default_factory=lambda: {}
    )


class LauncherRegistry:
    INSTANCE: ClassVar[Optional["LauncherRegistry"]] = None

    @staticmethod
    def instance():
        if LauncherRegistry.INSTANCE is None:
            LauncherRegistry.INSTANCE = LauncherRegistry()

        return LauncherRegistry.INSTANCE

    def __init__(self):
        self.LauncherLoader = LauncherLoader
        add_path_resolvers([], Configuration)

        # Use entry points for connectors and launchers
        for entry_point in pkg_resources.iter_entry_points("experimaestro.connectors"):
            entry_point.load().init_registry(self)

        for entry_point in pkg_resources.iter_entry_points("experimaestro.launchers"):
            entry_point.load().init_registry(self)

        # Read the configuration file
        path = Path("~/.config/experimaestro/launchers.yaml").expanduser()

        # self.configuration = yaml.load(path.read_text(), Loader=LauncherLoader)
        with path.open("rt") as fp:
            loader = LauncherLoader(self, fp)
            try:
                self.configuration: Configuration = loader.get_single_data()
            finally:
                loader.dispose()

    def register_launcher(self, identifier: str, cls: Type[YAMLDataClass]):
        add_path_resolvers(["launchers", identifier, None], cls)

    def register_connector(self, identifier: str, cls: Type[YAMLDataClass]):
        add_path_resolvers(["connectors", identifier, None], cls)

    def find(self, spec: Optional[LauncherSpec]) -> List["Launcher"]:
        launchers = []
        for launcher in itertools.chain(*self.configuration.launchers.values()):
            if spec is None or launcher.fullfills_spec(spec):
                launchers.append(launcher)

        return launchers


def find_launcher(spec: Optional[LauncherSpec]) -> List["Launcher"]:
    """Find a launcher matching a given specification"""
    return LauncherRegistry.instance().find(spec)
