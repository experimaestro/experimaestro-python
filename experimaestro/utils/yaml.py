import logging
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar
import typing
from yaml import Loader, Dumper, MappingNode

from experimaestro import typingutils

logger = logging.getLogger("xpm.yaml")


class YAMLException(Exception):
    def __init__(self, key: str, name: str, line: int, column: int, message: str):
        super().__init__(
            f"Exception while setting {key} in {name}:{line}:{column}: {message}"
        )


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


def register_yaml(
    cls, yaml_loader: Type[Loader], yaml_dumper: Optional[Type[Dumper]] = None
):
    yaml_tag = getattr(cls, "yaml_tag", None)
    if yaml_tag is None and getattr(cls, "from_yaml", None):
        cls.yaml_tag = f"!{cls.__module__}.{cls.__qualname__}"
        logger.debug("Registering constructor %s / cls %s", cls.yaml_tag, cls.from_yaml)
        yaml_loader.add_constructor(cls.yaml_tag, cls.from_yaml)
        if yaml_dumper and (to_yaml := getattr(cls, "to_yaml", None)):
            logger.debug("Registering YAML dumper for %s", cls)
            yaml_dumper.add_representer(cls, to_yaml)
        return True

    # None when no from_yaml
    return yaml_tag is not None


class YAMLDataClass:
    """Base class for dataclass driven YAML objects"""

    __dataclass_fields__: ClassVar[Dict[str, Any]]
    yaml_tag: ClassVar[str]

    @classmethod
    def to_yaml(cls, dumper: Dumper, data):
        return dumper.represent_dict(
            {key: getattr(data, key) for key, value in cls.__dataclass_fields__.items()}
        )

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
                        for x in typingutils.get_args(fieldtype)
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


class YAMLList(List[T]):
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
        (list_element_type,) = typingutils.get_args(list_type)
        return list_element_type


class YAMLDict(Dict[str, T]):
    @classmethod
    def from_yaml(cls, loader, node):
        map = {}
        ctype = cls.get_ctype()

        for node_key, node_value in node.value:
            key = loader.construct_scalar(node_key)
            value = loader.construct_object(node_value, deep=True)
            assert isinstance(value, ctype), f"{value} is not of type {ctype}"
            map[key] = value

        print(map)
        return cls(**map)

    @classmethod
    def get_ctype(cls):
        (list_type,) = [
            p for p in cls.__orig_bases__ if typing.get_origin(p) is YAMLDict
        ]
        (value_type,) = typingutils.get_args(list_type)
        return value_type


class Initialize:
    def __init__(self, fn):
        self.fn = fn


def add_path_resolvers(
    loader: Type[Loader],
    path: List[Optional[str]],
    cls: typing.Type,
    dumper: Optional[Type[Dumper]] = None,
):
    cls = typingutils.get_type(cls) or cls

    if list_type := typingutils.get_list(cls):
        add_path_resolvers(loader, path + [None], list_type, dumper=dumper)

    elif dict_type := typingutils.get_dict(cls):
        key_type, value_type = dict_type
        assert issubclass(key_type, str), f"Key of dict must be a string ({key_type})"
        add_path_resolvers(loader, path + [None], value_type, dumper=dumper)

    else:
        if register_yaml(cls, loader, yaml_dumper=dumper):
            logger.debug(f"Adding {cls.yaml_tag} -> {path}")

            if issubclass(cls, YAMLDataClass):
                # Add others
                loader.add_path_resolver(cls.yaml_tag, path, dict)
                for field in cls.__dataclass_fields__.values():
                    # logger.debug("Processing %s", field)
                    field_type = typingutils.get_type(field.type) or field.type
                    add_path_resolvers(
                        loader, path + [field.name], field_type, dumper=dumper
                    )

            elif issubclass(cls, YAMLDict):
                loader.add_path_resolver(cls.yaml_tag, path + [None], dict)
                add_path_resolvers(
                    loader, path + [None], cls.get_ctype(), dumper=dumper
                )

            elif issubclass(cls, YAMLList):
                loader.add_path_resolver(cls.yaml_tag, path, list)
                add_path_resolvers(
                    loader, path + [None], cls.get_ctype(), dumper=dumper
                )
