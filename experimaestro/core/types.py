import inspect
import logging
import sys
from typing import Union, Dict, Iterator, List, Type as TypingType
from collections import ChainMap
from pathlib import Path
from docstring_parser.parser import parse
import experimaestro.typingutils as typingutils
from experimaestro.utils import logger
from typing_extensions import get_type_hints
import typing_extensions
from .objects import Config
from .arguments import Argument


class Identifier:
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return other.name.__eq__(self.name)

    def __getattr__(self, key):
        return self(key)

    def __getitem__(self, key):
        return self(key)

    def __call__(self, key):
        return Identifier(self.name + "." + key)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def identifier(name: Union[str, Identifier]):
    if isinstance(name, Identifier):
        return name
    return Identifier(str(name))


class Type:
    """Any experimaestro type is a child class"""

    DEFINED: Dict[type, "Type"] = {}

    def __init__(self, tn: Union[str, Identifier], description=None):
        if tn is None:
            tn = None
        elif isinstance(tn, str):
            tn = Identifier(tn)
        self.identifier = tn
        self.description = description

    @property
    def ignore(self):
        """Ignore by default"""
        return False

    def __str__(self):
        return "Type({})".format(self.identifier)

    def __repr__(self):
        return "Type({})".format(self.identifier)

    def name(self):
        return str(self.identifier)

    def isArray(self):
        return False

    @staticmethod
    def fromType(key):
        """Returns the type object corresponding to the given type"""
        logger.debug("Searching for type %s", key)

        if key is None:
            return Any

        defined = Type.DEFINED.get(key, None)
        if defined:
            return defined

        if isinstance(key, Type):
            return key

        if isinstance(key, TypeProxy):
            return key()

        if isinstance(key, Config):
            return key.__xpmtype__

        if inspect.isclass(key) and issubclass(key, Config):
            return key.xpmtype()

        t = typingutils.get_list(key)
        if t:
            return ArrayType(Type.fromType(t))

        raise Exception("No type found for %s", key)


class ObjectType(Type):
    """ObjectType contains class-level information about
    experimaestro configurations and tasks
    """

    REGISTERED: Dict[str, TypingType["Config"]] = {}

    @staticmethod
    def removeConfig(t: Config):
        return t.xpmtype().objecttype

    def __init__(
        self,
        tp: type,
        identifier: str = None,
    ):
        """Creates a type"""
        from .objects import Config

        basetype = Config

        # Get the identifier
        if identifier is None and "__xpmid__" in tp.__dict__:
            identifier = Identifier(getattr(tp, "__xpmid__", None))

        package = tp.__module__.lower()
        name = tp.__qualname__.lower()

        if identifier is None:
            qname = f"{package}.{name}"
            assert (
                getattr(sys, "_called_from_test", False) or "<locals>" not in qname
            ), "Configurations should not be within functions"
            identifier = Identifier(qname)

        super().__init__(identifier, None)

        # --- Creates the config type and not config type

        if not issubclass(tp, Config):
            # Makes configtype a subclass of tp and basetype
            # and makes tp an inner class of configtype (configtype.Object)

            objecttype = tp

            self.configtype = type(tp.__name__, (basetype, tp), {})

            self.configtype.Object = tp
        else:

            # 1) Creates tp that gets rids of Config
            objectbases = tuple(
                ObjectType.removeConfig(b) if issubclass(b, Config) else b
                for b in tp.__bases__
                if b not in [Config]
            )
            objecttype = type("Object", objectbases, tp.__dict__.copy())
            objecttype.__module__ = tp.__module__

            # 2) Config type is based on objecttype and tp
            # Removes non config classes in the bases
            configbases = tuple(b for b in tp.__bases__ if issubclass(b, Config)) + (
                objecttype,
            )
            try:
                self.configtype = type(tp.__name__, tuple(configbases), {})
            except TypeError:

                def print_r(bases, sep=""):
                    for basis in bases:
                        if basis != object:
                            logging.error(f"{sep}{basis}")
                            logging.error(basis.__bases__, sep + "  ")

                objecttype.__name__ = "Object"
                objecttype.__qualname__ = f"{tp.__qualname__}.Object"
                logging.error(f" //// {tp}")
                print_r(configbases)
                raise
            self.configtype.Object = objecttype

        self.configtype.__module__ = tp.__module__
        self.configtype.__qualname__ = tp.__qualname__
        self.configtype.__xpmtype__ = self
        self.objecttype = objecttype

        objecttype.__xpmtype__ = self
        objecttype.__name__ = "Object"
        objecttype.__qualname__ = f"{tp.__qualname__}.Object"

        self.originaltype = tp

        # Other initializations
        self.__initialized__ = False
        self._runtype = None
        self.annotations = []

    def addAnnotation(self, annotation):
        assert not self.__initialized__
        self.annotations.append(annotation)

    def __initialize__(self):
        """Effectively parses information"""

        # Check if not initialized
        if self.__initialized__:
            return

        self.__initialized__ = True

        # Get the module
        module = inspect.getmodule(self.originaltype)
        self._file = Path(inspect.getfile(self.originaltype)).absolute()
        self._module = module.__name__
        self._package = module.__package__

        # The class of the object
        self.task = None

        self._arguments = ChainMap({}, *(tp.arguments for tp in self.parents()))

        # Add arguments from annotations
        for annotation in self.annotations:
            annotation.process(self)

        # Get description from documentation
        paramhelp = {}
        if "__doc__" in self.objecttype.__dict__:
            parseddoc = parse(self.objecttype.__doc__)
            self.description = parseddoc.short_description
            for param in parseddoc.params:
                paramhelp[param.arg_name] = param.description

        # Add arguments from type hints
        from .arguments import TypeAnnotation

        if hasattr(self.objecttype, "__annotations__"):
            typekeys = set(self.objecttype.__dict__.get("__annotations__", {}).keys())
            hints = get_type_hints(self.objecttype, include_extras=True)
            for key, typehint in hints.items():
                # Filter out hints from parent classes
                if key in typekeys:
                    options = None
                    _default = getattr(self.objecttype, key, None)
                    if isinstance(typehint, typing_extensions._AnnotatedAlias):
                        for value in typehint.__metadata__:
                            if isinstance(value, TypeAnnotation):
                                options = value(options, _default)
                        if options is not None:
                            if options.kwargs.get("help", None) is None:
                                options.kwargs["help"] = paramhelp.get(key, None)
                            self.addArgument(
                                options.create(
                                    key, self.configtype, typehint.__args__[0]
                                )
                            )

    @property
    def arguments(self):
        self.__initialize__()
        return self._arguments

    def addArgument(self, argument: Argument):
        self._arguments[argument.name] = argument

        # The the attribute for the config type
        setattr(
            self.configtype,
            argument.name,
            property(
                lambda _self: _self.__xpm__.get(argument.name),
                lambda _self, value: _self.__xpm__.set(argument.name, value),
            ),
        )

        # Check default value
        if argument.default is not None:
            argument.type.validate(argument.default)

    def getArgument(self, key: str) -> Argument:
        self.__initialize__()
        return self._arguments[key]

    def parents(self) -> Iterator["ObjectType"]:
        for tp in self.configtype.__bases__:
            if issubclass(tp, Config) and tp not in [Config]:
                yield tp.xpmtype()

    def validate(self, value):
        """Ensures that the value is compatible with this type"""
        self.__initialize__()

        if isinstance(value, dict):
            # This is a unserialized object
            valuetype = value.get("$type", None)
            if valuetype is None:
                raise ValueError("Object has no $type")

            classtype = ObjectType.REGISTERED.get(valuetype, None)
            if classtype:
                try:
                    return classtype(**value)
                except:
                    logger.exception("Could not build object of class %s" % (classtype))
                    raise

            logger.debug("Using argument type (not real type)")
            return self.configtype(**value)

        if value is None:
            return None

        if not isinstance(value, Config):
            raise ValueError(f"{value} is not an experimaestro type or task")

        types = self.configtype

        if not isinstance(value, types):
            raise ValueError("%s is not a subtype of %s" % (value, types))

        if self.task and not value.__xpm__.job:
            raise ValueError("The value must be submitted before giving it")
        return value


class TypeProxy:
    def __call__(self) -> Type:
        """Returns the real type"""
        raise NotImplementedError()


def definetype(*types):
    def call(typeclass):
        instance = typeclass(types[0].__name__)
        for t in types:
            Type.DEFINED[t] = instance
        return typeclass

    return call


@definetype(int)
class IntType(Type):
    def validate(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Value of type {type(value)} is not an integer")
        return value


@definetype(str)
class StrType(Type):
    def validate(self, value):
        return str(value)


@definetype(float)
class FloatType(Type):
    def validate(self, value):
        return float(value)


@definetype(bool)
class BoolType(Type):
    def validate(self, value):
        return bool(value)


@definetype(Path)
class PathType(Type):
    def validate(self, value):
        if isinstance(value, dict) and value.get("$type", None) == "path":
            return Path(value.get("$value"))
        return Path(value)

    @property
    def ignore(self):
        """Ignore by default"""
        return True


class AnyType(Type):
    def __init__(self):
        super().__init__("any")

    def validate(self, value):
        return value


Any = AnyType()


class ArrayType(Type):
    def __init__(self, type: Type):
        self.type = type

    def name(self):
        return f"Array[{self.type.name()}]"

    def validate(self, value):
        if not isinstance(value, List):
            raise ValueError("value is not a list")

        return [self.type.validate(x) for x in value]

    def __str__(self):
        return f"Array({self.type})"

    def __repr__(self):
        return f"Array({self.type})"
