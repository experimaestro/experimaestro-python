import inspect
import sys
from typing import Union, Dict, Iterator, List, Type as TypingType
from collections import ChainMap
from pathlib import Path
import typing
from docstring_parser.parser import parse
import experimaestro.typingutils as typingutils
from experimaestro.utils import logger
from .arguments import Argument
from enum import Enum
import ast
import textwrap

if sys.version_info.major == 3 and sys.version_info.minor < 9:
    from typing_extensions import _AnnotatedAlias, get_type_hints
else:
    from typing import _AnnotatedAlias, get_type_hints


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
            pass
        elif isinstance(tn, str):
            tn = Identifier(tn)
        self.identifier = tn
        self._description = description

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

    def validate(self):
        raise NotImplementedError(f"validate ({self.__class__})")

    @staticmethod
    def fromType(key):
        """Returns the type object corresponding to the given type"""
        logger.debug("Searching for type %s", key)
        from .objects import Config

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
            return key.__getxpmtype__()

        if inspect.isclass(key):
            if issubclass(key, Enum):
                return EnumType(key)

            if issubclass(key, Config):
                return key.__getxpmtype__()

        t = typingutils.get_list(key)
        if t:
            return ArrayType(Type.fromType(t))

        t = typingutils.get_dict(key)
        if t:
            return DictType(Type.fromType(t[0]), Type.fromType(t[1]))

        raise Exception("No type found for %s", key)


class DeprecatedAttribute:
    def __init__(self, fn):
        self.fn = fn
        self.key = fn.__name__
        self.warned = False

    def __set_name__(self, owner, name):
        self.key = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        raise NotImplementedError(f"{instance} {owner}")

    def __set__(self, instance, value):
        if not self.warned:
            logger.warning(f"Parameter {self.key} is deprecated")
        self.fn(instance, value)


class ObjectType(Type):
    """ObjectType contains class-level information about
    experimaestro configurations and tasks

    Attributes:

    objecttype: The python Type of the associated object
    configtype: The python Type of the configuration object that uses property for arguments
    """

    # Those entries should not be copied in the __dict__
    FORBIDDEN_KEYS = set(("__dict__", "__weakref__"))

    def __init__(
        self,
        tp: type,
        identifier: Union[str, Identifier] = None,
    ):
        """Creates a type"""
        from .objects import Config, TypeConfig

        # Task related attributes
        self.taskcommandfactory = None
        self.task = None
        self._title = None

        # Get the identifier
        if identifier is None and hasattr(tp, "__xpmid__"):
            __xpmid__ = getattr(tp, "__xpmid__")
            if inspect.ismethod(__xpmid__):
                identifier = Identifier(__xpmid__())
            elif "__xpmid__" in tp.__dict__:
                identifier = Identifier(__xpmid__)

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

        self.originaltype = tp
        if not issubclass(tp, Config):
            # Adds Config as a base class if not present
            __bases__ = () if tp.__bases__ == (object,) else tp.__bases__
            __dict__ = dict(tp.__dict__)

            __dict__ = {
                key: value
                for key, value in tp.__dict__.items()
                if key not in ObjectType.FORBIDDEN_KEYS
            }
            self.basetype = type(tp.__name__, (Config,) + __bases__, __dict__)
            self.basetype.__module__ = tp.__module__
            self.basetype.__qualname__ = tp.__qualname__
        else:
            self.basetype = tp

        # Create the type-specific configuration class
        __configbases__ = tuple(
            s.__getxpmtype__().configtype
            for s in tp.__bases__
            if issubclass(s, Config) and (s is not Config)
        ) or (TypeConfig,)

        self.configtype = type("TypeConfig", __configbases__ + (self.basetype,), {})
        self.configtype.__qualname__ = f"{self.basetype.__qualname__}.TypeConfig"
        self.configtype.__module__ = tp.__module__

        # Create the type-specific object class
        # (now, the same as basetype - TODO: remove references)
        self.objecttype = self.basetype  # type: type
        self.basetype._ = self.configtype

        # Return type is used by tasks to change the output
        if hasattr(self.basetype, "taskoutputs") or False:
            self.returntype = get_type_hints(getattr(self.basetype, "taskoutputs")).get(
                "return", typing.Any
            )
        else:
            self.returntype = self.objecttype

        # Registers ourselves
        self.basetype.__xpmtype__ = self
        self.configtype.__xpmtype__ = self

        # Other initializations
        self.__initialized__ = False
        self._runtype = None
        self.annotations = []
        self._deprecated = False

    def addAnnotation(self, annotation):
        assert not self.__initialized__
        self.annotations.append(annotation)

    def getpythontaskcommand(self, pythonpath=None):
        import experimaestro.commandline as commandline

        command = commandline.Command()
        command.add(commandline.CommandPath(pythonpath or sys.executable))
        command.add(commandline.CommandString("-m"))
        command.add(commandline.CommandString("experimaestro"))
        command.add(commandline.CommandString("run"))
        command.add(commandline.CommandParameters())
        commandLine = commandline.CommandLine()
        commandLine.add(command)

        return commandline.CommandLineTask(commandLine)

    def __initialize__(self):
        """Effectively gather configuration information"""
        # Check if not initialized
        if self.__initialized__:
            return
        self.__initialized__ = True

        from .objects import Task

        # Get the module
        module = inspect.getmodule(self.originaltype)
        self._file = Path(inspect.getfile(self.originaltype)).absolute()
        self._module = module.__name__
        self._package = module.__package__

        # The class of the object

        self._arguments = ChainMap(
            {}, *(tp.arguments for tp in self.parents())
        )  # type: ChainMap[Argument, Any]

        # Add arguments from annotations
        for annotation in self.annotations:
            annotation.process(self)

        # Add task
        if self.taskcommandfactory is not None:
            self.task = self.taskcommandfactory(self)
        elif issubclass(self.basetype, Task):
            self.task = self.getpythontaskcommand()

        # Add arguments from type hints
        from .arguments import TypeAnnotation

        if hasattr(self.basetype, "__annotations__"):
            typekeys = set(self.basetype.__dict__.get("__annotations__", {}).keys())
            hints = get_type_hints(self.basetype, include_extras=True)
            for key, typehint in hints.items():
                # Filter out hints from parent classes
                if key in typekeys:
                    options = None
                    if isinstance(typehint, _AnnotatedAlias):
                        for value in typehint.__metadata__:
                            if isinstance(value, TypeAnnotation):
                                options = value(options)
                        if options is not None:
                            try:
                                self.addArgument(
                                    options.create(
                                        key, self.objecttype, typehint.__args__[0]
                                    )
                                )
                            except Exception:
                                logger.error(
                                    "while adding argument %s of %s",
                                    key,
                                    self.objecttype,
                                )
                                raise

    def name(self):
        return f"{self.basetype.__module__}.{self.basetype.__qualname__}"

    def __parsedoc__(self):
        """Parse the documentation"""
        # Initialize the object if needed
        if self._title is not None:
            return
        self.__initialize__()

        # Get description from documentation
        __doc__ = self.basetype.__dict__.get("__doc__", None)
        if __doc__:
            parseddoc = parse(__doc__)
            self._title = parseddoc.short_description
            self._description = parseddoc.long_description
            for param in parseddoc.params:
                argument = self._arguments.get(param.arg_name, None)
                if argument is None:
                    logger.warning(
                        "Found documentation for undeclared argument %s", param.arg_name
                    )
                else:
                    argument.help = param.description

        # Get argument help from annotations (PEP 257)
        parsed = ast.parse(textwrap.dedent(inspect.getsource(self.originaltype)))

        argname = None  # Current argument name
        for node in parsed.body[0].body:
            if isinstance(node, ast.AnnAssign):
                argname = node.target.id
            else:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    argument = self._arguments.get(argname, None)
                    if argument is not None:
                        argument.help = node.value.value

                argname = None

    def deprecate(self):
        if len(self.basetype.__bases__) != 1:
            raise RuntimeError(
                "Deprecated configurations must have only one parent (the new configuration)"
            )
        assert not self._deprecated, "Already deprecated"

        # Uses the parent identifier (and saves the deprecated one for path updates)
        self._deprecated_identifier = self.identifier
        self.identifier = self.basetype.__bases__[0].__getxpmtype__().identifier
        self._deprecated = True

    @property
    def description(self) -> str:
        self.__parsedoc__()
        return self._description

    @property
    def title(self) -> Dict[str, Argument]:
        self.__parsedoc__()
        return self._title or str(self.identifier)

    @property
    def arguments(self) -> Dict[str, Argument]:
        self.__initialize__()
        return self._arguments

    def addArgument(self, argument: Argument):
        self._arguments[argument.name] = argument
        argument.objecttype = self

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
        from .objects import Config, Task

        for tp in self.basetype.__bases__:
            if issubclass(tp, Config) and tp not in [Config, Task]:
                yield tp.__xpmtype__

    def validate(self, value):
        """Ensures that the value is compatible with this type"""
        from .objects import Config, Proxy

        self.__initialize__()

        if value is None:
            return None

        if isinstance(value, Proxy):
            _value = value.__unwrap__()
            _value = self.validate(_value)

            # If this is a simple type, OK
            if isinstance(_value, (int, float, str, bool, Path)):
                return _value

            # Returns the proxy
            return value

        if not isinstance(value, Config):
            raise ValueError(f"{value} is not an experimaestro type or task")

        types = self.basetype

        if not isinstance(value, types):
            raise ValueError("%s is not a subtype of %s" % (value, types))

        if self.task and not value.__xpm__.job:
            raise ValueError("The value must be submitted before giving it")
        return value

    def fullyqualifiedname(self) -> str:
        """Returns the fully qualified (Python) name"""
        return f"{self.basetype.__module__}.{self.basetype.__qualname__}"


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
        if isinstance(value, float):
            import math

            rest, intvalue = math.modf(value)
            if rest != 0:
                raise TypeError(f"Value {value} is not an integer but a float")
            return int(intvalue)

        if not isinstance(value, int):
            raise TypeError(f"Value of type {type(value)} is not an integer")
        return value


@definetype(str)
class StrType(Type):
    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError("value is not a string")
        return str(value)


@definetype(float)
class FloatType(Type):
    def validate(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("value is not a float")
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

        if not isinstance(value, (str, Path)):
            raise TypeError("value is not a pathlike value")
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
        return f"List[{self.type.name()}]"

    def validate(self, value):
        if not isinstance(value, List):
            raise ValueError("value is not a list")

        return [self.type.validate(x) for x in value]

    def __str__(self):
        return f"Array({self.type})"

    def __repr__(self):
        return f"Array({self.type})"


class EnumType(Type):
    def __init__(self, type: typing.Type[Enum]):
        self.type = type

    def validate(self, value):
        assert isinstance(value, self.type), f"{value} is not of type {self.type}"
        return value

    def __str__(self):
        return f"Enum({self.type})"

    def __repr__(self):
        return f"Enum({self.type})"


class DictType(Type):
    def __init__(self, keytype: Type, valuetype: Type):
        self.keytype = keytype
        self.valuetype = valuetype

    def name(self):
        return f"Dict[{self.keytype.name()},{self.valuetype.name()}]"

    def validate(self, value):
        if not isinstance(value, dict):
            raise ValueError("value is not a dict")

        return {
            self.keytype.validate(key): self.valuetype.validate(value)
            for key, value in value.items()
        }

    def __str__(self):
        return f"Dict({self.keytype.name()},{self.valuetype.name()})"

    def __repr__(self):
        return str(self)
