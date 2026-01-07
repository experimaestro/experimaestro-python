from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import sys
from typing import (
    Set,
    TypeVar,
    Union,
    Dict,
    Iterator,
    List,
    Optional,
    get_args,
    get_origin,
)
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

from typing import _AnnotatedAlias, get_type_hints

if typing.TYPE_CHECKING:
    from experimaestro.scheduler.base import Job
    from experimaestro.launchers import Launcher
    from experimaestro.core.objects import Config
    from experimaestro.core.partial import Partial


@dataclass
class DeprecationInfo:
    """Information about a deprecated configuration type."""

    #: The original identifier before deprecation
    original_identifier: "Identifier"

    #: The target configuration class to convert to
    target: type

    #: If True, creating an instance immediately converts to the target type
    replace: bool = False


class Identifier:
    def __init__(self, name: str):
        assert isinstance(name, str)
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

        if union_t := typingutils.get_union(key):
            return UnionType([Type.fromType(t) for t in union_t])

        # Takes care of generics, like List[int], not List
        if get_origin(key):
            return GenericType(key)

        if isinstance(key, TypeVar):
            return TypeVarType(key)

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


class SubmitHook(ABC):
    """Hook called before the job is submitted to the scheduler

    This allows modifying e.g. the run environnement
    """

    def __call__(self, cls: typing.Type["Config"]):
        """Decorates a an XPM configuration"""
        cls.__getxpmtype__().submit_hooks.add(self)
        return cls

    @abstractmethod
    def process(self, job: "Job", launcher: "Launcher"):
        """Apply the hook for the job/launcher"""
        ...

    @abstractmethod
    def spec(self):
        """Returns an identifier tuple for hashing/equality"""
        ...

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return False
        return self.__spec__ == other.__spec__

    def __hash__(self):
        return hash((self.__class__, self.spec()))


class XPMValue:
    """Jut marks a XPMValue"""

    pass


class ObjectType(Type):
    submit_hooks: Set[SubmitHook]
    """Hooks associated with this configuration"""

    """ObjectType contains class-level information about
    experimaestro configurations and tasks

    :param value_type: The Python type of the associated object
    :param config_type: The Python type of the configuration object
    """

    def __init__(
        self,
        tp: type,
        identifier: Union[str, Identifier, None] = None,
    ):
        """Creates a type"""
        from .objects import Config, ConfigMixin

        # Task related attributes
        self.taskcommandfactory = None
        self.task = None
        self._title = None
        self.submit_hooks = set()

        # Warning flag for non-resumable task directory cleanup
        self.warned_clean_not_resumable = False

        # --- Get the identifier
        if identifier is None and hasattr(tp, "__xpmid__"):
            __xpmid__ = getattr(tp, "__xpmid__")
            if isinstance(__xpmid__, Identifier):
                identifier = __xpmid__
            elif inspect.ismethod(__xpmid__):
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
        assert issubclass(tp, Config)
        self.value_type = tp

        # --- Create the type-specific configuration class (XPMConfig)
        __configbases__ = tuple(
            s.__getxpmtype__().config_type
            for s in tp.__bases__
            if issubclass(s, Config) and (s is not Config)
        ) or (ConfigMixin,)

        *tp_qual, tp_name = self.value_type.__qualname__.split(".")
        self.config_type = type(
            f"{tp_name}.XPMConfig", __configbases__ + (self.value_type,), {}
        )
        self.config_type.__qualname__ = ".".join(tp_qual + [self.config_type.__name__])
        self.config_type.__module__ = tp.__module__

        # --- Get the return type
        if hasattr(self.value_type, "task_outputs") or False:
            self.returntype = get_type_hints(
                getattr(self.value_type, "task_outputs")
            ).get("return", typing.Any)
        else:
            self.returntype = self.value_type

        # --- Registers ourselves
        self.value_type.__xpmtype__ = self
        self.config_type.__xpmtype__ = self

        # --- Other initializations
        self.__initialized__ = False
        self._runtype = None
        self.annotations = []
        self._deprecation: Optional[DeprecationInfo] = None

        # --- Value class (for external value types, e.g., nn.Module subclasses)
        self._original_type: type = tp  # Keep reference to original config class

        # --- Partial for partial identifier computation
        self._partials: Dict[str, "Partial"] = {}

    def set_value_type(self, value_class: type) -> None:
        """Register an explicit value class for this configuration.

        The value class will be used when creating instances via .instance().
        """
        self.value_type = value_class

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
        self._module = module.__name__
        self._package = module.__package__

        if self._module and self._package:
            self._file = None
        else:
            self._file = Path(inspect.getfile(self.originaltype)).absolute()

        assert (self._module and self._package) or self._file, (
            f"Could not detect module/file for {self.originaltype}"
        )

        # The class of the object

        self._arguments = ChainMap({}, *(tp.arguments for tp in self.parents()))  # type: ChainMap[Argument, Any]

        # Add arguments from annotations
        for annotation in self.annotations:
            annotation.process(self)

        # Add task
        if self.taskcommandfactory is not None:
            self.task = self.taskcommandfactory(self)
        elif issubclass(self._original_type, Task):
            self.task = self.getpythontaskcommand()

        # Add arguments from type hints
        # Use _original_type since value_type may have been overridden by set_value_type
        from .arguments import TypeAnnotation

        if hasattr(self._original_type, "__annotations__"):
            typekeys = set(
                self._original_type.__dict__.get("__annotations__", {}).keys()
            )
            hints = get_type_hints(self._original_type, include_extras=True)
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
                                        key, self._original_type, typehint.__args__[0]
                                    )
                                )
                            except Exception:
                                logger.error(
                                    "while adding argument %s of %s",
                                    key,
                                    self._original_type,
                                )
                                raise

        # Collect partial from class attributes
        from .partial import Partial as PartialClass

        for name, value in self._original_type.__dict__.items():
            if isinstance(value, PartialClass):
                # Auto-set name from attribute name if not already set
                if value.name is None:
                    value.name = name
                self._partials[name] = value

    def name(self):
        return f"{self.value_type.__module__}.{self.value_type.__qualname__}"

    def __parsedoc__(self):
        """Parse the documentation"""
        # Initialize the object if needed
        if self._title is not None:
            return
        self.__initialize__()

        # Get description from documentation
        # Use _original_type since value_type may have been overridden
        __doc__ = self._original_type.__dict__.get("__doc__", None)
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

    def deprecate(self, target=None, replace: bool = False):
        """Mark this configuration type as deprecated.

        Args:
            target: Optional target configuration class. If provided, uses
                    target's identifier. If None, uses parent class's identifier
                    (legacy behavior requiring single inheritance).
            replace: If True, creating an instance of this class immediately
                    returns a converted instance of the target class.

        When a target is specified, the deprecated class should define a
        __convert__ method that returns an equivalent target configuration.
        The identifier is computed from the converted configuration.
        """
        assert self._deprecation is None, "Already deprecated"

        # Save the deprecated identifier for migration tools (fix_deprecated)
        original_identifier = self.identifier

        if target is not None:
            # New mechanism: explicit target class
            target_xpmtype = target.__getxpmtype__()
            self.identifier = target_xpmtype.identifier
            deprecation_target = target
        else:
            # Legacy mechanism: parent class is the target
            if len(self.value_type.__bases__) != 1:
                raise RuntimeError(
                    "Deprecated configurations must have only one parent (the new configuration)"
                )
            parent = self.value_type.__bases__[0].__getxpmtype__()
            self.identifier = parent.identifier
            deprecation_target = self.value_type.__bases__[0]

        self._deprecation = DeprecationInfo(
            original_identifier=original_identifier,
            target=deprecation_target,
            replace=replace,
        )

    @property
    def deprecated(self) -> bool:
        """Returns true if this type is deprecated"""
        return self._deprecation is not None

    @property
    def _deprecated_identifier(self) -> Optional["Identifier"]:
        """Returns the original identifier before deprecation (for backwards compatibility)"""
        return self._deprecation.original_identifier if self._deprecation else None

    @property
    def description(self) -> str:
        self.__parsedoc__()
        return self._description

    @property
    def title(self) -> str:
        self.__parsedoc__()
        return self._title or str(self.identifier)

    @property
    def arguments(self) -> Dict[str, Argument]:
        self.__initialize__()
        return self._arguments

    def addArgument(self, argument: Argument):
        # Check if this argument overrides a parent argument
        # _arguments is a ChainMap where maps[0] is current class, maps[1:] are parents
        parent_argument = None
        for parent_map in self._arguments.maps[1:]:
            if argument.name in parent_map:
                parent_argument = parent_map[argument.name]
                break

        if parent_argument is not None:
            # Check type compatibility (child type should be subtype of parent type)
            self._check_override_type_compatibility(argument, parent_argument)

            # Warn if overrides flag is not set
            if not argument.overrides:
                logger.warning(
                    "Parameter '%s' in %s overrides parent parameter from %s. "
                    "Use field(overrides=True) to suppress this warning.",
                    argument.name,
                    self._original_type.__qualname__,
                    (
                        parent_argument.objecttype._original_type.__qualname__
                        if parent_argument.objecttype
                        else "unknown"
                    ),
                )

        self._arguments[argument.name] = argument
        argument.objecttype = self

        # Check default value
        if argument.default is not None:
            argument.type.validate(argument.default)

    def _check_override_type_compatibility(
        self, child_arg: Argument, parent_arg: Argument
    ):
        """Check that the child argument type is compatible with the parent type.

        For Config types, the child type should be a subtype of the parent type
        (covariant). For other types, we check for exact match.
        """
        child_type = child_arg.type
        parent_type = parent_arg.type

        # Check if both are ObjectType (Config types)
        if isinstance(child_type, ObjectType) and isinstance(parent_type, ObjectType):
            child_pytype = child_type.value_type
            parent_pytype = parent_type.value_type

            # Check if child is a subtype of parent
            if not issubclass(child_pytype, parent_pytype):
                raise TypeError(
                    f"Parameter '{child_arg.name}' type {child_pytype.__qualname__} "
                    f"is not a subtype of parent type {parent_pytype.__qualname__}. "
                    f"Override types must be subtypes of the parent type."
                )
        elif type(child_type) is not type(parent_type):
            # For non-Config types, check for exact type match
            # Different type classes (e.g., IntType vs StrType) are incompatible
            raise TypeError(
                f"Parameter '{child_arg.name}' type {type(child_type).__name__} "
                f"is not compatible with parent type {type(parent_type).__name__}. "
                f"Override types must be the same type or a subtype."
            )
        # Same type class is allowed (e.g., both are IntType)

    def getArgument(self, key: str) -> Argument:
        self.__initialize__()
        return self._arguments[key]

    def parents(self) -> Iterator["ObjectType"]:
        from .objects import Config, Task

        # Use _original_type to avoid issues when value_type has been
        # overridden by set_value_type (the value class would create
        # circular references since it inherits from the config class)
        for tp in self._original_type.__bases__:
            if issubclass(tp, Config) and tp not in [Config, Task]:
                yield tp.__xpmtype__

    def validate(self, value):
        """Ensures that the value is compatible with this type"""
        from .objects import Config

        self.__initialize__()

        if value is None:
            return None

        if not isinstance(value, Config):
            raise ValueError(f"{value} is not an experimaestro type or task")

        types = self.value_type

        if not isinstance(value, types):
            raise ValueError(
                "%s is not a subtype of %s (MRO: %s)"
                % (value, types, ", ".join(str(cls) for cls in value.__class__.__mro__))
            )

        # Check that the task has been submitted
        if self.task and not value.__xpm__.job:
            raise ValueError("The value must be submitted before giving it")
        return value

    def fullyqualifiedname(self) -> str:
        """Returns the fully qualified (Python) name"""
        return f"{self.value_type.__module__}.{self.value_type.__qualname__}"


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
            raise TypeError(f"value is not a pathlike value ({type(value)})")
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


class TypeVarType(Type):
    def __init__(self, typevar: TypeVar):
        self.typevar = typevar

    def name(self):
        return str(self.typevar)

    def validate(self, value):
        return value

    def __str__(self):
        return f"TypeVar({self.typevar})"

    def __repr__(self):
        return f"TypeVar({self.typevar})"


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


class UnionType(Type):
    def __init__(self, types: List[Type]):
        self.types = types

    def name(self):
        return "Union[" + ", ".join(t.name() for t in self.types) + "]"

    def __str__(self):
        return "[" + " | ".join(t.name() for t in self.types) + " ]"

    def __repr__(self):
        return str(self)

    def validate(self, value):
        for subtype in self.types:
            try:
                return subtype.validate(value)
            except ValueError:
                pass
            except TypeError:
                pass

        if not isinstance(value, dict):
            raise ValueError(f"value is not within the types {self}")


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


class GenericType(Type):
    def __init__(self, type: typing.Type):
        self.type = type
        self.origin = get_origin(type)

        self.args = get_args(type)

    def name(self):
        return str(self.type)

    def __repr__(self):
        return repr(self.type)

    def identifier(self):
        """Returns the identifier of the type"""
        return Identifier(f"{self.origin}.{self.type}")

    def validate(self, value):
        # Now, let's check generics...
        mros = typingutils.generic_mro(type(value))
        matching = next(
            (mro for mro in mros if (get_origin(mro) or mro) is self.origin), None
        )
        target = get_origin(self.type) or self.type

        if matching is None:
            raise ValueError(
                f"{type(value)} is not of type {target} ({type(value).__mro__})"
            )

        return value
