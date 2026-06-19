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
from pydantic_core import core_schema, SchemaValidator
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
        self.name = name.lower()

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

    def __core_schema__(self):
        """Return a pydantic-core schema for validating a value of this type.

        Validation is performed entirely by pydantic-core: each concrete Type
        builds a schema describing how to validate one value, composing its
        children's schemas for nested generics. The leaf types whose coercion
        is domain-specific (int/float/bool/path, Config, user generics) use
        plain/before validator functions so behavior -- especially coercion,
        which feeds the identifier hash -- is preserved exactly.
        """
        raise NotImplementedError(f"__core_schema__ ({self.__class__})")

    @property
    def _schema_validator(self) -> SchemaValidator:
        """Cached pydantic-core validator built from ``__core_schema__()``.

        Built once per Type node (Type instances are long-lived / shared via
        ``DEFINED``), not once per value.
        """
        sv = self.__dict__.get("_sv_cache")
        if sv is None:
            sv = SchemaValidator(self.__core_schema__())
            self.__dict__["_sv_cache"] = sv
        return sv

    def _validate_with_schema(self, value):
        """Validate through the cached pydantic-core schema.

        Raises ``pydantic_core.ValidationError`` (a ``ValueError`` subclass) on
        failure, matching the ValueError contract callers already expect.
        """
        return self._schema_validator.validate_python(value)

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

    def validate(self, value):
        """Validate (and coerce) a value through this type's pydantic schema.

        Raises ``pydantic_core.ValidationError`` (a ``ValueError`` subclass) on
        failure.
        """
        return self._validate_with_schema(value)

    @staticmethod
    def fromType(key):
        """Returns the type object corresponding to the given type"""
        logger.debug("Searching for type %s", key)
        from .objects import Config

        if key is None:
            return Any

        # NoneType, as it appears in Optional[X] == X | None
        if key is type(None):
            return NoneTypeNone

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

        t = typingutils.get_set(key)
        if t:
            return SetType(Type.fromType(t))

        t = typingutils.get_dict(key)
        if t:
            return DictType(Type.fromType(t[0]), Type.fromType(t[1]))

        t = typingutils.get_tuple(key)
        if t is not None:
            subtypes, variadic = t
            return TupleType([Type.fromType(st) for st in subtypes], variadic)

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
            # Use __annotations__ property (not __dict__) for Python 3.14+
            # compatibility (PEP 649: deferred annotation evaluation)
            typekeys = set(self._original_type.__annotations__.keys())
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
            try:
                argument.type.validate(argument.default)
            except TypeError as e:
                raise TypeError(
                    f"Value {argument.default} is not valid for argument {argument.name}: {e}"
                )

    def _check_override_type_compatibility(
        self, child_arg: Argument, parent_arg: Argument
    ):
        """Check that the child argument type is compatible with the parent type.

        For Config types, the child type should be a subtype of the parent type
        (covariant). For other types, we check for exact match.
        """
        from .objects import Config

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
        elif isinstance(child_type, GenericType) and isinstance(
            parent_type, GenericType
        ):
            # Check generic type compatibility using utility function
            try:
                typingutils.is_generic_subtype(
                    child_type.type, parent_type.type, config_base=Config
                )
            except TypeError as e:
                raise TypeError(f"Parameter '{child_arg.name}': {e}") from None
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

    def __core_schema__(self):
        # Config values carry domain rules pydantic cannot express (sealed,
        # submitted task, subtype against value_type), so this is a plain
        # validator. Keeping Config as a leaf also avoids cyclic schemas and
        # any interaction with pydantic's metaclass.
        return core_schema.no_info_plain_validator_function(self._check)

    def _check(self, value):
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


def _path_value(value):
    # Accept str / Path and normalise to Path. The serialized {"type": "path"}
    # form is resolved to a real Path by the deserializer
    # (ConfigInformation._objectFromParameters) before it ever reaches here.
    if not isinstance(value, (str, Path)):
        raise ValueError(f"value is not a pathlike value ({type(value)})")
    return Path(value)


@definetype(int)
class IntType(Type):
    def __core_schema__(self):
        # Native lax int: parses numeric strings, normalizes bool -> int,
        # coerces integral floats, rejects non-integral floats.
        return core_schema.int_schema()


@definetype(str)
class StrType(Type):
    def __core_schema__(self):
        # strict: only str instances are accepted (no int/bytes coercion)
        return core_schema.str_schema(strict=True)


@definetype(float)
class FloatType(Type):
    def __core_schema__(self):
        # Native lax float: parses numeric strings, coerces int/bool.
        return core_schema.float_schema()


@definetype(bool)
class BoolType(Type):
    def __core_schema__(self):
        # Native lax bool: "true"/"false"/0/1 etc.; rejects arbitrary objects.
        return core_schema.bool_schema()


@definetype(Path)
class PathType(Type):
    def __core_schema__(self):
        return core_schema.no_info_plain_validator_function(_path_value)

    @property
    def ignore(self):
        """Ignore by default"""
        return True


class AnyType(Type):
    def __init__(self):
        super().__init__("any")

    def __core_schema__(self):
        return core_schema.any_schema()


class NoneTypeType(Type):
    """The ``None`` type, used as a member of ``Optional[X]`` (i.e. ``X | None``)."""

    def __init__(self):
        super().__init__("none")

    def name(self):
        return "None"

    def __core_schema__(self):
        return core_schema.none_schema()


class TypeVarType(Type):
    def __init__(self, typevar: TypeVar):
        self.typevar = typevar

    def name(self):
        return str(self.typevar)

    def __core_schema__(self):
        return core_schema.any_schema()

    def __str__(self):
        return f"TypeVar({self.typevar})"

    def __repr__(self):
        return f"TypeVar({self.typevar})"


Any = AnyType()
NoneTypeNone = NoneTypeType()


class ArrayType(Type):
    def __init__(self, type: Type):
        self.type = type

    def name(self):
        return f"List[{self.type.name()}]"

    def __core_schema__(self):
        # strict matches legacy (only list instances accepted); element
        # validation is delegated to the child schema (native recursion).
        return core_schema.list_schema(self.type.__core_schema__(), strict=True)

    def __str__(self):
        return f"Array({self.type})"

    def __repr__(self):
        return f"Array({self.type})"


def _set_sealed_check(value):
    """Pinned legacy rule: Config elements of a set must be sealed."""
    from .objects import Config

    if isinstance(value, set):
        for x in value:
            if isinstance(x, Config) and not x.__xpm__._sealed:
                raise ValueError(
                    f"Config {x.__class__.__qualname__} in set is not sealed. "
                    f"Use sealed_set() to create sets of Config objects."
                )
    return value


class SetType(Type):
    def __init__(self, type: Type):
        self.type = type

    def name(self):
        return f"Set[{self.type.name()}]"

    def __core_schema__(self):
        # The sealed-config check runs first (before-validator), then native
        # set validation with strict=True (only set instances accepted).
        return core_schema.no_info_before_validator_function(
            _set_sealed_check,
            core_schema.set_schema(self.type.__core_schema__(), strict=True),
        )

    def __str__(self):
        return f"Set({self.type})"

    def __repr__(self):
        return f"Set({self.type})"


class EnumType(Type):
    def __init__(self, type: typing.Type[Enum]):
        self.type = type

    def name(self):
        return self.type.__name__

    def __core_schema__(self):
        return core_schema.is_instance_schema(self.type)

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

    def __core_schema__(self):
        # left_to_right preserves the legacy "first matching member wins"
        # semantics (it iterated self.types in order), which keeps coercion
        # deterministic -- important since coerced values feed the identifier.
        # Native union resolution fixes the legacy bug where a non-matching
        # dict silently returned None and where member errors were discarded.
        # left_to_right preserves the legacy "first matching member wins"
        # semantics, which keeps coercion deterministic (it feeds the identifier).
        return core_schema.union_schema(
            [t.__core_schema__() for t in self.types],
            mode="left_to_right",
        )


class TupleType(Type):
    """A tuple type, either fixed ``tuple[X, Y]`` or variadic ``tuple[X, ...]``.

    Previously such annotations fell through to ``GenericType``, which only
    checked the instance class' declared type arguments and never validated
    element values or arity.
    """

    def __init__(self, types: List["Type"], variadic: bool):
        self.types = types
        self.variadic = variadic

    def name(self):
        if self.variadic:
            return f"Tuple[{self.types[0].name()}, ...]"
        return "Tuple[" + ", ".join(t.name() for t in self.types) + "]"

    def __str__(self):
        inner = ", ".join(str(t) for t in self.types)
        return f"Tuple({inner}{', ...' if self.variadic else ''})"

    def __repr__(self):
        return str(self)

    def __core_schema__(self):
        if self.variadic:
            return core_schema.tuple_schema(
                [self.types[0].__core_schema__()], variadic_item_index=0
            )
        return core_schema.tuple_schema([t.__core_schema__() for t in self.types])


class DictType(Type):
    def __init__(self, keytype: Type, valuetype: Type):
        self.keytype = keytype
        self.valuetype = valuetype

    def name(self):
        return f"Dict[{self.keytype.name()},{self.valuetype.name()}]"

    def __core_schema__(self):
        # strict matches legacy (only dict instances accepted)
        return core_schema.dict_schema(
            self.keytype.__core_schema__(),
            self.valuetype.__core_schema__(),
            strict=True,
        )

    def __str__(self):
        return f"Dict({self.keytype.name()},{self.valuetype.name()})"

    def __repr__(self):
        return str(self)


def _is_type_arg_compatible(actual, expected) -> bool:
    """Check if an actual type argument is compatible with the expected one.

    Handles Union types: actual is compatible with Union[X, Y, Z] if actual
    is a subtype of any member of the union.
    """
    # typing.Any acts as a wildcard on either side (matches mypy/pyright)
    if expected is typing.Any or actual is typing.Any:
        return True

    # Both concrete types: use subclass check
    if isinstance(expected, type) and isinstance(actual, type):
        return issubclass(actual, expected)

    # Expected is a Union: actual must be compatible with at least one member
    if get_origin(expected) is typing.Union:
        return any(
            _is_type_arg_compatible(actual, member) for member in get_args(expected)
        )

    # Actual is a Union: each member of actual must be compatible with expected
    if get_origin(actual) is typing.Union:
        return all(
            _is_type_arg_compatible(member, expected) for member in get_args(actual)
        )

    # For generic types (e.g., List[str]), check origin and args recursively
    expected_origin = get_origin(expected)
    actual_origin = get_origin(actual)
    if expected_origin is not None and actual_origin is not None:
        if not issubclass(actual_origin, expected_origin):
            return False
        expected_args = get_args(expected)
        actual_args = get_args(actual)
        if expected_args and actual_args:
            return all(
                _is_type_arg_compatible(a, e)
                for a, e in zip(actual_args, expected_args)
            )
        return True

    # Fallback: exact equality
    return expected == actual


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

    def __core_schema__(self):
        # User generics (Generic[T] / value classes) are validated by MRO logic
        # pydantic cannot express, so this is a plain validator function.
        return core_schema.no_info_plain_validator_function(self._check)

    def _check(self, value):
        # Now, let's check generics...
        mros = typingutils.generic_mro(type(value))
        matching = next(
            (mro for mro in mros if (get_origin(mro) or mro) is self.origin), None
        )
        target = get_origin(self.type) or self.type

        if matching is None:
            raise ValueError(
                f"{type(value)} is not of type {target} (MRO:{type(value).__mro__})"
            )

        # Check type arguments compatibility
        matching_args = get_args(matching)
        if self.args and matching_args:
            for expected, actual in zip(self.args, matching_args):
                if isinstance(expected, TypeVar) or isinstance(actual, TypeVar):
                    continue
                if not _is_type_arg_compatible(actual, expected):
                    raise ValueError(
                        f"{type(value).__qualname__} has type argument "
                        f"{actual} which is not compatible with "
                        f"expected {expected}"
                    )

        return value
