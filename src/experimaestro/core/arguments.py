"""Management of the arguments (params, options, etc) associated with the XPM objects"""

import warnings
from typing import Optional, TypeVar, TYPE_CHECKING, Callable, Any
from experimaestro.typingutils import get_optional
from pathlib import Path
from typing import Annotated

if TYPE_CHECKING:
    import experimaestro.core.types
    from experimaestro.core.partial import ParameterGroup

# Track deprecation warnings per module (max 10 per module)
_deprecation_warning_counts: dict[str, int] = {}
_MAX_WARNINGS_PER_MODULE = 10


class Argument:
    """Represents an argument of a configuration or task"""

    objecttype: Optional["experimaestro.core.types.ObjectType"]
    """The object for which this argument was declared"""

    def __init__(
        self,
        name,
        type: "experimaestro.core.types.Type",
        required=None,
        help=None,
        generator=None,
        ignored=None,
        field_or_default=None,
        checker=None,
        constant=False,
        is_data=False,
        overrides=False,
        groups: set["ParameterGroup"] = None,
    ):
        """Creates a new argument

        Args:
            name (str): The name of the argument

            type (experimaestro.core.types.Type): The type of the argument

            required (bool, optional): True if required (if None, determines
            automatically). Defaults to None.

            help (str, optional): Help string. Defaults to None.

            generator (Generator, optional): The value generator (e.g. for
            paths). Defaults to None.

            ignored (bool, optional): True if ignored (if None, computed
            automatically). Defaults to None.

            field_or_default (any | field, optional): Default value or field
            object containing default information. Defaults to None.

            checker (any, optional): Value checker. Defaults to None.

            constant (bool, optional): If true, the value is constant. Defaults
            to False.

            is_data (bool, optional): Flag for paths that are data path (to be
            serialized). Defaults to False.

            overrides (bool, optional): If True, this argument intentionally
            overrides a parent argument. Suppresses the warning that would
            otherwise be issued. Defaults to False.

            groups (set[ParameterGroup], optional): Set of groups this parameter
            belongs to. Used with partial to compute partial identifiers.
            Defaults to None (empty set).
        """
        required = (field_or_default is None) if required is None else required
        if field_or_default is not None and required is not None and required:
            raise Exception(
                "argument '%s' is required but default value is given" % name
            )

        self.name = name
        self._help = help
        self.checker = checker
        self.type = type
        self.constant = constant
        self.ignored = self.type.ignore if ignored is None else ignored
        self.required = required
        self.objecttype = None
        self.is_data = is_data
        self.overrides = overrides

        self.generator = generator
        self.default = None
        self.ignore_generated = False
        self.ignore_default_in_identifier = False
        self.groups = groups if groups else set()

        if field_or_default is not None:
            assert self.generator is None, (
                "generator and field_or_default are exclusive options"
            )
            if isinstance(field_or_default, field):
                self.ignore_generated = field_or_default.ignore_generated
                # Allow field to override the overrides flag
                if field_or_default.overrides:
                    self.overrides = True

                # Process groups from field
                if field_or_default.groups:
                    self.groups = field_or_default.groups

                if field_or_default.ignore_default is not None:
                    # ignore_default: value is default AND ignored in identifier
                    self.default = field_or_default.ignore_default
                    self.ignore_default_in_identifier = True
                elif field_or_default.default is not None:
                    # default: value is default but NOT ignored in identifier
                    self.default = field_or_default.default
                    self.ignore_default_in_identifier = False
                elif field_or_default.default_factory is not None:
                    self.generator = field_or_default.default_factory
            else:
                # Bare default: backwards compatible, ignore in identifier
                self.default = field_or_default
                self.ignore_default_in_identifier = True

        assert not self.constant or self.default is not None, (
            "Cannot be constant without default"
        )

    def __repr__(self):
        return "Param[{name}:{type}]".format(**self.__dict__)

    def validate(self, value):
        value = self.type.validate(value)
        if self.checker:
            if not self.checker.check(value):
                raise ValueError("Value %s is not valid", value)
        return value

    def isoutput(self):
        if self.generator:
            return self.generator.isoutput()
        return False

    @property
    def help(self):
        if self._help is None and self.objecttype is not None:
            self.objecttype.__parsedoc__()
        return self._help

    @help.setter
    def help(self, help: str):
        self._help = help


class ArgumentOptions:
    """Helper class when using type hints"""

    def __init__(self):
        self.kwargs = {}
        self.constant = False

    def create(self, name, originaltype, typehint):
        from experimaestro.core.types import Type

        optionaltype = get_optional(typehint)
        type = Type.fromType(optionaltype or typehint)

        if (
            "field_or_default" not in self.kwargs
            or self.kwargs["field_or_default"] is None
        ):
            defaultvalue = getattr(originaltype, name, None)
            self.kwargs["field_or_default"] = defaultvalue

            # Emit deprecation warning for bare default values (not wrapped in field)
            # Skip warning for Constant parameters - they are inherently constant, not defaults
            if (
                defaultvalue is not None
                and not isinstance(defaultvalue, field)
                and not self.kwargs.get("constant")
            ):
                module = originaltype.__module__
                count = _deprecation_warning_counts.get(module, 0)
                if count < _MAX_WARNINGS_PER_MODULE:
                    _deprecation_warning_counts[module] = count + 1
                    class_name = originaltype.__qualname__
                    remaining = _MAX_WARNINGS_PER_MODULE - count - 1
                    limit_msg = (
                        f" ({remaining} more warnings for this module)"
                        if remaining > 0
                        else " (no more warnings for this module)"
                    )
                    warnings.warn(
                        f"Deprecated: parameter `{name}` in {module}.{class_name} "
                        f"has an ambiguous default value. Use `field(ignore_default=...)` "
                        f"to keep current behavior (default ignored in identifier) or "
                        f"`field(default=...)` to include default in identifier. "
                        f"Run `experimaestro refactor default-values` to fix automatically."
                        f"{limit_msg}",
                        DeprecationWarning,
                        stacklevel=6,
                    )

        self.kwargs["required"] = (optionaltype is None) and (
            self.kwargs["field_or_default"] is None
        )

        return Argument(name, type, **self.kwargs)


class TypeAnnotation:
    def __call__(self, options: Optional[ArgumentOptions]):
        if options is None:
            options = ArgumentOptions()
        self.annotate(options)
        return options

    def annotate(self, options: ArgumentOptions):
        pass


class _Param(TypeAnnotation):
    """Base annotation for types"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def annotate(self, options: ArgumentOptions):
        options.kwargs.update(self.kwargs)
        return options


T = TypeVar("T")

paramHint = _Param()
Param = Annotated[T, paramHint]
"""Type annotation for configuration parameters.

Parameters annotated with ``Param[T]`` are included in the configuration
identifier computation and must be set before the configuration is sealed.

Example::

    class MyConfig(Config):
        name: Param[str]
        count: Param[int] = field(default=10)
        threshold: Param[float] = field(ignore_default=0.5)
"""

optionHint = _Param(ignored=True)
Option = Annotated[T, optionHint]
"""Deprecated alias for Meta. Use Meta instead."""

Meta = Annotated[T, optionHint]
"""Type annotation for meta-parameters (ignored in identifier computation).

Use ``Meta[T]`` for parameters that should not affect the task identity,
such as output paths or runtime configuration.

Example::

    class MyTask(Task):
        # This affects the task identity
        learning_rate: Param[float]

        # This does not affect the identity
        checkpoint_path: Meta[Path] = field(default_factory=PathGenerator("model.pt"))
"""

dataHint = _Param(ignored=True, is_data=True)
DataPath = Annotated[Path, dataHint]
"""Type annotation for data paths that should be serialized.

Use ``DataPath`` for paths that point to data files that should be
preserved when serializing/deserializing a configuration. The path
is copied during serialization.

Example::

    class MyConfig(Config):
        model_weights: DataPath
"""


class field:
    """Specify additional properties for a configuration parameter.

    Use ``field()`` to control default value behavior and parameter grouping.

    Example::

        class MyConfig(Config):
            # Default included in identifier
            count: Param[int] = field(default=10)

            # Default ignored in identifier (backwards compatible)
            threshold: Param[float] = field(ignore_default=0.5)

            # Generated path
            output: Meta[Path] = field(default_factory=PathGenerator("out.txt"))

            # Parameter in a group (for partial identifiers)
            lr: Param[float] = field(groups=[training_group])
    """

    def __init__(
        self,
        *,
        default: Any = None,
        default_factory: Callable = None,
        ignore_default: Any = None,
        ignore_generated=False,
        overrides=False,
        groups: list["ParameterGroup"] = None,
    ):
        """Create a field specification.

        :param default: Default value (included in identifier computation)
        :param default_factory: Callable that generates the default value
        :param ignore_default: Default value that is ignored in identifier computation
            when the actual value equals this default. Use for backwards-compatible
            behavior with bare default values.
        :param ignore_generated: If True, the generated value is hidden from tasks.
            Useful for adding a field that changes the identifier but won't be used.
        :param overrides: If True, suppress warning when overriding parent parameter
        :param groups: List of ParameterGroup objects for partial identifiers.
            Used with partial to compute identifiers that exclude certain groups.
        """
        assert not ((default is not None) and (default_factory is not None)), (
            "default and default_factory are mutually exclusive options"
        )

        assert not ((default is not None) and (ignore_default is not None)), (
            "default and ignore_default are mutually exclusive options"
        )

        assert not ((ignore_default is not None) and (default_factory is not None)), (
            "ignore_default and default_factory are mutually exclusive options"
        )

        self.default_factory = default_factory
        self.default = default
        self.ignore_default = ignore_default
        self.ignore_generated = ignore_generated
        self.overrides = overrides
        self.groups = set(groups) if groups else set()


class help(TypeAnnotation):
    def __init__(self, text: str):
        self.text = text

    def annotate(self, options: ArgumentOptions):
        options.kwargs["help"] = self.text


class ConstantHint(TypeAnnotation):
    def annotate(self, options: ArgumentOptions):
        options.kwargs["constant"] = True


constantHint = ConstantHint()
Constant = Annotated[T, constantHint]
"""Type annotation for constant (read-only) parameters.

Constants must have a default value and cannot be modified after creation.
They are included in the identifier computation.

Example::

    class MyConfig(Config):
        version: Constant[str] = "1.0"
"""
