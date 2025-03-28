"""Management of the arguments (params, options, etc) associated with the XPM objects"""

from typing import Optional, TypeVar, TYPE_CHECKING, Callable, Any
from experimaestro.typingutils import get_optional
from pathlib import Path
import sys

if TYPE_CHECKING:
    from typing_extensions import Annotated
    import experimaestro.core.types
else:
    if sys.version_info.major == 3 and sys.version_info.minor < 9:
        from typing_extensions import Annotated
    else:
        from typing import Annotated


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
        default=None,
        checker=None,
        constant=False,
        is_data=False,
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

            default (any, optional): . Defaults to None.

            checker (any, optional): Value checker. Defaults to None.

            constant (bool, optional): If true, the value is constant. Defaults
            to False.

            is_data (bool, optional): Flag for paths that are data path (to be
            serialized). Defaults to False.
        """
        required = (default is None) if required is None else required
        if default is not None and required is not None and required:
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

        self.generator = generator
        self.default = None

        if default is not None:
            assert self.generator is None, "generator and default are exclusive options"
            if isinstance(default, field):
                if default.default is not None:
                    self.default = default.default
                elif default.default_factory is not None:
                    self.generator = default.default_factory
            else:
                self.default = default

        assert (
            not self.constant or self.default is not None
        ), "Cannot be constant without default"

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

        if "default" not in self.kwargs or self.kwargs["default"] is None:
            defaultvalue = getattr(originaltype, name, None)
            self.kwargs["default"] = defaultvalue

        self.kwargs["required"] = (optionaltype is None) and (
            self.kwargs["default"] is None
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

optionHint = _Param(ignored=True)
Option = Annotated[T, optionHint]
Meta = Annotated[T, optionHint]

dataHint = _Param(ignored=True, is_data=True)
DataPath = Annotated[Path, dataHint]
"""Annotates a path that should be kept to restore an object to its state"""


class field:
    """Extra information for a given experimaestro field (param or meta)"""

    def __init__(self, *, default: Any = None, default_factory: Callable = None):
        assert not (
            (default is not None) and (default_factory is not None)
        ), "default and default_factory are mutually exclusive options"

        self.default_factory = default_factory
        self.default = default


class help(TypeAnnotation):
    def __init__(self, text: str):
        self.text = text

    def annotate(self, options: ArgumentOptions):
        options.kwargs["help"] = self.text


class default(TypeAnnotation):
    """Adds a default value (useful when we have problems with setattr and class
    properties)"""

    def __init__(self, value):
        self.value = value

    def annotate(self, options: ArgumentOptions):
        options.kwargs["default"] = self.value


class ConstantHint(TypeAnnotation):
    def annotate(self, options: ArgumentOptions):
        options.kwargs["constant"] = True


constantHint = ConstantHint()
Constant = Annotated[T, constantHint]
