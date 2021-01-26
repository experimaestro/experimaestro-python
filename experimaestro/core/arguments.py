"""Management of the arguments (params, options, etc) associated with the XPM objects"""

from pathlib import Path
from typing import Any, Optional, TypeVar
from experimaestro.generators import PathGenerator
from experimaestro.typingutils import get_optional
from typing_extensions import Annotated


class Argument:
    def __init__(
        self,
        name,
        type: "experimaestro.core.types.Type",
        required=None,
        help=None,
        generator=None,
        ignored=False,
        default=None,
        checker=None,
        subparam=False,
    ):
        required = (default is None) if required is None else required
        if default is not None and required is not None and required:
            raise Exception(
                "argument '%s' is required but default value is given" % name
            )

        self.name = name
        self.help = help
        self.checker = checker
        self.type = type
        self.ignored = self.type.ignore if ignored is None else ignored
        self.required = required
        self.default = default
        self.generator = generator
        self.subparam = subparam

    def __repr__(self):
        return "Param[{name}:{type}]".format(**self.__dict__)


class ArgumentOptions:
    """Helper class when using type hints"""

    kwargs = {}

    def create(self, name, originaltype, typehint):
        from experimaestro.core.types import Type

        optionaltype = get_optional(typehint)
        type = Type.fromType(optionaltype or typehint)

        defaultvalue = getattr(originaltype, name, None)
        self.kwargs["default"] = defaultvalue

        self.kwargs["required"] = (optionaltype is None) and (defaultvalue is None)

        return Argument(name, type, **self.kwargs)


class TypeAnnotation:
    def __call__(self, options: Optional[ArgumentOptions], defaultvalue: Any):
        if options is None:
            options = ArgumentOptions()
        self.annotate(options, defaultvalue)
        return options

    def annotate(self, options: ArgumentOptions, defaultvalue: Any):
        pass


class _Param(TypeAnnotation):
    """Base annotation for types"""

    def __init__(self, ignored=None):
        self.ignored = ignored

    def annotate(self, options: ArgumentOptions, defaultvalue):
        options.kwargs["ignored"] = self.ignored
        return options


paramHint = _Param()
optionHint = _Param()

T = TypeVar("T")
Param = Annotated[T, paramHint]
Option = Annotated[T, optionHint]


class help(TypeAnnotation):
    def __init__(self, text: str):
        self.text = text

    def annotate(self, options: ArgumentOptions, defaultvalue: Any):
        options.kwargs["help"] = self.text


class _PathOption(TypeAnnotation):
    def annotate(self, options: ArgumentOptions, defaultvalue: Any):
        options.kwargs["generator"] = PathGenerator(defaultvalue)


PathOption = Annotated[Path, _PathOption()]
