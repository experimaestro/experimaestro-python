"""Experiment specifications generated from a :class:`~experimaestro.Config`"""

import copy
import functools
import inspect
import types
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast, get_origin

import attr

from experimaestro.core import types as xpmtypes
from experimaestro.core.arguments import Argument
from experimaestro.generators import Generator

from .grid import GenericParams, GridSearch
from .specification import MISSING

# Type of the objects returned by attr.field()
_CountingAttr = type(attr.field())

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform

T = TypeVar("T")

_SCALARS: dict[type, type] = {
    xpmtypes.IntType: int,
    xpmtypes.FloatType: float,
    xpmtypes.StrType: str,
    xpmtypes.BoolType: bool,
    xpmtypes.PathType: Path,
}


def _python_type(xpmtype: xpmtypes.Type) -> Any:
    """Python type of a YAML-representable experimaestro type (None otherwise)

    Configurations (and containers holding them) are not representable: they
    are given to :meth:`ConfigSpec.to_config` instead.
    """
    if (python_type := _SCALARS.get(type(xpmtype))) is not None:
        return python_type

    match xpmtype:
        case xpmtypes.AnyType():
            return Any
        case xpmtypes.NoneTypeType():
            return type(None)
        case xpmtypes.EnumType():
            return xpmtype.type
        case xpmtypes.ArrayType():
            item = _python_type(xpmtype.type)
            return None if item is None else list.__class_getitem__(item)
        case xpmtypes.SetType():
            item = _python_type(xpmtype.type)
            return None if item is None else set.__class_getitem__(item)
        case xpmtypes.DictType():
            key = _python_type(xpmtype.keytype)
            value = _python_type(xpmtype.valuetype)
            if key is None or value is None:
                return None
            return dict.__class_getitem__((key, value))
        case xpmtypes.TupleType():
            items = [_python_type(t) for t in xpmtype.types]
            if any(item is None for item in items):
                return None
            if xpmtype.variadic:
                return tuple.__class_getitem__((items[0], ...))
            return tuple.__class_getitem__(tuple(items))
        case xpmtypes.UnionType():
            items = [_python_type(t) for t in xpmtype.types]
            if any(item is None for item in items):
                return None
            union = items[0]
            for item in items[1:]:
                union = union | item
            return union

    return None


def _default(argument: Argument) -> Any:
    """The attrs default of the field mirroring an argument"""
    if argument.required:
        return MISSING
    if argument.generator is not None:
        return attr.Factory(argument.generator)
    if argument.default is None:
        return None
    if isinstance(argument.default, (list, dict, set)):
        value = argument.default
        return attr.Factory(lambda: copy.deepcopy(value))
    return argument.default


_NOT_GIVEN: Any = object()


@attr.define(kw_only=True)
class _Options:
    """Options of a ConfigSpec class (inherited by its subclasses)"""

    meta: bool = False
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    defaults: dict[str, Any] = attr.Factory(dict)
    grid: tuple[str, ...] = ()

    def update(self, meta, include, exclude, defaults, grid) -> "_Options":
        return _Options(
            meta=self.meta if meta is None else meta,
            include=self.include if include is None else tuple(include),
            exclude=self.exclude + tuple(exclude or ()),
            defaults={**self.defaults, **(defaults or {})},
            grid=self.grid + tuple(grid or ()),
        )


def _fields_from_config(
    config_class: Any, options: _Options, qualname: str
) -> dict[str, Any]:
    """The attrs fields mirroring the parameters of a configuration"""
    arguments = config_class.__getxpmtype__().arguments
    named = set(options.include or ()) | set(options.exclude)
    named |= set(options.defaults) | set(options.grid)
    if unknown := named - set(arguments):
        raise TypeError(
            f"{qualname}: unknown parameters {sorted(unknown)}"
            f" for {config_class.__qualname__}"
        )

    fields: dict[str, Any] = {}
    for name, argument in arguments.items():
        if (options.include is not None and name not in options.include) or (
            name in options.exclude
        ):
            continue
        # Meta parameters are explicitly ignored (some types, e.g. paths,
        # are ignored by default in identifiers)
        is_meta = argument.ignored and not argument.type.ignore
        if argument.constant or (is_meta and not options.meta):
            continue
        if isinstance(argument.generator, Generator):
            continue
        if (python_type := _python_type(argument.type)) is None:
            continue
        if not argument.required and argument.default is None:
            # The type of an Optional[T] parameter is T: accept None too
            python_type = python_type | None
        if name in options.grid:
            python_type = cast(Any, GridSearch)[python_type]

        if name in options.defaults:
            default = options.defaults[name]
            if isinstance(default, (list, dict, set)):
                default = attr.Factory(functools.partial(copy.deepcopy, default))
        else:
            default = _default(argument)

        fields[name] = attr.field(
            type=python_type, default=default, metadata={"help": argument.help}
        )

    for name in set(options.defaults) | set(options.grid):
        if name not in fields:
            raise TypeError(
                f"{qualname}: parameter {name} of {config_class.__qualname__}"
                " cannot be read from YAML"
            )
    return fields


@dataclass_transform(kw_only_default=True)
class ConfigSpec(Generic[T]):
    """Experiment specification mirroring the parameters of a configuration

    ``ConfigSpec[MyConfig]`` is a specification class (to be used as a field
    of an :class:`~experimaestro.experiments.ExperimentSpecification`) with one
    field per parameter of ``MyConfig`` whose type can be read from YAML.
    Required parameters default to ``MISSING``, the others to the
    configuration default. Parameters whose value is (or contains) a
    configuration, constants and generated values (e.g. paths) are never
    included: they are given to :meth:`to_config`.

    The fields can be changed with these options, given to :func:`spec_field`
    or as class keyword arguments of a subclass (whose subclasses inherit
    them):

    - ``meta`` (default ``False``): also include the ``Meta`` parameters;
    - ``include``: only include these parameters;
    - ``exclude``: do not include these parameters;
    - ``defaults``: default values of the parameters;
    - ``grid``: parameters accepting a grid search (``GridSearch[T]``).

    Fields declared in the class body of a subclass replace the generated ones
    and can add new fields.

    ```python
    @specification
    class Specification(ExperimentSpecification):
        model: ConfigSpec[Model]
        encoder: ConfigSpec[Encoder] = spec_field(
            defaults={"size": 128}, grid=["size"]
        )


    def run(helper, spec: Specification):
        model = spec.model.to_config(encoder=spec.encoder.to_config())
    ```
    """

    __xpm_config_class__: ClassVar[Any]
    __xpm_template__: ClassVar[bool]
    __xpm_config_fields__: ClassVar[tuple[str, ...]]
    __xpm_options__: ClassVar[_Options]

    _templates: ClassVar[dict[type, type]] = {}

    def __class_getitem__(cls, params):
        if (
            cls is ConfigSpec
            and isinstance(params, type)
            and hasattr(params, "__getxpmtype__")
        ):
            # ConfigSpec[MyConfig] is a template: the fields are generated in
            # its subclasses (attrs fields are always inherited, so options
            # like exclude could not remove them)
            if (template := ConfigSpec._templates.get(params)) is None:
                template = ConfigSpec._templates[params] = types.new_class(
                    f"{params.__name__}Spec",
                    (ConfigSpec,),
                    {"config": params, "template": True},
                )
                template.__qualname__ = f"ConfigSpec[{params.__qualname__}]"
            return template
        return super().__class_getitem__(params)

    def __new__(cls, *args, **kwargs):
        # ConfigSpec[MyConfig](...) builds the default specification class
        return super().__new__(_concrete(cls))

    def __init_subclass__(
        cls,
        *,
        config: Any = None,
        template: bool = False,
        meta: bool | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        defaults: dict[str, Any] | None = None,
        grid: list[str] | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls.__xpm_template__ = template
        if config is None:
            config = getattr(cls, "__xpm_config_class__", None)
        if config is None:
            if getattr(cls, "__parameters__", ()):
                # Generic subclass (e.g. ConfigSpec[T])
                return
            raise TypeError(
                f"{cls.__qualname__} must derive from ConfigSpec[SomeConfig]"
            )
        options = getattr(cls, "__xpm_options__", _Options()).update(
            meta, include, exclude, defaults, grid
        )
        cls.__xpm_config_class__ = config
        cls.__xpm_options__ = options
        if template:
            return

        generated = _fields_from_config(config, options, cls.__qualname__)
        arguments = config.__getxpmtype__().arguments

        # Fields of a (non template) parent class are inherited, unless this
        # class changes them
        inherited = {}
        for base in cls.__mro__[1:]:
            if attr.has(base):
                inherited = attr.fields_dict(base)
                break
        changed = set(defaults or ()) | set(grid or ())
        these = {
            name: field
            for name, field in generated.items()
            if name not in inherited or name in changed
        }
        config_fields = list(getattr(cls, "__xpm_config_fields__", ()))
        if removed := [
            name
            for name in config_fields
            if name in (exclude or ()) or (include is not None and name not in include)
        ]:
            raise TypeError(
                f"{cls.__qualname__}: cannot remove the fields {removed}"
                " inherited from its parent class"
            )
        config_fields += [name for name in generated if name not in config_fields]

        # Fields declared in the class body (overriding or new): attrs takes
        # their type from the class annotations
        for name, annotation in inspect.get_annotations(cls).items():
            if annotation is ClassVar or get_origin(annotation) is ClassVar:
                continue
            default = cls.__dict__.get(name, MISSING)
            these[name] = (
                default
                if isinstance(default, _CountingAttr)
                else attr.field(default=default)
            )
            if name in arguments and name not in config_fields:
                config_fields.append(name)

        cls.__xpm_config_fields__ = tuple(config_fields)
        attr.define(
            cls,
            these=these,
            kw_only=True,
            slots=False,
            hash=True,
            eq=True,
            auto_attribs=False,
        )

    def to_config(self, **extra) -> T:
        """Builds the configuration

        :param extra: values of the parameters not in this specification (e.g.
            configurations); they replace the specification values
        :returns: the configuration (``MyConfig.C(...)``)
        """
        values = {}
        for name in self.__xpm_config_fields__:
            value = getattr(self, name)
            if isinstance(value, GenericParams):
                if value.is_grid:
                    raise ValueError(
                        f"{type(self).__qualname__}.{name} is a grid search:"
                        " use generate_grid() first"
                    )
                value = value.value
            if value is MISSING:
                continue
            if isinstance(value, (list, dict, set)):
                value = copy.deepcopy(value)
            values[name] = value
        values.update(extra)
        return self.__xpm_config_class__.C(**values)


_concrete_classes: dict[type, type] = {}


def _concrete(cls: type) -> type:
    """The specification class to use for a (possibly template) class"""
    if not cls.__dict__.get("__xpm_template__", False):
        return cls
    if (concrete := _concrete_classes.get(cls)) is None:
        concrete = _concrete_classes[cls] = types.new_class(cls.__name__, (cls,))
        concrete.__qualname__ = cls.__qualname__
    return concrete


@attr.frozen
class _SpecField:
    """A ConfigSpec field of a specification (see :func:`spec_field`)"""

    options: dict[str, Any]
    field_kwargs: dict[str, Any]


def spec_field(
    *,
    meta: bool | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    defaults: dict[str, Any] | None = None,
    grid: list[str] | None = None,
    **kwargs,
) -> Any:
    """Field of a specification whose type is a ConfigSpec

    The options (see :class:`ConfigSpec`) change the fields of the
    specification class, and the field defaults to the specification class
    with all its default values. Other keyword arguments are given to
    ``attr.field``. This is only valid in a
    :func:`~experimaestro.experiments.specification` class.

    ```python
    model: ConfigSpec[Model] = spec_field(meta=True, defaults={"lr": 1e-4})
    ```
    """
    options = {
        "meta": meta,
        "include": include,
        "exclude": exclude,
        "defaults": defaults,
        "grid": grid,
    }
    return _SpecField(options, kwargs)


def resolve_spec_fields(cls):
    """Prepares the ConfigSpec fields of a class before attrs processes it

    A field annotated with a ConfigSpec class defaults to an instance of it;
    with :func:`spec_field`, its type is a ConfigSpec subclass with the given
    options.
    """
    try:
        annotations = inspect.get_annotations(cls, eval_str=True)
    except Exception as e:
        # Some annotation cannot be resolved (yet): fine if no field uses
        # spec_field (ConfigSpec fields then need an explicit default)
        if any(isinstance(value, _SpecField) for value in vars(cls).values()):
            raise TypeError(
                f"{cls.__qualname__}: cannot resolve the annotations"
                f" (needed by spec_field): {e}"
            ) from e
        return

    changed = False
    for name, annotation in list(annotations.items()):
        value = cls.__dict__.get(name, _NOT_GIVEN)
        if not isinstance(value, _SpecField) and value is not _NOT_GIVEN:
            continue

        spec_class = annotation
        if not (isinstance(spec_class, type) and issubclass(spec_class, ConfigSpec)):
            if isinstance(value, _SpecField):
                raise TypeError(
                    f"{cls.__qualname__}.{name}: spec_field() needs a"
                    f" ConfigSpec type, not {annotation!r}"
                )
            continue

        field_kwargs = {}
        if isinstance(value, _SpecField):
            qualname = f"{spec_class.__qualname__}({cls.__qualname__}.{name})"
            spec_class = types.new_class(
                spec_class.__name__,
                (spec_class,),
                {
                    key: option
                    for key, option in value.options.items()
                    if option is not None
                },
            )
            spec_class.__qualname__ = qualname
            field_kwargs = value.field_kwargs
        else:
            spec_class = _concrete(spec_class)
        if spec_class is not annotation:
            annotations[name] = spec_class
            changed = True

        setattr(cls, name, attr.field(factory=spec_class, **field_kwargs))

    if changed:
        cls.__annotations__ = annotations
