import logging
from enum import Enum
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Type,
    TypeVar,
    Annotated,
    get_origin,
    get_args,
    Set,
    Union,
)
import attr
from pydantic import create_model, ConfigDict, BeforeValidator
from experimaestro.typingutils import (
    get_union,
    get_list_component,
    get_dict,
    get_set,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _suggest_enum(enum_cls: Type[Enum], value: Any) -> Any:
    """Give a helpful error for near-miss enum strings (e.g. ``WARN`` -> ``warn``).

    Valid inputs are passed through to Pydantic's enum validation untouched.
    A string that does not match a value but matches one case-insensitively
    (either the value or the member name) raises a message suggesting the
    canonical (lowercase) value rather than the generic enum error.
    """
    if isinstance(value, str):
        valid_values = {m.value for m in enum_cls}
        if value not in valid_values:
            lowered = value.lower()
            for m in enum_cls:
                canonical = m.value
                if (
                    isinstance(canonical, str)
                    and canonical.lower() == lowered
                    or m.name.lower() == lowered
                ):
                    raise ValueError(
                        f"{value!r} is not valid; use the lowercase value {canonical!r}"
                    )
    return value


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            if key in base:
                logger.debug(f"Overwriting key '{key}': {base[key]} -> {value}")
            base[key] = value
    return base


def from_dotlist(dotlist: List[str]) -> Dict[str, Any]:
    """Convert a dotlist (key=value) to a nested dictionary"""
    result = {}
    for item in dotlist:
        if "=" not in item:
            logger.warning(f"Invalid dotlist item (missing '='): {item}")
            continue
        key, value = item.split("=", 1)
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def validate_attrs(cls: Type[T], data: Any) -> T:
    """Validate data against an attrs class using Pydantic"""

    if isinstance(data, cls):
        return data

    if not isinstance(data, dict):
        # Let Pydantic handle it if it's not a dict (might be invalid)
        from pydantic import TypeAdapter

        return TypeAdapter(
            cls, config={"arbitrary_types_allowed": True}
        ).validate_python(data)

    if not attr.has(cls):
        # Fallback to standard Pydantic validation if not an attrs class
        from pydantic import TypeAdapter

        return TypeAdapter(
            cls, config={"arbitrary_types_allowed": True}
        ).validate_python(data)

    def wrap_type(t: Any) -> Any:
        if t is Any:
            return Any

        # Handle Annotated
        if get_origin(t) is Annotated:
            args = get_args(t)
            return Annotated[(wrap_type(args[0]),) + args[1:]]

        # Handle Optional/Union
        if union_args := get_union(t):
            return Union[tuple(wrap_type(arg) for arg in union_args)]

        # Handle List
        if list_arg := get_list_component(t):
            return List[wrap_type(list_arg)]

        # Handle Dict
        if dict_args := get_dict(t):
            return Dict[wrap_type(dict_args[0]), wrap_type(dict_args[1])]

        # Handle Set
        if set_arg := get_set(t):
            return Set[wrap_type(set_arg)]

        if attr.has(t):
            return Annotated[t, BeforeValidator(partial(validate_attrs, t))]

        return t

    fields = attr.fields(cls)
    model_fields = {}
    for f in fields:
        # Get the type, handle missing types as Any
        f_type = f.type if f.type is not None else Any

        # Recursively handle nested attrs classes and complex types
        f_type = wrap_type(f_type)

        # Suggest the canonical (lowercase) value for near-miss enum strings
        elif isinstance(f_type, type) and issubclass(f_type, Enum):
            f_type = Annotated[f_type, BeforeValidator(partial(_suggest_enum, f_type))]

        # Handle default value
        if f.default is not attr.NOTHING:
            if isinstance(f.default, attr.Factory):
                model_fields[f.name] = (f_type, None)
            else:
                model_fields[f.name] = (f_type, f.default)
        else:
            model_fields[f.name] = (f_type, ...)

    # Create a dynamic Pydantic model
    PydanticModel = create_model(
        cls.__name__,
        __config__=ConfigDict(extra="allow", arbitrary_types_allowed=True),
        **model_fields,
    )

    # Validate
    validated_model = PydanticModel(**data)

    # Extract only the fields that the attrs class expects
    attrs_data = {}
    for f in fields:
        if hasattr(validated_model, f.name):
            val = getattr(validated_model, f.name)

            # If the value is None and it wasn't explicitly provided in the data,
            # and there's a factory, let attrs handle it at instantiation
            if (
                val is None
                and f.name not in data
                and isinstance(f.default, attr.Factory)
            ):
                continue

            attrs_data[f.name] = val

    return cls(**attrs_data)
