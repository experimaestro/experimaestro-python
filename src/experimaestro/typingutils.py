import sys
import types
import typing
from typing import Generic, Protocol

if sys.version_info.major == 3:
    if sys.version_info.minor < 11:
        from typing import _collect_type_vars as _collect_parameters
    else:

        def _collect_parameters(bases):
            """Collect type parameters from generic bases"""
            parameters = []
            for base in bases:
                if hasattr(base, "__parameters__"):
                    parameters.extend(base.__parameters__)
            return tuple(parameters)

        # from typing import _collect_parameters

    from typing import _AnnotatedAlias as AnnotatedAlias, get_args, get_origin

GenericAlias = typing._GenericAlias


def isgenericalias(typehint):
    """Returns True if it is a generic type alias"""
    # Works with Python 3.7, 3.8 and 3.9
    return isinstance(typehint, GenericAlias)


def _is_union(typehint):
    """Returns True if typehint is a Union type.

    Handles both typing.Union and types.UnionType (PEP 604, X | Y syntax).
    Compatible with Python 3.10+ including 3.14 where Union representation changed.
    """
    return get_origin(typehint) is typing.Union or isinstance(typehint, types.UnionType)


def get_union(typehint):
    """Return the list of types of a union (or the type itself if it is not an union)"""
    if _is_union(typehint):
        return get_args(typehint)
    return None


def get_optional(typehint):
    if _is_union(typehint):
        args = get_args(typehint)
        if len(args) == 2:
            for ix in (0, 1):
                argtype = args[ix]
                origin = get_origin(argtype) or argtype
                if issubclass(origin, type(None)):
                    return args[1 - ix]
    return None


def get_list_component(typehint):
    """Returns the list component (or None if not a list)"""
    if isgenericalias(typehint):
        # Python 3.6, 3.7+ test
        if typehint.__origin__ in [typing.List, list]:
            assert len(typehint.__args__) == 1
            return typehint.__args__[0]
    return None


get_list = get_list_component
"""
    :deprecated: use get_list_component
"""


def is_annotated(typehint):
    return isinstance(typehint, AnnotatedAlias)


def get_type(typehint):
    """Returns the type discarding Annotated and optional"""
    while True:
        if t := get_optional(typehint):
            typehint = t
        if is_annotated(typehint):
            typehint = get_args(typehint)[0]
        else:
            break
    return typehint


def get_dict(typehint):
    if isgenericalias(typehint):
        # Python 3.6, 3.7+ test
        if typehint.__origin__ in [typing.Dict, dict]:
            assert len(typehint.__args__) == 2
            return typehint.__args__
    return None


# From https://github.com/python/typing/issues/777


def _generic_mro(result, tp):
    origin = typing.get_origin(tp) or tp

    result[origin] = tp
    if hasattr(origin, "__orig_bases__"):
        parameters = _collect_parameters(origin.__orig_bases__)
        substitution = dict(zip(parameters, get_args(tp)))
        for base in origin.__orig_bases__:
            if get_origin(base) in result:
                continue
            base_parameters = getattr(base, "__parameters__", ())
            if base_parameters:
                base = base[tuple(substitution.get(p, p) for p in base_parameters)]
            _generic_mro(result, base)


def generic_mro(tp):
    origin = get_origin(tp)
    if origin is None and not hasattr(tp, "__orig_bases__"):
        if not isinstance(tp, type):
            raise TypeError(f"{tp!r} is not a type or a generic alias")
        return tp.__mro__
    # sentinel value to avoid to subscript Generic and Protocol
    result = {Generic: Generic, Protocol: Protocol}
    _generic_mro(result, tp)
    cls = origin if origin is not None else tp
    return tuple(result.get(sub_cls, sub_cls) for sub_cls in cls.__mro__)


def is_generic_subtype(child_type, parent_type, *, config_base: type | None = None):
    """Check if child_type is a valid subtype of parent_type for generic types.

    For generic types like Generic[T], checks that:
    1. The origin types match (e.g., both are List, Dict, etc.)
    2. Type arguments are compatible (subtypes for config_base types)

    Args:
        child_type: The child generic type to check
        parent_type: The parent generic type to check against
        config_base: Base class for Config types. If provided, type arguments
            that are subclasses of this will be checked for subtype compatibility.
            If None, type arguments must match exactly.

    Returns:
        True if child_type is a valid subtype of parent_type

    Raises:
        TypeError: If the types are incompatible, with a descriptive message
    """
    child_origin = get_origin(child_type)
    parent_origin = get_origin(parent_type)

    # Check that origin types match
    if child_origin is not parent_origin:
        raise TypeError(
            f"Generic origin {child_origin} is not compatible with "
            f"parent origin {parent_origin}. "
            f"Override types must have the same generic origin."
        )

    # Check type arguments compatibility
    child_args = get_args(child_type)
    parent_args = get_args(parent_type)

    if len(child_args) != len(parent_args):
        raise TypeError(
            f"Generic type has {len(child_args)} type arguments "
            f"but parent has {len(parent_args)}."
        )

    for child_arg, parent_arg in zip(child_args, parent_args):
        # Check if both are types that should be checked for subtype compatibility
        if (
            config_base is not None
            and isinstance(child_arg, type)
            and isinstance(parent_arg, type)
            and issubclass(parent_arg, config_base)
        ):
            # For Config types, child must be a subtype of parent
            if not issubclass(child_arg, parent_arg):
                raise TypeError(
                    f"Type argument {child_arg.__qualname__} "
                    f"is not a subtype of parent type argument {parent_arg.__qualname__}. "
                    f"Override generic type arguments must be subtypes."
                )
        elif child_arg != parent_arg:
            # For non-Config types, require exact match
            raise TypeError(
                f"Type argument {child_arg} does not match "
                f"parent type argument {parent_arg}."
            )

    return True
