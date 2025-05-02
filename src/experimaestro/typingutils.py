import sys
import typing
from typing import Generic, Protocol

if sys.version_info.major == 3:
    if sys.version_info.minor < 11:
        from typing import _collect_type_vars as _collect_parameters
    else:
        from typing import _collect_parameters

    from typing import _AnnotatedAlias as AnnotatedAlias, get_args, get_origin

GenericAlias = typing._GenericAlias


def isgenericalias(typehint):
    """Returns True if it is a generic type alias"""
    # Works with Python 3.7, 3.8 and 3.9
    return isinstance(typehint, GenericAlias)


def get_union(typehint):
    """Return the list of types of a union (or the type itself if it is not an union)"""
    if isgenericalias(typehint) and typehint.__origin__ == typing.Union:
        return typehint.__args__
    return None


def get_optional(typehint):
    if isgenericalias(typehint) and typehint.__origin__ == typing.Union:
        if len(typehint.__args__) == 2:
            for ix in (0, 1):
                argtype = typehint.__args__[ix]
                origin = get_origin(argtype) or argtype
                if issubclass(origin, type(None)):
                    return typehint.__args__[1 - ix]
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
        if isinstance(typehint, AnnotatedAlias):
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
