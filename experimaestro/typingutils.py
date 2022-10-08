import sys
import typing

if sys.version_info.major == 3 and sys.version_info.minor < 9:
    from typing_extensions import _AnnotatedAlias as AnnotatedAlias, get_args
else:
    from typing import _AnnotatedAlias as AnnotatedAlias, get_args

GenericAlias = typing._GenericAlias


def isgenericalias(typehint):
    """Returns True if it is a generic type alias"""
    # Works with Python 3.7, 3.8 and 3.9
    return isinstance(typehint, GenericAlias)


def get_optional(typehint):
    if isgenericalias(typehint) and typehint.__origin__ == typing.Union:
        if len(typehint.__args__) == 2:
            for ix in (0, 1):
                if typehint.__args__[ix] == type(None):
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
