import typing


def isgenericalias(typehint):
    """Returns True if it is a generic type alias"""
    # Works with Python 3.7, 3.8 and 3.9
    return isinstance(typehint, typing._GenericAlias)


def get_optional(typehint):
    if isgenericalias(typehint) and typehint.__origin__ == typing.Union:
        if len(typehint.__args__) == 2:
            for ix in (0, 1):
                if typehint.__args__[ix] == type(None):
                    return typehint.__args__[1 - ix]
    return None


def get_list(typehint):
    if isgenericalias(typehint):
        # Python 3.6, 3.7+ test
        if typehint.__origin__ in [typing.List, list]:
            assert len(typehint.__args__) == 1
            return typehint.__args__[0]
    return None


def get_dict(typehint):
    if isgenericalias(typehint):
        # Python 3.6, 3.7+ test
        if typehint.__origin__ in [typing.Dict, dict]:
            assert len(typehint.__args__) == 2
            return typehint.__args__
    return None
