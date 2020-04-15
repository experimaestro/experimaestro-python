import typing
TYPING_ALIAS = type(typing.List)

def get_optional(typehint): 
    if type(typehint) == TYPING_ALIAS and typehint.__origin__ == typing.Union:
        if len(typehint.__args__) == 2:
            for ix in (0,1):
                if typehint.__args__[ix] == type(None):
                    return typehint.__args__[1-ix] 
    return None

def get_list(typehint):
    if type(typehint) == TYPING_ALIAS:
        # Python 3.6, 3.7+ test
        if typehint.__origin__ in [typing.List, list]:
            assert len(typehint.__args__) == 1
            return typehint.__args__[0]
    return None