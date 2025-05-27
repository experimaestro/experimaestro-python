from contextlib import contextmanager
from typing import Any, Dict, Set


def getqualattr(module, qualname):
    """Get a qualified attributed value"""
    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls


@contextmanager
def add_to_path(p):
    """Temporarily add a path to sys.path"""
    import sys

    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


class ObjectStore:
    def __init__(self):
        self.store: Dict[int, Any] = {}
        self.constructed: Set[int] = set()

    def set_constructed(self, identifier: int):
        self.constructed.add(identifier)

    def is_constructed(self, identifier: int):
        return identifier in self.constructed

    def retrieve(self, identifier: int):
        return self.store.get(identifier, None)

    def add_stub(self, identifier: int, stub: Any):
        self.store[identifier] = stub


class SealedError(Exception):
    """Exception when trying to modify a sealed configuration"""

    pass


class TaggedValue:
    def __init__(self, value):
        self.value = value


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
