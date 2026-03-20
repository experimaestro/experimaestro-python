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


class ConfigWrapper:
    """Wraps a config value with properties that modify how the parent processes it.

    Properties:
        tagged: If True, the value is tagged (appears in experiment tags)
        stop_tags: If True, tags from this sub-config don't propagate to parent

    Wrappers can be nested — properties are merged automatically::

        stop_tags(tag(value))  # both tagged=True and stop_tags=True
    """

    def __init__(self, value, *, tagged: bool = False, stop_tags: bool = False):
        self.value = value
        self.tagged = tagged
        self.stop_tags = stop_tags

    @staticmethod
    def ensure(value) -> "ConfigWrapper":
        """Return value as a ConfigWrapper, reusing it if already one."""
        if isinstance(value, ConfigWrapper):
            return value
        return ConfigWrapper(value)

    def apply(self, config_info: "ConfigInformation", arg_name: str, *, source: str):  # noqa: F821
        """Apply all wrapper properties to the parent ConfigInformation"""
        if self.tagged:
            config_info.addtag(arg_name, self.value, source=source)
        if self.stop_tags:
            config_info._args_properties.setdefault(arg_name, set()).add("stop_tags")


# Backwards-compatible alias
TaggedValue = ConfigWrapper


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
