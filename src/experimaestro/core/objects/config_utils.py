from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


def getqualattr(module, qualname):
    """Get a qualified attributed value"""
    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls


def has_qualattr(module, qualname) -> bool:
    """Returns whether a qualified attribute exists"""
    try:
        getqualattr(module, qualname)
    except AttributeError:
        return False
    return True


def load_definition_module(
    module_name: str, qualname: str, path: str, file_modules: dict[str, ModuleType]
) -> ModuleType:
    """Load the module of an object definition that has a source file

    The source file is only a fallback: a module that can be imported by name
    is preferred (the file path is host-specific). A module loaded from its
    file is executed only once, and cached in ``file_modules``.

    :param module_name: The module name
    :param qualname: The qualified name of the type, used to check that a
        module imported by name is the right one
    :param path: The path of the module source file
    :param file_modules: Modules already loaded, indexed by file path
    """
    if mod := file_modules.get(path):
        return mod

    if module_name != "_main_":
        mod = sys.modules.get(module_name)
        if mod is None:
            try:
                mod = importlib.import_module(module_name)
            except ModuleNotFoundError as e:
                # Do not hide import errors raised within the module
                if e.name != module_name:
                    raise
        if mod is not None and has_qualattr(mod, qualname):
            file_modules[path] = mod
            return mod

    if not Path(path).is_file():
        raise ModuleNotFoundError(
            f"Module {module_name} cannot be imported, and its source file "
            f"{path} does not exist",
            name=module_name,
        )

    with add_to_path(str(Path(path).parent)):
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

    file_modules[path] = mod
    return mod


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
        self.store: dict[int, Any] = {}
        self.constructed: set[int] = set()

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
