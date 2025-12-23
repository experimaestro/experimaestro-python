# Import Python modules

import inspect
from typing import Callable, Type as TypingType, TypeVar, Union
from sortedcontainers import SortedDict
import experimaestro.core.objects as objects
import experimaestro.core.types as types

from .core.objects import Config, Task
from .core.types import Any, TypeProxy, Type
from .utils import logger

# --- Annotations to define tasks and types

T = TypeVar("T")


def configmethod(method):
    """(deprecated) Annotate a method that should be kept in the configuration object"""
    return method


class Array(TypeProxy):
    """Array of object"""

    def __init__(self, type):
        self.type = Type.fromType(type)

    def __call__(self):
        return types.ArrayType(self.type)


class Choice(TypeProxy):
    """A string with a choice among several alternative"""

    def __init__(self, *args):
        self.choices = args

    def __call__(self):
        return types.StringType


def config_only(method):
    """Marks a configuration-only method"""
    assert inspect.ismethod(method)


# --- Path generators for task stdout/stderr


def STDERR(jobcontext, config):
    return "%s.err" % jobcontext.name


def STDOUT(jobcontext, config):
    return "%s.out" % jobcontext.name


# --- Cache


def cache(name: str):
    """Use a cache path for a given config"""

    def annotate(method):
        return objects.cache(method, name)

    return annotate


# --- Tags


def tag(value):
    """Tag a value"""
    return objects.TaggedValue(value)


class TagDict(SortedDict):
    """A hashable dictionary"""

    def __hash__(self):
        return hash(tuple((key, value) for key, value in self.items()))

    def __setitem__(self, key, value):
        raise Exception("A tag dictionary is not mutable")


def tags(value) -> TagDict:
    """Return the tags associated with a value"""
    return TagDict(value.__xpm__.tags())


def _normalizepathcomponent(v: Any):
    if isinstance(v, str):
        return v.replace("/", "-")
    return v


def tagspath(value: Config):
    """Return a unique path made of tags and their values"""
    return "_".join(
        f"""{_normalizepathcomponent(key)}={_normalizepathcomponent(value)}"""
        for key, value in tags(value).items()
    )


# --- Deprecated


def deprecate(
    config_or_target: Union[TypingType[Config], Callable, None] = None,
    *,
    replace: bool = False,
):
    """Deprecate a configuration / task or an attribute (via a method)

    Usage:

        # Method 1: Deprecated class inherits from new class (legacy)
        @deprecate
        class OldConfig(NewConfig):
            pass

        # Method 2: Specify target class explicitly with __convert__
        @deprecate(NewConfig)
        class OldConfig(Config):
            value: Param[int]

            def __convert__(self):
                return NewConfig.C(values=[self.value])

        # Method 3: Immediate replacement with __convert__
        @deprecate(NewConfig, replace=True)
        class OldConfig(Config):
            value: Param[int]

            def __convert__(self):
                return NewConfig.C(values=[self.value])

        # Method 4: Deprecate a parameter
        class MyConfig(Config):
            @deprecate
            def oldattribute(self, value):
                # Do something with the value
                pass

    When using @deprecate(TargetConfig), the deprecated class should define a
    __convert__ method that returns an equivalent instance of the target class.
    The identifier is computed from the converted configuration, so deprecated
    and new configurations will have the same identifier when equivalent.

    With replace=True, creating an instance of the deprecated class immediately
    returns the converted new config. The deprecated identifier is preserved for
    fix_deprecated to create symlinks between old and new job directories.
    """
    # Case 1: @deprecate on a function (deprecated attribute)
    if inspect.isfunction(config_or_target):
        from experimaestro.core.types import DeprecatedAttribute

        return DeprecatedAttribute(config_or_target)

    # Case 2: @deprecate (no parens) on a class - legacy pattern
    # The class inherits from its target (NewConfig), not directly from Config
    if config_or_target is not None and inspect.isclass(config_or_target):
        # Check if this looks like a deprecated class (legacy pattern)
        # Legacy pattern: @deprecate class OldConfig(NewConfig) where NewConfig is a Config subclass
        # The deprecated class inherits from exactly one Config subclass (the target)
        # We exclude Config and Task as base classes since those indicate the new pattern
        base_classes_for_new_pattern = (Config, Task)
        if (
            not replace
            and len(config_or_target.__bases__) == 1
            and config_or_target.__bases__[0] not in base_classes_for_new_pattern
            and issubclass(config_or_target.__bases__[0], Config)
        ):
            # This is the legacy pattern: @deprecate on a class
            deprecated_class = config_or_target
            deprecated_class.__getxpmtype__().deprecate()
            return deprecated_class

        # Otherwise, this is the new pattern: @deprecate(TargetConfig)
        target = config_or_target

        def decorator(deprecated_class: TypingType[Config]):
            deprecated_class.__getxpmtype__().deprecate(target=target, replace=replace)
            return deprecated_class

        return decorator

    # Case 3: @deprecate() with parentheses but no arguments (legacy, uses parent class)
    if config_or_target is None:

        def decorator(deprecated_class: TypingType[Config]):
            deprecated_class.__getxpmtype__().deprecate()
            return deprecated_class

        return decorator

    raise NotImplementedError("Cannot deprecate %s" % config_or_target)


def deprecateClass(klass):
    import inspect

    def __init__(self, *args, **kwargs):
        frameinfo = inspect.stack()[1]
        logger.warning(
            "Class %s is deprecated: use %s in %s:%s (%s)",
            klass.__name__,
            klass.__bases__[0].__name__,
            frameinfo.filename,
            frameinfo.lineno,
            frameinfo.code_context,
        )
        super(klass, self).__init__(*args, **kwargs)

    klass.__init__ = __init__
    return klass


def initializer(method):
    """Defines a method as an initializer that can only be called once"""

    def wrapper(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        setattr(self, method.__name__, lambda *args, **kwargs: value)

    return wrapper
