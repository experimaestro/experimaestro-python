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
    """Decorator for caching method results to disk.

    The cache is stored in the workspace's config directory, keyed by the
    configuration's identifier.

    Example::

        class MyConfig(Config):
            data_path: Param[Path]

            @cache("processed.pkl")
            def process(self, cache_path: Path):
                if cache_path.exists():
                    return pickle.load(cache_path.open("rb"))
                result = expensive_computation(self.data_path)
                pickle.dump(result, cache_path.open("wb"))
                return result

    :param name: Filename for the cache file
    :return: A decorator that wraps the method with caching logic
    """

    def annotate(method):
        return objects.cache(method, name)

    return annotate


# --- Tags


def tag(value):
    """Tag a parameter value for tracking in experiments.

    Tagged values appear in experiment logs and can be used for filtering
    and organizing results. Tags are included in the task's ``__tags__``
    dictionary.

    Example::

        task = MyTask.C(
            learning_rate=tag(0.001),  # Will appear in task tags
            batch_size=32,
        ).submit()

    :param value: The value to tag (str, int, float, or bool)
    :return: A tagged value wrapper that preserves the original value
    """
    return objects.TaggedValue(value)


class TagDict(SortedDict):
    """A hashable dictionary"""

    def __hash__(self):
        return hash(tuple((key, value) for key, value in self.items()))

    def __setitem__(self, key, value):
        raise Exception("A tag dictionary is not mutable")


def tags(value) -> TagDict:
    """Return the tags associated with a configuration.

    Returns a dictionary of all tagged parameter values from this configuration
    and its nested configurations.

    Example::

        config = MyTask.C(learning_rate=tag(0.001), epochs=tag(100))
        task_tags = tags(config)  # {"learning_rate": 0.001, "epochs": 100}

    :param value: A configuration object
    :return: A TagDict with tag names as keys and tagged values as values
    """
    return TagDict(value.__xpm__.tags())


def _normalizepathcomponent(v: Any):
    if isinstance(v, str):
        return v.replace("/", "-")
    return v


def tagspath(value: Config) -> str:
    """Generate a unique path string from a configuration's tags.

    Useful for creating tag-based directory structures. Tags are sorted
    alphabetically and joined with underscores.

    Example::

        config = MyTask.C(learning_rate=tag(0.001), epochs=tag(100))
        path = tagspath(config)  # "epochs=100_learning_rate=0.001"

    :param value: A configuration object
    :return: A string with sorted tags in ``key=value`` format, joined by ``_``
    """
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
    """Deprecate a configuration/task class or a parameter.

    Deprecated configurations maintain backwards compatibility while allowing
    migration to new structures. The identifier is computed from the converted
    configuration, ensuring consistency.

    **Usage patterns:**

    1. Simple deprecation (class inherits from new class)::

        @deprecate
        class OldConfig(NewConfig):
            pass

    2. Deprecation with conversion::

        @deprecate(NewConfig)
        class OldConfig(Config):
            value: Param[int]

            def __convert__(self):
                return NewConfig.C(values=[self.value])

    3. Immediate replacement::

        @deprecate(NewConfig, replace=True)
        class OldConfig(Config):
            value: Param[int]

            def __convert__(self):
                return NewConfig.C(values=[self.value])

    4. Deprecate a parameter::

        class MyConfig(Config):
            new_param: Param[list[int]]

            @deprecate
            def old_param(self, value: int):
                self.new_param = [value]

    :param config_or_target: Target class for conversion, or the deprecated
        class/method when used as a simple decorator
    :param replace: If True, creating the deprecated class immediately returns
        the converted instance
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
    """Decorator for methods that should only execute once.

    After the first call, subsequent calls return the cached result.
    This is useful for lazy initialization of expensive resources.

    Example::

        class MyConfig(Config):
            @initializer
            def model(self):
                return load_expensive_model()

    :param method: The method to wrap
    :return: A wrapper that caches the result after first execution
    """

    def wrapper(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        setattr(self, method.__name__, lambda *args, **kwargs: value)

    return wrapper
