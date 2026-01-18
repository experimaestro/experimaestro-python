"""Module mocking utilities for pre-experiment setup.

This module provides utilities to mock Python modules during experiment setup,
allowing code that imports heavy dependencies (like PyTorch, transformers, etc.)
to be parsed and configured without actually loading those libraries.

This is particularly useful for:
- Speeding up experiment configuration parsing
- Allowing dry-run simulations without heavy dependencies
- Static code analysis of experiment code

Example usage in a pre_experiment.py file:

    from experimaestro.experiments import mock_modules

    mock_modules(['torch', 'transformers', 'pytorch_lightning'])

    # All decorator patterns work automatically:
    # @torch.compile, @torch.no_grad(), @torch.jit.script, etc.
"""

import sys
import warnings
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


def noop_decorator(fn=None, *args, **kwargs):
    """No-op decorator that works with or without arguments.

    Can be used as:
        @noop_decorator
        def func(): ...

    Or:
        @noop_decorator(some_arg=True)
        def func(): ...
    """
    if fn is not None:
        return fn
    return lambda f: f


def _noop_callable(*args, **kwargs):
    """A callable that works as a universal no-op, including as a decorator.

    Handles both decorator patterns:
        @decorator
        def func(): ...

        @decorator(arg=1)
        def func(): ...
    """
    # Case 1: @decorator - called directly with a function
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    # Case 2: @decorator(...) - called with arguments, return a decorator
    return _noop_callable


class _NoopCallableDescriptor:
    """Descriptor that provides a no-op callable for both class and instance access.

    This allows attributes like `MyClass.apply(...)` and `instance.apply(...)`
    to work without requiring a custom metaclass.
    """

    def __get__(self, obj, objtype=None):
        return _noop_callable


class _FakeBaseClass:
    """Base class used when inheriting from fake classes.

    When fake classes are used as base classes via __mro_entries__, this class
    is returned instead of object. This provides common attributes like `apply`
    that some libraries (e.g., torch.autograd.Function) expect on subclasses.

    This class deliberately does NOT use a custom metaclass to avoid metaclass
    conflicts when mixed with other classes that have custom metaclasses.
    """

    # Common class-level attributes that should be callable.
    # These are defined as descriptors so they work for both class and instance access.
    apply = _NoopCallableDescriptor()

    def __init__(self, *args, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class _FakeClass:
    """Fake class that can be inherited from without metaclass conflicts.

    This class allows code that inherits from mocked classes (like torch.nn.Module)
    to be parsed without errors. When used as a base class, it resolves to
    `_FakeBaseClass` to provide dynamic attribute access on subclasses.
    """

    _finder: "FakeModuleFinder | None" = None
    _path: str = ""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        # If called with a single callable (decorator pattern), return it unchanged
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # Otherwise return self to allow chaining: @decorator(arg=1) works
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        full_path = f"{self._path}.{name}"
        if self._finder and full_path in self._finder.decorators:
            return noop_decorator
        return (
            self._finder._make_class(full_path, name) if self._finder else _FakeClass()
        )

    @classmethod
    def __class_getitem__(cls, item):
        """Support for subscript notation like List[int] or TextEncoderBase[str, Tensor]."""
        return cls

    def __mro_entries__(self, bases):
        """Tell Python to use _FakeBaseClass when this class is used as a base.

        This provides common attributes like `apply` on subclasses.

        Returns an empty tuple if _FakeBaseClass is already in another base's MRO
        to avoid "duplicate base class" errors.
        """
        for base in bases:
            if base is self:
                continue
            # Check if base already has _FakeBaseClass in its MRO
            if hasattr(base, "__mro__") and _FakeBaseClass in base.__mro__:
                return ()
            # Check if base is another fake class/proxy (without recursing)
            if isinstance(base, (_FakeClass, _FakeClassProxy)):
                return ()
        return (_FakeBaseClass,)


class _FakeClassProxy:
    """Wrapper that intercepts attribute access on fake classes."""

    def __init__(self, fake_class):
        object.__setattr__(self, "_fake_class", fake_class)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fake_class = object.__getattribute__(self, "_fake_class")
        # Try to get from the fake class first
        try:
            return getattr(fake_class, name)
        except AttributeError:
            # Create a nested fake class for attribute access (e.g., torch.autograd.Function)
            finder = getattr(fake_class, "_finder", None)
            path = getattr(fake_class, "_path", "")
            full_path = f"{path}.{name}" if path else name
            if finder and full_path in finder.decorators:
                return noop_decorator
            if finder:
                return finder._make_class(full_path, name)
            return noop_decorator

    def __call__(self, *args, **kwargs):
        # If called with a single callable (decorator pattern), return it unchanged
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # Otherwise return self to allow chaining: @decorator(arg=1) works
        return self

    def __getitem__(self, item):
        """Support for subscript notation like Tensor[int] or Module[str]."""
        return self

    def __mro_entries__(self, bases):
        """When used as a base, return _FakeBaseClass for common attributes.

        Returns an empty tuple if _FakeBaseClass is already in another base's MRO
        to avoid "duplicate base class" errors.
        """
        for base in bases:
            if base is self:
                continue
            # Check if base already has _FakeBaseClass in its MRO
            if hasattr(base, "__mro__") and _FakeBaseClass in base.__mro__:
                return ()
            # Check if base is another fake class/proxy (without recursing)
            if isinstance(base, (_FakeClass, _FakeClassProxy)):
                return ()
        return (_FakeBaseClass,)


class FakeModule(ModuleType):
    """A fake module that returns fake classes for any attribute access."""

    def __init__(self, name: str, finder: "FakeModuleFinder"):
        super().__init__(name)
        self.__path__: list[str] = []
        self._finder = finder
        self._cache: dict[str, _FakeClassProxy] = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        full_path = f"{self.__name__}.{name}"

        if full_path in self._finder.decorators:
            return noop_decorator

        if name not in self._cache:
            self._cache[name] = self._finder._make_class(full_path, name)

        return self._cache[name]


class FakeModuleFinder(MetaPathFinder):
    """A meta path finder that intercepts imports for specified modules.

    When registered in sys.meta_path, this finder will intercept imports for
    the specified module names and return fake modules that silently accept
    any attribute access, method calls, and decorator applications.

    Args:
        modules: List of module names to mock (e.g., ['torch', 'transformers']).
            Submodules are automatically included (e.g., 'torch' includes 'torch.nn').
        decorators: List of full paths to treat as decorators (e.g., ['torch.compile']).
            These will return noop_decorator when accessed.

    Example:
        >>> import sys
        >>> from experimaestro.experiments.mockmodule import FakeModuleFinder
        >>> finder = FakeModuleFinder(
        ...     ['torch', 'transformers'],
        ...     decorators=['torch.compile', 'torch.no_grad']
        ... )
        >>> sys.meta_path.insert(0, finder)
        >>> import torch  # Now returns a fake module
        >>> @torch.no_grad()  # Works as a no-op decorator
        ... def my_func():
        ...     pass
    """

    def __init__(self, modules: list[str], decorators: list[str] | None = None):
        self.modules = set(modules)
        self.decorators = set(decorators or [])
        self._class_cache: dict[str, _FakeClassProxy] = {}

    def find_spec(self, fullname, path, target=None):
        if any(fullname == m or fullname.startswith(m + ".") for m in self.modules):
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        fullname = spec.name
        if fullname in sys.modules:
            return sys.modules[fullname]
        module = FakeModule(fullname, self)
        sys.modules[fullname] = module
        return module

    def exec_module(self, module):
        pass

    def _make_class(self, path: str, name: str) -> _FakeClassProxy:
        if path not in self._class_cache:
            cls = type(
                name,
                (_FakeClass,),
                {
                    "_finder": self,
                    "_path": path,
                },
            )
            # Wrap the class so attribute access is intercepted
            self._class_cache[path] = _FakeClassProxy(cls)
        return self._class_cache[path]

    def __repr__(self):
        return f"FakeModuleFinder({self.modules})"


def mock_modules(
    modules: list[str], decorators: list[str] | None = None
) -> FakeModuleFinder:
    """Mock specified modules so they can be imported without the actual dependencies.

    This is a convenience function that creates a FakeModuleFinder and registers it
    in sys.meta_path. Use this in pre_experiment.py files to speed up experiment
    configuration parsing.

    All mocked objects automatically work as decorators, supporting both patterns:
        @decorator
        def func(): ...

        @decorator(arg=1)
        def func(): ...

    Args:
        modules: List of module names to mock (e.g., ['torch', 'transformers']).
            Submodules are automatically included (e.g., 'torch' includes 'torch.nn').
        decorators: Deprecated. No longer needed as all mocked objects now
            automatically work as decorators.

    Returns:
        The FakeModuleFinder instance that was registered.

    Example:
        >>> from experimaestro.experiments import mock_modules
        >>> mock_modules(['torch', 'transformers'])
        >>> import torch  # Now returns a fake module
        >>> @torch.no_grad()  # Works as a no-op decorator
        ... def my_func():
        ...     pass
        >>> @torch.compile
        ... def my_other_func():
        ...     pass
    """
    if decorators is not None:
        warnings.warn(
            "The 'decorators' parameter is deprecated and will be removed in a future "
            "version. All mocked objects now automatically work as decorators.",
            DeprecationWarning,
            stacklevel=2,
        )
    finder = FakeModuleFinder(modules, decorators)
    sys.meta_path.insert(0, finder)
    return finder
