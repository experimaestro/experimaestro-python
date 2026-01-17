"""Module mocking utilities for pre-experiment setup.

This module provides utilities to mock Python modules during experiment setup,
allowing code that imports heavy dependencies (like PyTorch, transformers, etc.)
to be parsed and configured without actually loading those libraries.

This is particularly useful for:
- Speeding up experiment configuration parsing
- Allowing dry-run simulations without heavy dependencies
- Static code analysis of experiment code

Example usage in a pre_experiment.py file:

    from experimaestro.experiments.mockmodule import FakeModuleFinder
    import sys

    sys.meta_path.insert(0, FakeModuleFinder(
        ['torch', 'transformers', 'pytorch_lightning'],
        decorators=[
            'torch.compile',
            'torch.jit.script',
            'torch.no_grad',
            'torch.inference_mode',
        ]
    ))
"""

import sys
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


class _FakeClass:
    """Fake class that can be inherited from without metaclass conflicts.

    This class allows code that inherits from mocked classes (like torch.nn.Module)
    to be parsed without errors. When used as a base class, it resolves to `object`
    to avoid metaclass conflicts.
    """

    _finder: "FakeModuleFinder | None" = None
    _path: str = ""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _FakeClass()

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
        """Tell Python to use object instead of this class when used as a base.

        This avoids metaclass conflicts when inheriting from real classes.
        """
        return (object,)


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
            # Return noop_decorator for missing attributes
            return noop_decorator

    def __call__(self, *args, **kwargs):
        return _FakeClassProxy(_FakeClass())

    def __mro_entries__(self, bases):
        """When used as a base, return the actual fake class."""
        return (object.__getattribute__(self, "_fake_class"),)


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
