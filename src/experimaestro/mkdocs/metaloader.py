# Utility class that allow to avoid installing modules
# when building documentation

import sys
import re
import importlib.abc
import importlib.machinery


class Spec:
    def __init__(self, name):
        self.name = name


class Module:
    def __init__(self, loader, spec):
        self.__loader__ = loader
        self.__spec__ = spec
        self.__path__ = None
        self.__package__ = spec.name
        _, self.__name__ = f".{spec.name}".rsplit(".", 1)

    def __getattr__(self, key):
        return Module(self.__loader__, Spec(f"{self.__package__}.{key}"))

    def __repr__(self):
        return f"Fake module ({self.__package__})"


class DependencyInjectorLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return Module(self, spec)

    def exec_module(self, module):
        pass


class DependencyInjectorFinder(importlib.abc.MetaPathFinder):
    def __init__(self, loader, matcher):
        self._loader = loader
        self.matcher = matcher

    def find_spec(self, fullname: str, path, target=None):
        # If we match, then do it
        if self.matcher.match(fullname):
            return self._gen_spec(fullname)

    def _gen_spec(self, fullname):
        spec = importlib.machinery.ModuleSpec(fullname, self._loader)
        return spec

    @staticmethod
    def install(matcher):
        _loader = DependencyInjectorLoader()
        _finder = DependencyInjectorFinder(_loader, matcher)
        sys.meta_path.append(_finder)
