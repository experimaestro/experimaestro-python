from pathlib import Path
from enum import Enum
from typing import (
    Any,
    Dict,
    Tuple,
)
from contextlib import contextmanager


class ConfigWalkContext:
    """Context when generating values in configurations"""

    @property
    def path(self):
        """Returns the path of the job directory"""
        raise NotImplementedError()

    def __init__(self):
        self._configpath = None

    @property
    def task(self):
        return None

    def currentpath(self) -> Path:
        """Returns the configuration folder"""
        if self._configpath:
            return self.path / self._configpath
        return self.path

    @contextmanager
    def push(self, key: str):
        """Push a new key to contextualize paths"""
        p = self._configpath
        try:
            self._configpath = (Path("out") if p is None else p) / key
            yield key
        finally:
            self._configpath = p


class ConfigWalk:
    """Allows to perform an operation on all nested configurations"""

    def __init__(self, context: ConfigWalkContext = None, recurse_task=False):
        """

        :param recurse_task: Recurse into linked tasks
        :param context: The context, by default only tracks the position in the
            config tree
        """
        self.recurse_task = recurse_task
        self.context = ConfigWalkContext() if context is None else context

        # Stores already visited nodes
        self.visited = {}

    def preprocess(self, config) -> Tuple[bool, Any]:
        """Returns a tuple boolean/value

        The boolean value is used to stop the processing if False.
        The value is returned
        """
        return True, None

    def postprocess(self, stub, config, values: Dict[str, Any]):
        return stub

    def list(self, i: int):
        return self.context.push(str(i))

    def map(self, k: str):
        return self.context.push(k)

    def stub(self, config):
        return config

    def __call__(self, x):
        from experimaestro.core.objects import Config
        from experimaestro.core.objects import ConfigInformation  # noqa: F401

        if isinstance(x, Config):
            info = x.__xpm__  # type: ConfigInformation

            # Avoid loops
            xid = id(x)
            if xid in self.visited:
                return self.visited[xid]

            # Get a stub
            stub = self.stub(x)
            self.visited[xid] = stub

            # Pre-process
            flag, value = self.preprocess(x)

            if not flag:
                # Stop processing and returns value
                return value

            # Process all the arguments
            result = {}
            for arg, v in info.xpmvalues():
                if v is not None:
                    with self.map(arg.name):
                        result[arg.name] = self(v)
                else:
                    result[arg.name] = None

            # Deals with pre-tasks
            if info.pre_tasks:
                with self.map("__pre_tasks__"):
                    self(info.pre_tasks)

            if info.init_tasks:
                with self.map("__init_tasks__"):
                    self(info.init_tasks)

            # Process task if different
            if (
                x.__xpm__.task is not None
                and self.recurse_task
                and x.__xpm__.task is not x
            ):
                self(x.__xpm__.task)

            processed = self.postprocess(stub, x, result)
            self.visited[xid] = processed
            return processed

        if x is None:
            return None

        if isinstance(x, list):
            result = []
            for i, sv in enumerate(x):
                with self.list(i):
                    result.append(self(sv))
            return result

        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                assert isinstance(key, (str, float, int))
                with self.map(key):
                    result[key] = self(value)
            return result

        if isinstance(x, (float, int, str, Path, Enum)):
            return x

        raise NotImplementedError(f"Cannot handle a value of type {type(x)}")
