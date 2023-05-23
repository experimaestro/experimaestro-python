from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TypeVar

from experimaestro import Param

from .objects import Config, Proxy
from .arguments import DataPath

T = TypeVar("T")


class SerializedConfig(Config, Proxy, ABC):
    """A serializable configuration

    This can be used to define a loading mechanism when instanciating the
    configuration
    """

    config: Param[Config]
    """The configuration that will be serialized"""

    registered: List[Config]
    """(execution only) List of configurations that use this serialized config"""

    def __post_init__(self):
        super().__post_init__()
        self.registered = []

    def register(self, config: Config):
        self.registered.append(config)

    def __unwrap__(self):
        return self.config

    @abstractmethod
    def initialize(self):
        """Initialize the object

        This might imply loading saved data (e.g. learned models)
        """
        ...


class PathBasedSerializedConfig(SerializedConfig):
    """A path based serialized configuration

    The most common case it to have
    """

    path: DataPath[Path]
    """Path containing the data"""
