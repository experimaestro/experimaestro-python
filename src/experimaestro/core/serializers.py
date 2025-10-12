from typing import List, TypeVar
from experimaestro import Param

from .objects import Config, LightweightTask
from .arguments import DataPath


class SerializationLWTask(LightweightTask):
    """A serializable configuration

    This can be used to define a loading mechanism when instanciating the
    configuration
    """

    value: Param[Config]
    """The configuration that will be serialized"""

    registered: List[Config]
    """(execution only) List of configurations that use this serialized config"""

    def __post_init__(self):
        super().__post_init__()
        self.registered = []

    def register(self, config: Config):
        self.registered.append(config)


T = TypeVar("T", bound=Config)


class PathSerializationLWTask(SerializationLWTask):
    """A path based serialized configuration

    The most common case it to have
    """

    path: DataPath
    """Path containing the data"""
