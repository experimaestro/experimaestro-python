"""Defines an experimental environment"""

from pathlib import Path
from typing import Dict
import marshmallow as mm
from ..connectors import parsepath
from experimaestro.utils.settings import JsonSettings, PathField
from pytools import memoize


def schema(schema_cls):
    def annotate(object_cls):
        schema_cls.OBJECT_CLS = object_cls
        object_cls.SCHEMA = schema_cls
        return object_cls

    return annotate


class _Schema(mm.Schema):
    @mm.post_load
    def make_settings(self, data, **kwargs):
        settings = self.__class__.OBJECT_CLS()
        for key, value in data.items():
            setattr(settings, key, value)
        return settings


class EnvironmentSchema(_Schema):
    hostname = mm.fields.Str()
    """The hostname (can be empty for localhost)"""

    pythonpath = mm.fields.Str()
    """Path to python executable"""
    workdir = mm.fields.Str()
    environ = mm.fields.Dict(keys=mm.fields.Str(), values=mm.fields.Str())


class Schema(_Schema):
    environments = mm.fields.Dict(
        keys=mm.fields.Str(), values=mm.fields.Nested(EnvironmentSchema)
    )


@schema(Schema)
class Settings(JsonSettings):
    """User settings"""

    def __init__(self):
        self.environments: Dict[str, str] = {}


@schema(EnvironmentSchema)
class Environment:
    """This defines the environment for an experiment, and can be stored"""

    def __init__(self, workdir=None):
        self.hostname = None
        self._workdir = workdir
        self.pythonpath = None
        self.environ = {}

    @property
    def basepath(self):
        if self.hostname:
            from ..connectors.ssh import SshPath

            return SshPath(f"ssh://{self.hostname}")
        return Path()

    @property
    def workdir(self):
        assert self._workdir, "The working directory has not been set"
        return self.basepath / self._workdir

    @workdir.setter
    def workdir(self, value):
        self._workdir = value

    def setenv(self, key: str, value: str):
        """Set the environment variable with key"""
        self.environ[key] = value

    @staticmethod
    @memoize()
    def _load():
        path = (
            Path("~").expanduser() / ".config" / "experimaestro" / "environments.json"
        )
        return Settings.load(path)

    @staticmethod
    def get(name: str):
        """Retrieve an environment by name"""
        return Environment._load().environments[name]
