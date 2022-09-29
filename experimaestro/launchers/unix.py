from dataclasses import dataclass
from typing import List, Optional
from experimaestro.registry import (
    CPU,
    GPU,
    LauncherRegistry,
    LauncherSpec,
    YAMLDataClass,
)
from . import Launcher
from experimaestro.scriptbuilder import ShScriptBuilder


@dataclass
class UnixLauncherConfiguration(YAMLDataClass):
    yaml_tag = "!unixlauncher"

    connector: str
    cpu: Optional[CPU] = None
    gpus: Optional[List[GPU]] = None
    tokens: Optional[List[str]] = None

    def fullfills_spec(self, spec: LauncherSpec):
        return False


class UnixLauncher(Launcher):
    def scriptbuilder(self):
        return ShScriptBuilder()

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        registry.register_launcher("unix", UnixLauncherConfiguration)
