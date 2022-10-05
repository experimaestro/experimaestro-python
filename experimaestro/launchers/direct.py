from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional
from experimaestro.launcherfinder import (
    LauncherConfiguration,
    LauncherRegistry,
    HostRequirement,
)
from experimaestro.launcherfinder.registry import CPU, GPU, GPUList, YAMLDataClass
from experimaestro.launcherfinder.specs import (
    CPUSpecification,
    CudaSpecification,
    HostSpecification,
)
from experimaestro.launchers.decorator import LauncherDecorator
from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher


class DirectLauncher(Launcher):
    def scriptbuilder(self):
        return PythonScriptBuilder()

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        registry.register_launcher("local", DirectLauncherConfiguration)


@dataclass
class DirectLauncherConfiguration(YAMLDataClass, LauncherConfiguration):
    connector: str = "connector"
    cpu: CPU = field(default_factory=CPU)
    gpus: GPUList = field(default_factory=GPUList)
    tokens: Optional[Dict[str, int]] = None
    tags: List[str] = field(default_factory=lambda: [])
    weight: int = 0

    @cached_property
    def spec(self) -> HostSpecification:
        return HostSpecification(self.cpu.to_spec(), self.gpus.to_spec())

    def get(
        self, registry: LauncherRegistry, requirement: "HostRequirement"
    ) -> Optional[Launcher]:
        if requirement.match(self.spec):
            launcher = DirectLauncher(connector=registry.getConnector(self.connector))
            if self.tokens:
                for token_identifier, count in self.tokens.items():
                    token = registry.getToken(token_identifier)

                    launcher.addListener(
                        lambda job: job.dependencies.add(token.dependency(count))
                    )
            return launcher

        return None
