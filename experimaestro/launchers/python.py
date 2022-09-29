from experimaestro.registry import LauncherRegistry
from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher


class PythonLauncher(Launcher):
    def scriptbuilder(self):
        return PythonScriptBuilder()

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        # registry.register_launcher(PythonLauncherConfiguration)
        pass
