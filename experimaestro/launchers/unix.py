from . import Launcher
from experimaestro.scriptbuilder import ShScriptBuilder


class UnixLauncher(Launcher):
    def scriptbuilder(self):
        return ShScriptBuilder()
