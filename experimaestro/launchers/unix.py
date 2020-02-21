from . import Launcher
from experimaestro.connectors.local import LocalProcessBuilder
from experimaestro.scriptbuilder import ShScriptBuilder


class UnixLauncher(Launcher):
    def scriptbuilder(self):
        return ShScriptBuilder()

    def processbuilder(self):
        return LocalProcessBuilder()
