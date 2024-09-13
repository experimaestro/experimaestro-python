from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher


class DirectLauncher(Launcher):
    def scriptbuilder(self):
        return PythonScriptBuilder()

    def __str__(self):
        return f"DirectLauncher({self.connector})"
