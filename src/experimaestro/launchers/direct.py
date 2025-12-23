from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher


class DirectLauncher(Launcher):
    def scriptbuilder(self):
        return PythonScriptBuilder()

    def launcher_info_code(self) -> str:
        """Returns empty string as local launcher has no time limits."""
        return ""

    def __str__(self):
        return f"DirectLauncher({self.connector})"
