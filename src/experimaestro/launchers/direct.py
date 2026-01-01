from experimaestro.scriptbuilder import PythonScriptBuilder
from . import Launcher


class DirectLauncher(Launcher):
    """Launcher that runs tasks directly as local processes.

    This is the default launcher that executes tasks on the local machine
    without any job scheduler. Tasks are run as Python subprocesses.

    :param connector: The connector to use (defaults to LocalConnector)
    """

    def scriptbuilder(self):
        return PythonScriptBuilder()

    def launcher_info_code(self) -> str:
        """Returns empty string as local launcher has no time limits."""
        return ""

    def __str__(self):
        return f"DirectLauncher({self.connector})"
