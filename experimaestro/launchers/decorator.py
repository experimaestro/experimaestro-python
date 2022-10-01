from copy import copy
from . import Launcher


class LauncherDecorator(Launcher):
    """Allows to modify the behavior of a launcher"""

    def __init__(self, launcher: Launcher):
        super().__init__(launcher.connector)
        self.launcher = launcher

        self.environ = copy(launcher.environ)
        self.notificationURL = launcher.notificationURL
