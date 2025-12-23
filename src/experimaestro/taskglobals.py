from functools import cached_property
from pathlib import Path
from typing import Optional


class LauncherInformation:
    """Minimal launcher information available during task execution.

    This is a lightweight class used to query launcher-specific information
    (like remaining time) during task execution. It's set by the generated
    Python script that runs the task.
    """

    def remaining_time(self) -> Optional[float]:
        """Returns the remaining time in seconds before the job times out.

        Returns:
            The remaining time in seconds, or None if no time limit.
        """
        return None


class Env:
    _instance = None

    # The working directory path
    wspath: Optional[Path] = None

    # The current task path
    taskpath: Optional[Path] = None

    # Launcher information (only set when running a task)
    launcher_info: Optional[LauncherInformation] = None

    # Set to True when multi-processing when
    # in slave mode:
    # - no progress report
    slave: bool = False

    @cached_property
    def xpm_path(self):
        path = self.taskpath / ".experimaestro"
        path.mkdir(exist_ok=True)
        return path

    @staticmethod
    def instance():
        if Env._instance is None:
            Env._instance = Env()
        return Env._instance
