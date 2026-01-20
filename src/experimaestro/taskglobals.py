from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from experimaestro.carbon.base import BaseCarbonTracker
    from experimaestro.core.objects import Config


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

    # Carbon tracker (only set when carbon tracking is enabled)
    carbon_tracker: Optional["BaseCarbonTracker"] = None

    # Set to True when multi-processing when
    # in slave mode:
    # - no progress report
    slave: bool = False

    # The currently executing task (set during task.execute())
    # Used by signal handlers for graceful termination
    current_task: Optional["Config"] = None

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
