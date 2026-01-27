from pathlib import Path, PosixPath
import random
from typing import Callable, Dict, List, Optional
from experimaestro.commandline import AbstractCommand, Job, CommandLineJob
from experimaestro.connectors import Connector
from experimaestro.connectors.local import ProcessBuilder, LocalConnector
from experimaestro.connectors.ssh import SshPath, SshConnector
from abc import ABC, abstractmethod


class ScriptBuilder(ABC):
    """A script builder is responsible for generating the script
    used to launch a command line job"""

    lockfiles: List[Path]
    """The files that must be locked before starting the job"""

    command: "AbstractCommand"
    """Command to be run"""

    @abstractmethod
    def write(self, job: CommandLineJob) -> Path:
        """Write the commmand line job

        :params job: The job to be written
        """
        ...


SubmitListener = Callable[[Job], None]
"""Listen to job submissions"""


class Launcher(ABC):
    """Base class for task launchers.

    Launchers are responsible for executing tasks on a compute resource.
    They work with a :class:`~experimaestro.connectors.Connector` to
    access the target system and manage process execution.

    Subclasses include:

    - :class:`~experimaestro.launchers.direct.DirectLauncher`: Local execution
    - :class:`~experimaestro.launchers.slurm.SlurmLauncher`: SLURM cluster

    :param connector: The connector to use for accessing the compute resource
    :param priority: Priority for launcher selection in DynamicLauncher (higher = preferred)
    """

    submit_listeners: List[SubmitListener]
    priority: float
    """Priority for launcher selection (higher values = higher priority)"""

    def __init__(self, connector: Connector, *, priority: float = 0):
        self.connector = connector
        self.environ: Dict[str, str] = {}
        self.notificationURL: Optional[str] = None
        self.submit_listeners = []
        self.priority = priority

    def setenv(self, key: str, value: str):
        self.environ[key] = value

    def setNotificationURL(self, url: Optional[str]):
        self.notificationURL = url

    @abstractmethod
    def scriptbuilder(self) -> ScriptBuilder:
        """Returns a script builder"""
        ...

    def addListener(self, listener: SubmitListener):
        self.submit_listeners.append(listener)

    def onSubmit(self, job: Job):
        """Called when submitting a job

        Example of use: this allows the launcher to add token dependencies
        """
        for listener in self.submit_listeners:
            listener(job)

    def processbuilder(self) -> ProcessBuilder:
        """Returns the process builder for this launcher

        By default, returns the associated connector builder"""
        return self.connector.processbuilder()

    @abstractmethod
    def launcher_info_code(self) -> str:
        """Returns Python code to set up launcher info during task execution.

        This code is inserted into the generated task script to set up
        launcher-specific information (like LauncherInformation for
        querying remaining time).

        Returns:
            Python code as a string, or empty string if no setup needed.
        """
        ...

    @staticmethod
    def get(path: Path):
        """Get a default launcher for a given path"""
        if isinstance(path, PosixPath):
            from .direct import DirectLauncher

            return DirectLauncher(LocalConnector())

        if isinstance(path, SshPath):
            from .direct import DirectLauncher

            return DirectLauncher(SshConnector.fromPath(path))
        raise ValueError("Cannot create a default launcher for %s", type(path))


class DynamicLauncher(Launcher):
    """A launcher that dynamically selects from a list of launchers based on priority.

    This launcher maintains a sorted list of sub-launchers and delegates
    execution to one of them based on their priorities.

    The selection behavior is controlled by the ``sample`` parameter:

    - If ``sample=False`` (default): Selects the highest-priority launcher.
      If multiple launchers share the highest priority, one is chosen uniformly
      at random.
    - If ``sample=True``: Samples a launcher with probability proportional
      to its priority. All priorities must be positive in this mode.

    Before each job launch, the :meth:`update` method is called, which can
    be overridden in subclasses to refresh or modify the launcher list
    (e.g., querying cluster availability).

    :param launchers: List of launchers to select from
    :param sample: If True, sample proportionally to priority; if False,
        pick highest priority (default: False)
    :param connector: Optional connector (defaults to first launcher's connector)

    Example::

        from experimaestro.launchers import DynamicLauncher
        from experimaestro.launchers.slurm import SlurmLauncher

        # Create launchers with different priorities
        fast_launcher = SlurmLauncher(options=SlurmOptions(partition="fast"), priority=10)
        slow_launcher = SlurmLauncher(options=SlurmOptions(partition="slow"), priority=1)

        # DynamicLauncher will prefer fast_launcher due to higher priority
        dynamic = DynamicLauncher([fast_launcher, slow_launcher])

        # With sampling, fast_launcher has 10/(10+1) ≈ 91% chance of being selected
        dynamic_sampled = DynamicLauncher([fast_launcher, slow_launcher], sample=True)
    """

    def __init__(
        self,
        launchers: List[Launcher],
        *,
        sample: bool = False,
        connector: Optional[Connector] = None,
        priority: float = 0,
    ):
        if not launchers:
            raise ValueError("DynamicLauncher requires at least one launcher")

        # Use first launcher's connector if not specified
        if connector is None:
            connector = launchers[0].connector

        super().__init__(connector, priority=priority)
        self._launchers = launchers
        self._sample = sample
        self._selected_launcher: Optional[Launcher] = None

    @property
    def launchers(self) -> List[Launcher]:
        """Returns launchers sorted by priority (highest first)."""
        return sorted(self._launchers, key=lambda x: x.priority, reverse=True)

    def add_launcher(self, launcher: Launcher) -> None:
        """Add a launcher to the list.

        :param launcher: The launcher to add
        """
        self._launchers.append(launcher)

    def remove_launcher(self, launcher: Launcher) -> None:
        """Remove a launcher from the list.

        :param launcher: The launcher to remove
        """
        self._launchers.remove(launcher)

    def update(self) -> None:
        """Called before selecting a launcher.

        Override this method in subclasses to refresh the launcher list
        based on external factors (e.g., cluster availability, queue status).

        By default, does nothing.
        """
        pass

    def select_launcher(self) -> Launcher:
        """Select a launcher based on priority and sampling mode.

        :returns: The selected launcher
        """
        self.update()

        sorted_launchers = self.launchers
        if not sorted_launchers:
            raise RuntimeError("No launchers available after update()")

        if self._sample:
            # Sample proportionally to priority
            priorities = [lnch.priority for lnch in sorted_launchers]
            if any(p <= 0 for p in priorities):
                raise ValueError(
                    "All launcher priorities must be positive when sample=True"
                )
            total = sum(priorities)
            weights = [p / total for p in priorities]
            return random.choices(sorted_launchers, weights=weights, k=1)[0]
        else:
            # Pick highest priority, uniform random among ties
            max_priority = sorted_launchers[0].priority
            top_launchers = [
                lnch for lnch in sorted_launchers if lnch.priority == max_priority
            ]
            return random.choice(top_launchers)

    def scriptbuilder(self) -> ScriptBuilder:
        """Returns the script builder from the selected launcher."""
        self._selected_launcher = self.select_launcher()
        return self._selected_launcher.scriptbuilder()

    def processbuilder(self) -> ProcessBuilder:
        """Returns the process builder from the selected launcher."""
        if self._selected_launcher is None:
            self._selected_launcher = self.select_launcher()
        return self._selected_launcher.processbuilder()

    def launcher_info_code(self) -> str:
        """Returns launcher info code from the selected launcher."""
        if self._selected_launcher is None:
            self._selected_launcher = self.select_launcher()
        return self._selected_launcher.launcher_info_code()

    def onSubmit(self, job: Job):
        """Called when submitting a job."""
        super().onSubmit(job)
        if self._selected_launcher is not None:
            self._selected_launcher.onSubmit(job)

    def __str__(self):
        launcher_strs = ", ".join(str(lnch) for lnch in self._launchers[:3])
        if len(self._launchers) > 3:
            launcher_strs += f", ... ({len(self._launchers)} total)"
        return f"DynamicLauncher([{launcher_strs}], sample={self._sample})"
