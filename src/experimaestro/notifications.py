"""File-based progress notification system for experimaestro tasks.

Progress is reported by writing to job event files, which are then read
by monitors (TUI, web UI) via file watching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterator, Optional, TypeVar, overload
import sys
from tqdm.auto import tqdm as std_tqdm

from .utils import logger
from experimaestro.taskglobals import Env as TaskEnv
from .progress import FileBasedProgressReporter

T = TypeVar("T")


@dataclass
class LevelInformation:
    """Progress information for a single nesting level"""

    level: int
    desc: Optional[str]
    progress: float

    def to_dict(self) -> Dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "level": self.level,
            "desc": self.desc,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LevelInformation":
        """Create LevelInformation from a dictionary (e.g., from JSON).

        Args:
            d: Dictionary with keys 'level', 'progress', and optionally 'desc'

        Returns:
            LevelInformation instance
        """
        return cls(
            level=d.get("level", 0),
            desc=d.get("desc"),
            progress=d.get("progress", 0),
        )

    def __repr__(self) -> str:
        return f"[{self.level}] {self.desc} {int(self.progress * 1000) / 10}%"


# Type alias for progress information
ProgressInformation = list[LevelInformation]


def get_progress_information_from_dict(dicts: list[dict]) -> ProgressInformation:
    """Convert a list of progress dicts to ProgressInformation.

    Handles both dict and LevelInformation items for robustness.

    Args:
        dicts: List of dictionaries with 'level', 'progress', 'desc' keys

    Returns:
        List of LevelInformation instances (ProgressInformation)
    """
    return [LevelInformation.from_dict(p) if isinstance(p, dict) else p for p in dicts]


class Reporter:
    """File-based progress reporter for running tasks.

    Progress events are written to job event files at:
    .events/jobs/{task_id}/event-{job_id}-*.jsonl

    These files are watched by monitors (TUI, web UI) to display progress.
    """

    def __init__(self, path: Path):
        """Initialize the file-based reporter.

        Args:
            path: The task path ({workspace}/jobs/{task_id}/{job_id}/)
        """
        self.path = path
        self.file_reporter = FileBasedProgressReporter(task_path=path)
        self.levels: list[LevelInformation] = [LevelInformation(0, None, -1)]
        self.console = False

    def eoj(self):
        """End of job notification"""
        self.file_reporter.eoj()

    def set_progress(
        self, progress: float, level: int, desc: Optional[str], console=False
    ):
        """Set progress for a specific level.

        Args:
            progress: Progress value between 0.0 and 1.0
            level: Nesting level (0 is top level)
            desc: Optional description
            console: If True, also print to console
        """
        # Update in-memory levels
        self.levels = self.levels[: (level + 1)]
        while level >= len(self.levels):
            self.levels.append(LevelInformation(level, None, 0.0))
        if desc:
            self.levels[level].desc = desc
        self.levels[level].progress = progress

        # Write to file
        self.file_reporter.set_progress(progress, level, desc)

        # Optionally log to console
        if console:
            logger.info("Progress: %s", self.levels[level])

    INSTANCE: ClassVar[Optional["Reporter"]] = None

    @staticmethod
    def instance():
        """Get or create the singleton Reporter instance."""
        if Reporter.INSTANCE is None:
            taskpath = TaskEnv.instance().taskpath
            assert taskpath is not None, "Task path is not defined"
            Reporter.INSTANCE = Reporter(taskpath)
        return Reporter.INSTANCE


def progress(value: float, level=0, desc: Optional[str] = None, console=False):
    """Report task progress.

    Call this function from within a running task to report progress.
    Progress is written to job event files and displayed in monitors.

    Example::

        for i, batch in enumerate(dataloader):
            train(batch)
            progress(i / len(dataloader), desc="Training")

    :param value: Progress value between 0.0 and 1.0
    :param level: Nesting level for nested progress bars (default: 0)
    :param desc: Optional description of the current operation
    :param console: If True, also print to console
    """
    if TaskEnv.instance().slave:
        # Skip if in a slave process
        return
    Reporter.instance().set_progress(value, level, desc, console=console)


def report_eoj():
    """Notify that the job is over"""
    Reporter.instance().eoj()


def start_of_job():
    """Notify that the job has started running"""
    Reporter.instance().file_reporter.start_of_job()


class xpm_tqdm(std_tqdm):
    """Experimaestro-aware tqdm progress bar.

    A drop-in replacement for ``tqdm`` that automatically reports progress
    to job event files. Use this instead of the standard ``tqdm``
    in your task's ``execute()`` method.

    Example::

        from experimaestro import tqdm

        class MyTask(Task):
            def execute(self):
                for batch in tqdm(dataloader, desc="Training"):
                    train(batch)
    """

    def __init__(self, iterable=None, file=None, *args, **kwargs):
        _file = file or sys.stderr
        self.is_tty = hasattr(_file, "isatty") or _file.isatty()

        super().__init__(iterable, disable=False, file=file, *args, **kwargs)
        progress(0.0, level=self.pos, desc=kwargs.get("desc", None), console=False)

    def update(self, n=1):
        result = super().update(n)
        if self.total is not None and self.total > 0:
            progress(self.n / self.total, level=self.pos, console=False)
        return result

    def refresh(self, nolock=False, lock_args=None):
        if self.is_tty:
            super().refresh(nolock=nolock, lock_args=lock_args)


@overload
def tqdm(**kwargs) -> xpm_tqdm: ...


@overload
def tqdm(iterable: Optional[Iterator[T]] = None, **kwargs) -> Iterator[T]: ...


def tqdm(*args, **kwargs):
    """Create an experimaestro-aware progress bar.

    A drop-in replacement for ``tqdm.tqdm`` that automatically reports progress
    to job event files. Use this in task ``execute()`` methods.

    Example::

        from experimaestro import tqdm

        for epoch in tqdm(range(100), desc="Epochs"):
            for batch in tqdm(dataloader, desc="Batches"):
                train(batch)

    :param iterable: Iterable to wrap (optional)
    :param kwargs: Additional arguments passed to tqdm
    :return: A progress bar iterator
    """
    return xpm_tqdm(*args, **kwargs)  # type: ignore
