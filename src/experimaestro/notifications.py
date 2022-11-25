import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterator, Optional, TypeVar, overload
import os.path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import threading
import sys
import socket
from tqdm.auto import tqdm as std_tqdm

from .utils import logger
from experimaestro.taskglobals import Env as TaskEnv

# --- Progress and other notifications

T = TypeVar("T")


@dataclass
class LevelInformation:
    level: int
    desc: Optional[str]
    progress: float

    previous_progress: float = -1
    previous_desc: Optional[str] = None

    def modified(self, reporter: "Reporter"):
        return (
            abs(self.progress - self.previous_progress) > reporter.progress_threshold
        ) or (self.previous_desc != self.desc)

    def report(self):
        self.previous_progress = self.progress
        result = {"level": self.level, "progress": self.progress}
        if self.previous_desc != self.desc:
            self.previous_desc = self.desc
            result["desc"] = self.desc
        return result

    def __repr__(self) -> str:
        return f"[{self.level}] {self.desc} {int(self.progress*1000)/10}%"


class Reporter(threading.Thread):
    NOTIFICATION_FOLDER = ".notifications"

    def __init__(self, path: Path):
        """Starts a notification thread

        Arguments:
            path: The path where notification URLs will be put (one file per URL)
        """
        super().__init__(daemon=True)
        self.path = path / Reporter.NOTIFICATION_FOLDER
        self.path.mkdir(exist_ok=True)
        self.urls: Dict[str, str] = {}

        # Last check of notification URLs
        self.lastcheck = 0

        self.levels = [LevelInformation(0, None, -1)]

        self.stopping = False

        self.progress_threshold = 0.01
        self.cv = threading.Condition()
        self.start()

    def stop(self):
        self.stopping = True
        with self.cv:
            self.cv.notifyAll()

    @staticmethod
    def isfatal_httperror(e: Exception) -> bool:
        """Returns True if this HTTP error indicates that the server won't recover"""
        if isinstance(e, HTTPError):
            if e.code >= 400 and e.code < 500:
                return True
        elif isinstance(e, URLError):
            if isinstance(e.reason, ConnectionRefusedError):
                return True
            if isinstance(e.reason, socket.gaierror) and e.reason.errno == -2:
                return True

        return False

    def modified(self):
        return any(level.modified(self) for level in self.levels)

    def check_urls(self):
        mtime = os.path.getmtime(self.path)
        if mtime > self.lastcheck:
            for f in self.path.iterdir():
                self.urls[f.name] = f.read_text().strip()
                logger.info("Added new notification URL: %s", self.urls[f.name])
                f.unlink()

            self.lastcheck = os.path.getmtime(self.path)

    def run(self):
        logger.info("Running notification thread")

        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.stopping or self.modified())
                if self.stopping:
                    break

            # Notify (out of the CV locking)
            toremove = []

            # Check if new notification servers are on
            self.check_urls()

            if self.urls:
                # OK, let's go
                for level in self.levels:
                    if level.modified(self):
                        params = level.report()

                        # Go over all URLs
                        for key, baseurl in self.urls.items():
                            url = "{}/progress?{}".format(
                                baseurl, urllib.parse.urlencode(params)
                            )
                            logger.debug("Reporting progress %s", params)
                            try:
                                with urlopen(url) as _:
                                    logger.debug(
                                        "Notification send for %s [%s]",
                                        baseurl,
                                        level,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "Progress: %s [error while notifying %s]: %s",
                                    level,
                                    url,
                                    e,
                                )
                                if Reporter.isfatal_httperror(e):
                                    toremove.append(key)

                # Removes unvalid URLs
                for key in toremove:
                    logger.info("Removing notification URL %s", self.urls[key])
                    del self.urls[key]
            else:
                for level in self.levels:
                    if level.modified(self):
                        params = level.report()
                        logger.info("Progress: %s", level)

    def eoj(self):
        with self.cv:
            self.check_urls()
            if self.urls:
                # Go over all URLs
                for key, baseurl in self.urls.items():
                    url = "{}?status=eoj".format(baseurl)
                    try:
                        with urlopen(url) as _:
                            logger.debug(
                                "EOJ botification sent for %s",
                                baseurl,
                            )
                    except Exception:
                        logger.warning(
                            "Could not report EOJ",
                        )

    def set_progress(self, progress: float, level: int, desc: Optional[str]):
        """Sets the new progress if sufficiently different"""
        with self.cv:
            if (
                (level + 1) != len(self.levels)
                or (progress != self.levels[level].progress)
                or (desc != self.levels[level].desc)
            ):
                self.levels = self.levels[: (level + 1)]
                while level >= len(self.levels):
                    self.levels.append(LevelInformation(level, None, 0.0))
                if desc:
                    self.levels[level].desc = desc
                self.levels[level].progress = progress

                self.cv.notify_all()

    INSTANCE: ClassVar[Optional["Reporter"]] = None

    @staticmethod
    def instance():
        if Reporter.INSTANCE is None:
            taskpath = TaskEnv.instance().taskpath
            assert taskpath is not None, "Task path is not defined"
            Reporter.INSTANCE = Reporter(taskpath)
        return Reporter.INSTANCE


def progress(value: float, level=0, desc: Optional[str] = None):
    """When called from a running task, report the progress

    Args:
        level: The level (starting from 0)
        value: The current value
        desc: An optional description of the current task
    """
    if TaskEnv.instance().slave:
        # Skip if in a slave process
        return
    Reporter.instance().set_progress(value, level, desc)


def report_eoj():
    """Notify that the job is over"""
    Reporter.instance().eoj()


class xpm_tqdm(std_tqdm):
    """XPM wrapper for experimaestro that automatically reports progress to the server"""

    def __init__(self, iterable=None, file=None, *args, **kwargs):
        # Report progress bar
        # newprogress(title=, pos=abs(self.pos))
        _file = file or sys.stderr
        self.is_tty = hasattr(_file, "isatty") or _file.isatty()

        super().__init__(iterable, disable=False, file=file, *args, **kwargs)
        progress(0.0, level=self.pos, desc=kwargs.get("desc", None))

    def update(self, n=1):
        result = super().update(n)
        if self.total is not None:
            progress(self.n / self.total, level=self.pos)
        return result

    def refresh(self, nolock=False, lock_args=None):
        if self.is_tty:
            super().refresh(nolock=nolock, lock_args=lock_args)


@overload
def tqdm(**kwargs) -> xpm_tqdm:
    ...


@overload
def tqdm(iterable: Optional[Iterator[T]] = None, **kwargs) -> Iterator[T]:
    ...


def tqdm(*args, **kwargs):
    return xpm_tqdm(*args, **kwargs)  # type: ignore
