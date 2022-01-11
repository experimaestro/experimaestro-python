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

# --- Progress and other notifications

T = TypeVar("T")


class NotificationThread(threading.Thread):
    NOTIFICATION_FOLDER = ".notifications"

    def __init__(self, path: Path):
        """Starts a notification thread

        Arguments:
            path: The path where notification URLs will be put (one file per URL)
        """
        super().__init__(daemon=True)
        self.path = path / NotificationThread.NOTIFICATION_FOLDER
        self.path.mkdir(exist_ok=True)
        self.lastcheck = 0
        self.urls: Dict[str, str] = {}
        self.progress = 0
        self.previous_progress = -1

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
        if isinstance(e, HTTPError):
            if e.code >= 400 and e.code < 500:
                return True
        elif isinstance(e, URLError):
            if isinstance(e.reason, ConnectionRefusedError):
                return True
            if isinstance(e.reason, socket.gaierror) and e.reason.errno == -2:
                return True

        return False

    def run(self):
        logger.info("Running notification thread")

        while not self.stopping:
            with self.cv:
                self.cv.wait_for(
                    lambda: self.stopping
                    or abs(self.progress - self.previous_progress)
                    > self.progress_threshold
                )
                if not self.is_alive():
                    break

                reportprogress = (
                    abs(self.progress - self.previous_progress)
                    > self.progress_threshold
                )

            if reportprogress:
                self.previous_progress = self.progress
                toremove = []

                # Check if new notification servers are on
                mtime = os.path.getmtime(self.path)
                if mtime > self.lastcheck:
                    for f in self.path.iterdir():
                        self.urls[f.name] = f.read_text().strip()
                        logger.info("Added new notification URL: %s", self.urls[f.name])
                        f.unlink()

                    self.lastcheck = os.path.getmtime(self.path)

                if self.urls:
                    # OK, let's go
                    for key, baseurl in self.urls.items():
                        url = "{}/progress/{}".format(baseurl, self.progress)
                        try:
                            with urlopen(url) as _:
                                logger.info(
                                    "Notification send for %s [%s]",
                                    baseurl,
                                    self.progress,
                                )
                        except Exception as e:
                            logger.info(
                                "Progress: %.2f [error while notifying %s]: %s",
                                self.progress,
                                url,
                                e,
                            )
                            if NotificationThread.isfatal_httperror(e):
                                toremove.append(key)

                    # Removes unvalid URLs
                    for key in toremove:
                        logger.info("Removing notification URL %s", self.urls[key])
                        del self.urls[key]
                else:
                    logger.info("Progress: %.2f", self.progress)

    def setprogress(self, progress):
        """Sets the new progress if sufficiently different"""
        if abs(progress - self.previous_progress) > self.progress_threshold:
            with self.cv:
                self.progress = progress
                self.cv.notify_all()

    INSTANCE: ClassVar[Optional["NotificationThread"]] = None

    @staticmethod
    def instance():
        if NotificationThread.INSTANCE is None:
            from experimaestro.taskglobals import taskpath

            assert taskpath is not None, "Task path is not defined"
            NotificationThread.INSTANCE = NotificationThread(taskpath)
        return NotificationThread.INSTANCE


def progress(value: float):
    """When called from a running task, report the progress"""

    NotificationThread.instance().setprogress(value)


class xpm_tqdm(std_tqdm):
    """XPM wrapper for experimaestro that automatically reports progress to the server"""

    def __init__(self, iterable=None, file=None, *args, **kwargs):
        # Report progress bar
        # newprogress(title=, pos=abs(self.pos))
        _file = file or sys.stderr
        self.is_tty = hasattr(_file, "isatty") or _file.isatty()
        super().__init__(iterable, *args, file=file, **kwargs)

    def refresh(self, nolock=False, lock_args=None):
        if self.is_tty:
            super().refresh()

        pos = abs(self.pos)
        if pos == 0:
            d = self.format_dict
            # Just report the innermost progress
            if d["total"]:
                progress(d["n"] / d["total"])


@overload
def tqdm(**kwargs) -> xpm_tqdm:
    ...


@overload
def tqdm(iterable: Optional[Iterator[T]] = None, **kwargs) -> Iterator[T]:
    ...


def tqdm(*args, **kwargs):
    return xpm_tqdm(*args, **kwargs)  # type: ignore
