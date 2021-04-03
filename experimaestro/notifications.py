import os
from urllib.request import urlopen
import threading
import sys
from tqdm.auto import tqdm as std_tqdm

from .scheduler import NOTIFICATIONURL_VARNAME
from .utils import logger

# --- Progress and other notifications


class NotificationThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.url = os.environ.get(NOTIFICATIONURL_VARNAME, None)

        self.progress = 0
        self.previous_progress = -1

        self.stopping = False

        if self.url:
            self.progress_threshold = 0.01
            self.cv = threading.Condition()
            self.start()
        else:
            self.cv = None
            self.progress_threshold = 0.05

    def stop(self):
        if self.url:
            self.stopping = True
            with self.cv:
                self.cv.notifyAll()

    def run(self):
        logger.info("Running notification thread")

        while not self.stopping:
            with self.cv:
                self.cv.wait_for(
                    lambda: self.stopping
                    or self.progress - self.previous_progress > self.progress_threshold
                )
                if not self.is_alive():
                    break

                reportprogress = (
                    self.progress - self.previous_progress > self.progress_threshold
                )

            if reportprogress:
                self.previous_progress = self.progress
                url = "{}/progress/{}".format(self.url, self.progress)
                try:
                    with urlopen(url) as response:
                        pass
                except Exception:
                    logger.info(
                        "Progress: %.2f [error while notifying %s]", self.progress, url
                    )

    def setprogress(self, progress):
        if progress - self.previous_progress > self.progress_threshold:
            if self.url:
                with self.cv:
                    self.progress = progress
                    self.cv.notify_all()
            else:
                self.progress = progress
                self.previous_progress = progress
                logger.info("Progress: %.2f", self.progress)


INSTANCE = NotificationThread()


def progress(value: float):
    INSTANCE.setprogress(value)


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
            progress(d["n"] / d["total"])


tqdm = xpm_tqdm
