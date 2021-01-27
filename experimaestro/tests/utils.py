from contextlib import contextmanager
import tempfile
import shutil
import os
from pathlib import Path
import logging
import signal
import pytest

from experimaestro.launchers import Launcher
from experimaestro import experiment, task


class TimeInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, value):
        return self.start > value.end

    def __str__(self):
        return "%.4f - %.4f" % (self.start, self.end)

    def __repr__(self):
        return str(self)


def get_times(task: task) -> TimeInterval:
    logging.info("Reading times from %s", task.stdout())
    return TimeInterval(
        *(float(t) for t in task.stdout().read_text().strip().split("\n"))
    )


def get_times_frompath(path) -> TimeInterval:
    s = path.read_text().strip().split("\n")
    logging.info("Read times: %s", s)
    return TimeInterval(*(float(t) for t in s))


class TemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self):
        self.path = Path(
            tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        )
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if os.environ.get("XPM_KEEPWORKDIR", False) == "1":
            logging.warning("NOT Removing %s" % self.path)
        else:
            logging.warning("CLLLLLEAEANING UP %s", self.path)
            shutil.rmtree(self.path, ignore_errors=True)


class timeout:
    def __init__(self, seconds, error_message=None):
        if error_message is None:
            error_message = "test timed out after {}s.".format(seconds)
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        logging.error("Timeout - sending signal")
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


class TemporaryExperiment:
    def __init__(self, name, workdir=None, maxwait=10):
        self.name = name
        self.workdir = workdir
        self.clean_workdir = workdir is None
        self.timeout = timeout(maxwait)

    def __enter__(self):
        if self.clean_workdir:
            self.workdir = TemporaryDirectory(prefix="xpm", suffix=self.name)
            workdir = self.workdir.__enter__()
        else:
            workdir = self.workdir

        self.experiment = experiment(workdir, self.name)
        self.experiment.__enter__()

        # Set some useful environment variables
        self.experiment.workspace.launcher.setenv(
            "PYTHONPATH", str(Path(__file__).parents[2])
        )
        self.timeout.__enter__()

        logging.info("Created new temporary experiment (%s)", workdir)
        return self.experiment

    def __exit__(self, *args):
        self.experiment.__exit__(*args)
        self.timeout.__exit__(*args)
        if self.clean_workdir:
            self.workdir.__exit__(*args)


def is_posix():
    try:
        import posix

        return True
    except ImportError:
        return False
