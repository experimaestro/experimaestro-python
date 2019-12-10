import tempfile
import shutil
import os
from pathlib import Path
import logging
import signal

from experimaestro.launchers import Launcher
from experimaestro import experiment

class TemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix, dir=self.dir))
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if os.environ.get("XPM_KEEPWORKDIR", False):
            logging.warning("NOT Removing %s" %  self.path)
        else:
            shutil.rmtree(self.path)


class TimeoutError(Exception): pass 

class timeout:
  def __init__(self, seconds, error_message=None):
    if error_message is None:
      error_message = 'test timed out after {}s.'.format(seconds)
    self.seconds = seconds
    self.error_message = error_message

  def handle_timeout(self, signum, frame):
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
        workspace = self.experiment.__enter__()

        # Set some useful environment variables
        workspace.launcher.setenv("PYTHONPATH", str(Path(__file__).parents[2]))
        self.timeout.__enter__()

        return workspace

    def __exit__(self, *args):
        self.experiment.__exit__(*args)
        self.timeout.__exit__(*args)
        if self.clean_workdir:
          self.workdir.__exit__(*args)


logging.basicConfig(level=logging.DEBUG)