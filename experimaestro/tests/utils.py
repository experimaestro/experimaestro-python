import tempfile
import shutil
import os
from pathlib import Path
import logging
from experimaestro import Launcher, experiment

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

class TemporaryExperiment:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.workdir = TemporaryDirectory(prefix="xpm", suffix=self.name)
        workdir = self.workdir.__enter__()

        self.experiment = experiment(workdir, self.name)
        workspace = self.experiment.__enter__()

        # Set some useful environment variables
        workspace.launcher.setenv("LD_LIBRARY_PATH", os.getenv("LD_LIBRARY_PATH"))

        return workspace

    def __exit__(self, *args):
        self.experiment.__exit__(*args)
        self.workdir.__exit__(*args)

