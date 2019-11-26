import tempfile
import shutil
import os
from experimaestro import Launcher, experiment
import logging

class TemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if os.environ.get("XPM_KEEPWORKDIR", False):
            logging.warning("NOT Removing %s" %  self.path)
        else:
            shutil.rmtree(self.path)

class Experiment:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.workdir = TemporaryDirectory(prefix="xpm", suffix=self.name)
        workdir = self.workdir.__enter__()

        print(workdir)
        # Set some useful environment variables
        Launcher.DEFAULT.setenv("LD_LIBRARY_PATH", os.getenv("LD_LIBRARY_PATH"))

        # Sets the working workdir and the name of the xp
        experiment(workdir, self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        self.workdir.__exit__(exc_type, exc_value, traceback)

