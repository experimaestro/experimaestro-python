# --- Task and types definitions

import unittest
import os
from pathlib import Path
import logging

from experimaestro.click import cli, TASK_PREFIX
from experimaestro import Workspace, JOB_DONE

from .utils import Experiment
from .definitions_types import *

# --- Defines the experiment

class MainTest(unittest.TestCase):
    def test_simple(self):
        with Experiment("simple") as xp:

            self.assertEqual(TestInteger(value=5).submit()._job.wait(), JOB_DONE, "test integer failed")
            self.assertEqual(TestFloat(value=5.1).submit()._job.wait(), JOB_DONE, "test float failed")

if __name__ == '__main__':
    import sys
    logging.warn(sys.path)
    unittest.main()

