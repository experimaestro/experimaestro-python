# --- Task and types definitions

import unittest
import os
from pathlib import Path
import logging

from experimaestro import *
from experimaestro.scheduler import JobState
from experimaestro.click import cli, TASK_PREFIX

from .utils import TemporaryExperiment

# --- Define the tasks

from .tasks import *

# --- Defines the experiment

class MainTest(unittest.TestCase):
    def test_fail(self):
        """A not submitted task should not be accepted as an argument"""
        with TemporaryExperiment("helloworld"):
            hello = Say(word="hello")
            self.assertRaises(ValueError, Concat, strings=[hello])


    def test_simple(self):
        with TemporaryExperiment("helloworld", maxwait=1):
            # Submit the tasks
            hello = Say(word="hello").submit()
            world = Say(word="world").submit()

            # Concat will depend on the two first tasks
            concat = Concat(strings=[hello, world]).submit()

        self.assertEqual(concat.__xpm__.job.state, JobState.DONE)
        self.assertEqual(Path(concat._stdout()).read_text(), "HELLO WORLD\n")
        

if __name__ == '__main__':
    import sys
    logging.warn(sys.path)
    unittest.main()

