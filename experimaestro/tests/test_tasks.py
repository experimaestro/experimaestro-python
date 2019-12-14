# --- Task and types definitions

import sys
import os
from pathlib import Path
import logging
import pytest
import subprocess
import psutil

from experimaestro import *
from experimaestro.scheduler import JobState
from experimaestro.click import cli

from .utils import TemporaryDirectory, TemporaryExperiment

from .tasks import *
from . import restart

def test_simple():
    with TemporaryDirectory(prefix="xpm", suffix="helloworld") as workdir:
        assert(isinstance(workdir, Path))
        with TemporaryExperiment("helloworld", workdir=workdir, maxwait=2):
            # Submit the tasks
            hello = Say(word="hello").submit()
            world = Say(word="world").submit()

            # Concat will depend on the two first tasks
            concat = Concat(strings=[hello, world]).submit()

        assert concat.__xpm__.job.state == JobState.DONE
        assert Path(concat.stdout()).read_text() == "HELLO WORLD\n"
    
def test_not_submitted():
    """A not submitted task should not be accepted as an argument"""
    with TemporaryExperiment("helloworld", maxwait=2):
        hello = Say(word="hello")
        with pytest.raises(ValueError):
            Concat(strings=[hello])

def test_fail():
    """Failing task... should fail"""
    with TemporaryExperiment("failing", maxwait=2):
        fail = Fail().submit()
        fail.touch()

    assert fail.__xpm__.job.wait() == JobState.ERROR


def test_fail_dep():
    """Failing task... should cancel dependent"""
    with TemporaryExperiment("failingdep"):
        fail = Fail().submit()
        dep = FailConsumer(fail=fail).submit()
        fail.touch()

    assert fail.__xpm__.job.wait() == JobState.ERROR
    assert dep.__xpm__.job.wait() == JobState.ERROR

def test_function():
    """Failing task... should cancel dependent"""
    with TemporaryExperiment("function"):
        method = Method(a=1).submit()

    assert method.__xpm__.job.wait() == JobState.DONE

@pytest.mark.skip()
def test_done():
    """Already done job"""
    pass



def test_restart():
    """Restarting the experiment should take back running tasks"""

    with TemporaryExperiment("restart", maxwait=10) as ws:
        # Create the task and so we can get the file paths
        task = restart.Restart()
        task.submit(dryrun=True)

        # Start the experiment with another process, and kill the job
        command = [sys.executable, restart.__file__, ws.path]
        logging.debug("Starting other process with: %s", command)
        p = subprocess.Popen(command)
        while not task.touch.is_file():
            time.sleep(0.1)
        logging.debug("Process has started [file %s]", task.touch)
        p.terminate()
        logging.debug("Process finishing with status %d", p.wait())

        # Check that task is still running
        pid = int(task.__xpm__.job.pidpath.read_text())
        logging.info("Checking that job (PID %s) is still running", pid)
        p = psutil.Process(pid)
        assert p.is_running()

        # Now, submit the job - it should pick up the process
        # where it was left
        logging.debug("Submitting the job")
        Scheduler.CURRENT.submit(task.__xpm__.job)
        with task.wait.open("w") as fp:
            fp.write("done")

        assert task.__xpm__.job.wait() == JobState.DONE
