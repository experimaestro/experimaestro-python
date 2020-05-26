# --- Task and types definitions

import sys
import os
from pathlib import Path
import logging
import pytest
import subprocess
import signal
import psutil

from experimaestro import *
from experimaestro.scheduler import JobState
from experimaestro.click import cli

from .utils import TemporaryDirectory, TemporaryExperiment, is_posix

from .tasks import *
from . import restart


def test_simple_task():
    with TemporaryDirectory(prefix="xpm", suffix="helloworld") as workdir:
        assert isinstance(workdir, Path)
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


def test_foreign_type():
    """When the argument real type is in an non imported module"""
    with TemporaryExperiment("foreign_type", maxwait=2):
        # Submit the tasks
        from .tasks2 import ForeignClassB2

        b = ForeignClassB2(x=1, y=2)
        a = ForeignTaskA(b=b).submit()

        assert a.__xpm__.job.wait() == JobState.DONE
        assert a.stdout().read_text().strip() == "1"


def test_fail_dep():
    """Failing task... should cancel dependent"""
    with TemporaryExperiment("failingdep"):
        fail = Fail().submit()
        dep = FailConsumer(fail=fail).submit()
        fail.touch()

    assert fail.__xpm__.job.wait() == JobState.ERROR
    assert dep.__xpm__.job.wait() == JobState.ERROR


def test_unknown_attribute():
    """No check when setting attributes while executing"""
    with TemporaryExperiment("unknown"):
        method = SetUnknown().submit()

    assert method.__xpm__.job.wait() == JobState.DONE


def test_function():
    with TemporaryExperiment("function"):
        method = Method(a=1).submit()

    assert method.__xpm__.job.wait() == JobState.DONE


@pytest.mark.skip()
def test_done():
    """Checks that we do not run an already done job"""
    pass


def terminate(p):
    p.terminate()


def sigint(p):
    p.send_signal(signal.SIGINT)


TERMINATES_FUNC = [terminate]
if is_posix():
    TERMINATES_FUNC.append(sigint)


@pytest.mark.parametrize("terminate", TERMINATES_FUNC)
def test_restart(terminate):
    """Restarting the experiment should take back running tasks"""
    p = None
    xpmprocess = None
    try:
        with TemporaryExperiment("restart", maxwait=10) as xp:
            # Create the task and so we can get the file paths
            task = restart.Restart()
            task.submit(dryrun=True)

            # Start the experiment with another process, and kill the job
            command = [sys.executable, restart.__file__, xp.workspace.path]
            logging.debug("Starting other process with: %s", command)
            xpmprocess = subprocess.Popen(command)
            while not task.touch.is_file():
                time.sleep(0.1)

            pid = int(task.__xpm__.job.pidpath.read_text())
            p = psutil.Process(pid)

            logging.debug("Process has started [file %s, pid %d]", task.touch, pid)
            terminate(xpmprocess)
            errorcode = xpmprocess.wait(5)
            logging.debug("Process finishing with status %d", errorcode)

            # Check that task is still running
            logging.info("Checking that job (PID %s) is still running", pid)
            assert p.is_running()

            # Now, submit the job - it should pick up the process
            # where it was left
            logging.debug("Submitting the job")
            Scheduler.CURRENT.submit(task.__xpm__.job)
            with task.wait.open("w") as fp:
                fp.write("done")

            assert task.__xpm__.job.wait() == JobState.DONE
    finally:
        # Force kill
        if xpmprocess and xpmprocess.poll() is None:
            logging.warning("Forcing to quit process %s", xpmprocess.pid)
            xpmprocess.kill()

        if p and p.is_running():
            logging.warning("Forcing to quit process %s", p.pid)
            p.terminate()


@pytest.mark.skip("Test to implemented")
def test_submitted_twice():
    """Check that a job cannot be submitted twice"""
    pass


def test_configcache():
    """Test a configuration cache"""

    with TemporaryExperiment("configcache", maxwait=10) as xp:
        task = CacheConfigTask(data=CacheConfig()).submit()

    assert task.__xpm__.job.wait() == JobState.DONE

