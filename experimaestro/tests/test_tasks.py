# --- Task and types definitions

import os
from pathlib import Path
import logging
import pytest

from experimaestro import *
from experimaestro.scheduler import JobState
from experimaestro.click import cli

from .utils import TemporaryDirectory, TemporaryExperiment

from .tasks import *

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
def test_restart():
    """Restarting the experiment should take back running tasks"""
    pass

@pytest.mark.skip()
def test_done():
    """Already done job"""
    pass
