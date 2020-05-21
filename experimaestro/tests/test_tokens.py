import sys
import pytest
import logging
import time
from pathlib import Path

import subprocess
from experimaestro.tokens import CounterToken
from experimaestro.scheduler import JobState
from .utils import TemporaryExperiment, TemporaryDirectory, timeout
from .task_tokens import TokenTask
from experimaestro.ipc import IPCom

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


def get_times(task):
    return TimeInterval(
        *(float(t) for t in task.stdout().read_text().strip().split("\n"))
    )


def get_times_frompath(path):
    s = path.read_text().strip().split("\n")
    logging.info("Read times: %s", s)
    return TimeInterval(*(float(t) for t in s))


def token_experiment(xp, token):
    path = xp.workspace.path / "test_token.file"
    task1 = TokenTask(path=path, x=1)
    task2 = TokenTask(path=path, x=2)
    for task in [task1, task2]:
        if token:
            task.add_dependencies(token.dependency(1))
        task.submit()

    # Wait that both tasks are scheduled
    for task in [task1, task2]:
        while task.job.state == JobState.UNSCHEDULED:
            time.sleep(0.01)

    # Waiting to ensure that both jobs are launched
    path.write_text("Hello world")

    xp.wait()

    time1 = get_times(task1)
    time2 = get_times(task2)

    assert time1 > time2 or time2 > time1


@pytest.mark.xfail(strict=True)
def test_token_fail():
    """Simple token test: should fail without token"""
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token_experiment(xp, token)


def test_token():
    """Simple token test: should succeed without token"""
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token = xp.workspace.connector.createtoken("test-token", 1)
        token_experiment(xp, token)


def test_token_monitor():
    """Two different schedulers (within the same process)
    
    Test the ability of the token to monitor the filesystem
    """

    def run(xp, x, path):
        token = xp.workspace.connector.createtoken("test-token-monitor", 1)
        task = TokenTask(path=path, x=x).add_dependencies(token.dependency(1)).submit()
        return task

    with TemporaryExperiment("tokens1", maxwait=10) as xp1, TemporaryExperiment(
        "tokens2", maxwait=10
    ) as xp2:
        path = xp1.workspace.path / "test_token.file"
        task1 = run(xp1, 1, path)
        task2 = run(xp2, 2, path)

        time.sleep(0.5)
        path.write_text("Hello world")

        xp1.wait()
        xp2.wait()
        time1 = get_times(task1)
        time2 = get_times(task2)

        logging.info("%s vs %s", time1, time2)
        assert time1 > time2 or time2 > time1


import multiprocessing



def test_token_reschedule():
    """Test whether a job can be re-submitted if it failed to acquire a token due to multiple schedulers concurrency

    - task 1 and 2 are started in two different processes, using the token
    - we wait for both to be scheduled
    - we write a file so that both can finish
    """
    queue1 = multiprocessing.Queue(3)
    queue2 = multiprocessing.Queue(3)

    with TemporaryDirectory("reschedule") as workdir:
        lockingpath = workdir / "lockingpath"

        command = [sys.executable, Path(__file__).parent / "token_reschedule.py", workdir]
        
        ready1 = workdir / "ready.1"
        time1 = workdir / "time.1"
        p1 = subprocess.Popen(command + ["1", lockingpath, str(ready1), str(time1)])

        ready2 = workdir / "ready.1"
        time2 = workdir / "time.2"
        p2 = subprocess.Popen(command + ["2", lockingpath, str(ready2), str(time2)])

        try:
            with timeout(10):
                # Wait that both processes are ready
                while not ready1.is_file():
                    time.sleep(0.01)
                while not ready2.is_file():
                    time.sleep(0.01)
                logging.info("Both processes are ready: allowing tasks to finish")
                lockingpath.write_text("Let's go")

                # Waiting for the output
                while not time1.is_file():
                    time.sleep(0.01)
                while not time2.is_file():
                    time.sleep(0.01)

                time1 = get_times_frompath(time1)
                time2 = get_times_frompath(time2)

                logging.info("%s vs %s", time1, time2)
                assert time1 > time2 or time2 > time1
        except TimeoutError as e:
            p1.terminate()
            p2.terminate()
            pytest.fail("Timeout")

        except Exception as e:
            logging.warning("Other exception: killing processes (just in case)")
            p1.terminate()
            p2.terminate()
            pytest.fail("Other exception")


@pytest.mark.skip("TODO: not implemented")
def test_token_process():
    """Test tokens with two processes"""
    raise NotImplementedError()
