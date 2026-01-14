import sys
import json
import filelock
import pytest
import logging
import time
from pathlib import Path

import subprocess
from experimaestro import Task, Param
from experimaestro.tokens import CounterToken, TokenLockFile
from experimaestro.scheduler import JobState
from .utils import (
    TemporaryExperiment,
    TemporaryDirectory,
    timeout,
    get_times,
    get_times_frompath,
)
from .task_tokens import TokenTask
from . import restart


def token_experiment(xp, token, ntasks=3):
    """Starts two tasks with a token

    Waits that the two tasks are scheduled. Each task finishes
    not before the tasks are scheduled (through the use
    of a file to be created)
    """
    path = xp.workspace.path / "test_token.file"

    tasks = []
    for it in range(ntasks):
        task = TokenTask.C(path=path, x=it)
        if token:
            task.add_dependencies(token.dependency(1))
        tasks.append(task.submit())

    # Wait that both tasks are scheduled
    logging.info("Waiting that the two tasks are scheduled")
    for task in tasks:
        while task.__xpm__.job.state == JobState.UNSCHEDULED:
            time.sleep(0.01)

    # Wait a bit (ENHANCE: find a better way)
    time.sleep(1)

    # Now any task can finish
    logging.info("%d tasks scheduled - waiting for completion", ntasks)
    path.write_text("Hello world")

    xp.wait()

    times = [get_times(task) for task in tasks]
    logging.info("Times: %s", ",".join([str(t) for t in times]))
    times = sorted(times)
    for i in range(1, ntasks):
        assert (times[i - 1] > times[i]) or (times[i] > times[i - 1])


@pytest.mark.xfail(
    strict=False,
    reason="Timing-dependent: tasks may run sequentially even without token",
)
def test_token_fail():
    """Simple token test: should fail without token (but may pass due to timing)"""
    with TemporaryExperiment("tokens") as xp:
        token_experiment(xp, None)


def test_token_ok():
    """Simple token test: should succeed with token"""
    with TemporaryExperiment("tokens") as xp:
        token = CounterToken("token-ok", xp.workdir / "token", 1)
        token_experiment(xp, token)

    logging.info("Finished token_ok test")


class dummy_task(Task):
    x: Param[int]

    def execute(self):
        pass


def test_token_cleanup():
    """Test that tokens are correctly cleaned up if the process finished"""
    with TemporaryExperiment("token_cleanup") as xp:
        token = CounterToken("token-cleanup", xp.workdir / "token-cleanup", 1)

        task = dummy_task.C(x=1)
        dependency = token.dependency(1)
        task.add_dependencies(dependency)
        # Just to create the directory
        task.submit()

        xp.wait()

        # Just lock directly (but without process)
        # The absence of process should be detected right away
        logging.info("Lock without process")
        TokenLockFile.from_dependency(dependency)
        task2 = dummy_task.C(x=2)
        task2.add_dependencies(token.dependency(1)).submit()
        xp.wait()

        # Just lock directly (with process)
        logging.info("Lock with process")
        job = dependency.target
        with filelock.FileLock(job.lockpath):
            logging.info("Creating dependency %s", dependency)
            TokenLockFile.from_dependency(dependency)
            lockingpath = job.path / "testtoken.signal"
            command = [
                sys.executable,
                Path(__file__).parent / "scripts" / "waitforfile.py",
                lockingpath,
            ]

            p1 = subprocess.Popen(command)
            job.pidpath.write_text(json.dumps({"pid": p1.pid, "type": "local"}))

            task3 = dummy_task.C(x=3)
            task3.add_dependencies(token.dependency(1)).submit()

            # Ends the script "waitforfile.py"
            lockingpath.write_text("Let's go")

            # task3 should start
            xp.wait()


def test_token_monitor():
    """Two different experiments (within the same process and workspace)

    Test the ability of the token to monitor the filesystem
    """

    def run(xp, x, path, token):
        task = (
            TokenTask.C(path=path, x=x).add_dependencies(token.dependency(1)).submit()
        )
        return task

    with TemporaryExperiment("tokens1") as xp1:
        # Use the same workspace for both experiments
        with TemporaryExperiment("tokens2", workdir=xp1.workspace.path) as xp2:
            path = xp1.workspace.path / "test_token.file"
            # Shared token in workdir (not global token store)
            token = CounterToken(
                "test-token-monitor", xp1.workspace.path / "token-monitor", 1
            )
            task1 = run(xp1, 1, path, token)
            task2 = run(xp2, 2, path, token)

            time.sleep(0.5)
            path.write_text("Hello world")

        time1 = get_times(task1)
        time2 = get_times(task2)

        logging.info("%s vs %s", time1, time2)
        assert time1 > time2 or time2 > time1


def test_token_reschedule():
    """Test whether a job can be re-submitted if it failed to acquire a token
    due to multiple schedulers concurrency

    - task 1 and 2 are started in two different processes, using the token
    - we wait for both to be scheduled
    - we write a file so that both can finish
    """
    # queue1 = multiprocessing.Queue(3)
    # queue2 = multiprocessing.Queue(3)

    with TemporaryDirectory("reschedule") as workdir:
        lockingpath = workdir / "lockingpath"

        command = [
            sys.executable,
            Path(__file__).parent / "token_reschedule.py",
            workdir,
        ]

        ready1 = workdir / "ready.1"
        time1 = workdir / "time.1"
        p1 = subprocess.Popen(command + ["1", lockingpath, str(ready1), str(time1)])

        ready2 = workdir / "ready.2"
        time2 = workdir / "time.2"
        p2 = subprocess.Popen(command + ["2", lockingpath, str(ready2), str(time2)])

        try:
            with timeout():
                logging.info("Waiting for both experiments to be ready")
                # Wait that both processes are ready
                while not ready1.is_file():
                    time.sleep(0.01)
                while not ready2.is_file():
                    time.sleep(0.01)

                # Create the locking path
                logging.info(
                    "Both processes are ready:allowing tasks to finish by writing in %s",
                    lockingpath,
                )
                lockingpath.write_text("Let's go")

                # Waiting for the output
                logging.info("Waiting for XP1 to finish (%s)", time1)
                while not time1.is_file():
                    time.sleep(0.01)
                logging.info("Experiment 1 finished")

                logging.info("Waiting for XP2 to finish")
                while not time2.is_file():
                    time.sleep(0.01)
                logging.info("Experiment 2 finished")

                logging.info("Both processes are ready: allowing tasks to finish")

                time1 = get_times_frompath(time1)
                time2 = get_times_frompath(time2)

                logging.info("%s vs %s", time1, time2)
                assert time1 > time2 or time2 > time1
        except TimeoutError:
            p1.terminate()
            p2.terminate()
            pytest.fail("Timeout")

        except Exception:
            logging.warning("Other exception: killing processes (just in case)")
            p1.terminate()
            p2.terminate()
            pytest.fail("Other exception")


@pytest.mark.skip("ENHANCE: not implemented")
def test_token_process():
    """Test tokens with two processes"""
    raise NotImplementedError()


def restart_function(xp):
    token = CounterToken("restart-token", xp.workdir / "token", 1)
    token(1, restart.Restart.C()).submit()


@pytest.mark.parametrize("terminate", restart.TERMINATES_FUNC)
def test_token_restart(terminate):
    """Restarting the experiment should take back running tasks"""
    restart.restart(terminate, restart_function)
