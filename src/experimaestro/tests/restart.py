import os
import time
from pathlib import Path
import sys
from typing import Callable
from experimaestro import Task, Meta, field, PathGenerator
import psutil
import logging
import subprocess
import json
import signal

from experimaestro.scheduler.workspace import RunMode
from experimaestro.tests.utils import TemporaryExperiment, is_posix
from experimaestro.scheduler import JobState
from . import restart_main


def terminate(p):
    p.terminate()


def sigint(p):
    p.send_signal(signal.SIGINT)


TERMINATES_FUNC = [terminate]
if is_posix():
    TERMINATES_FUNC.append(sigint)

MAX_RESTART_WAIT = 50  # 5 seconds


class Restart(Task):
    touch: Meta[Path] = field(default_factory=PathGenerator("touch"))
    wait: Meta[Path] = field(default_factory=PathGenerator("wait"))

    def execute(self):
        # Write the file "touch" to notify that we started
        with open(self.touch, "w") as out:
            out.write("hello")

        # Wait for the file "wait" before exiting
        while not self.wait.is_file():
            time.sleep(0.1)


def restart(terminate: Callable, experiment):
    """Check if a new experimaestro process is able to take back
    a running job

    1. Runs an experiment and kills it using "terminate" while keeping the job active
    2. Runs the same experiment
        2.1 Submit the same job
        2.2 Asserts that the job is running
        2.3 Signals to the job to end
        2.4 Asserts that the job is done

    Args:
        terminate (Callable): How to terminate the process (SIGINT / terminate)
        experiment ([type]): [description]
    """
    p = None
    xpmprocess = None
    try:
        with TemporaryExperiment("restart", timeout_multiplier=9) as xp:
            # Create the task with dry_run and so we can get the file paths
            task = Restart.C()
            task.submit(run_mode=RunMode.DRY_RUN)

        # Start the experiment with another process, and kill the job
        command = [
            sys.executable,
            restart_main.__file__,
            xp.workspace.path,
            experiment.__module__,
            experiment.__name__,
        ]

        logging.debug("Starting other process with: %s", command)
        xpmprocess = subprocess.Popen(command)

        counter = 0
        while not task.touch.is_file():
            time.sleep(0.1)
            counter += 1
            if counter >= MAX_RESTART_WAIT:
                terminate(xpmprocess)
                assert False, "Timeout waiting for task to be executed"

        jobinfo = json.loads(task.__xpm__.job.pidpath.read_text())
        pid = int(jobinfo["pid"])
        p = psutil.Process(pid)

        # Now, kills experimaestro
        logging.debug("Process has started [file %s, pid %d]", task.touch, pid)
        terminate(xpmprocess)
        errorcode = xpmprocess.wait(5)
        logging.debug("Process finishing with status %d", errorcode)

        # Check that task is still running
        logging.info("Checking that job (PID %s) is still running", pid)
        assert p.is_running()

        with TemporaryExperiment("restart", timeout_multiplier=9) as xp:
            # Now, submit the job - it should pick up the process
            # where it was left
            logging.debug("Submitting the job (continues the submit)")
            job = task.__xpm__.job
            scheduler = xp.current().scheduler

            assert scheduler.submit(job) is None

            while scheduler.getJobState(job).result() == JobState.READY:
                time.sleep(0.1)

            currentState = scheduler.getJobState(job).result()
            assert currentState == JobState.RUNNING, (
                f"Job is not running (state is {currentState})"
            )

            # Notify the task
            with task.wait.open("w") as fp:
                fp.write("done")

            assert job.finalState().result() == JobState.DONE
    finally:
        # Force kill
        if xpmprocess and xpmprocess.poll() is None:
            logging.warning("Forcing to quit process %s", xpmprocess.pid)
            xpmprocess.kill()

        if p and p.is_running():
            logging.warning("Forcing to quit process %s", p.pid)
            p.terminate()


def ctrlc_leaves_jobs_running(experiment):
    """Check that Ctrl-C interrupts the driver but leaves jobs running.

    Pressing Ctrl-C in a terminal sends SIGINT to the whole *foreground
    process group* — the experiment driver and every local task it launched.
    The driver must stop promptly, but the submitted jobs must keep running
    (so they can be picked up again later).

    We reproduce a terminal Ctrl-C by launching the driver in its own session
    (so it is a process-group leader, and signalling its group does not hit
    the test runner) and sending SIGINT to that whole process group.
    """
    p = None
    xpmprocess = None
    task = None
    try:
        with TemporaryExperiment("ctrlc", timeout_multiplier=9) as xp:
            # Compute the task file paths without running it
            task = Restart.C()
            task.submit(run_mode=RunMode.DRY_RUN)

        command = [
            sys.executable,
            restart_main.__file__,
            xp.workspace.path,
            experiment.__module__,
            experiment.__name__,
        ]

        logging.debug("Starting driver process with: %s", command)
        # start_new_session: the driver becomes a session/process-group leader,
        # so os.killpg targets only the experiment (driver + its local tasks),
        # never the pytest process.
        xpmprocess = subprocess.Popen(command, start_new_session=True)
        pgid = os.getpgid(xpmprocess.pid)

        # Wait for the task to start
        counter = 0
        while not task.touch.is_file():
            time.sleep(0.1)
            counter += 1
            if xpmprocess.poll() is not None:
                assert False, (
                    f"Driver exited early (code {xpmprocess.returncode}) "
                    "before the task started"
                )
            if counter >= MAX_RESTART_WAIT:
                os.killpg(pgid, signal.SIGKILL)
                assert False, "Timeout waiting for task to be executed"

        jobinfo = json.loads(task.__xpm__.job.pidpath.read_text())
        pid = int(jobinfo["pid"])
        p = psutil.Process(pid)
        assert p.is_running()

        # Mimic a terminal Ctrl-C: SIGINT to the whole foreground process group
        logging.debug("Sending SIGINT to process group %d", pgid)
        os.killpg(pgid, signal.SIGINT)

        # The driver must stop promptly (regression: it used to hang)
        try:
            errorcode = xpmprocess.wait(10)
            logging.debug("Driver exited with status %d", errorcode)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            assert False, "Experiment driver did not exit after Ctrl-C (SIGINT)"

        # ... but the job must still be running
        assert p.is_running(), "Job was killed by Ctrl-C (it should keep running)"
    finally:
        # Tell the task to finish, then force-clean any leftovers
        try:
            if task is not None:
                with task.wait.open("w") as fp:
                    fp.write("done")
        except Exception:
            pass

        if p is not None and p.is_running():
            logging.warning("Forcing to quit job process %s", p.pid)
            p.terminate()

        if xpmprocess and xpmprocess.poll() is None:
            logging.warning("Forcing to quit driver process %s", xpmprocess.pid)
            xpmprocess.kill()
