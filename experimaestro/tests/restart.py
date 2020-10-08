import time
import sys
from experimaestro import task, pathoption, Scheduler
import psutil
import logging
import subprocess
from experimaestro.tests.utils import TemporaryExperiment
from experimaestro.scheduler import JobState
from . import restart_main

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


@pathoption("touch", "touch")
@pathoption("wait", "wait")
@task("restart")
class Restart:
    def execute(self):
        # Write the file "touch" to notify that we started
        with open(self.touch, "w") as out:
            out.write("hello")

        # Wait for the file "wait" before exiting
        while not self.wait.is_file():
            time.sleep(0.1)


def restart(terminate, experiment):
    p = None
    xpmprocess = None
    try:
        with TemporaryExperiment("restart", maxwait=10) as xp:
            # Create the task and so we can get the file paths
            task = Restart()
            task.submit(dryrun=True)

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

        with TemporaryExperiment("restart", maxwait=10) as xp:
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
