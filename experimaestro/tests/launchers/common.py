from pathlib import Path
import sys
import time
import logging
import asyncio
from experimaestro.scheduler import JobState
from experimaestro.connectors import Process, Redirect
from experimaestro.launchers import Launcher
from experimaestro import Task, Param

logger = logging.getLogger("xpm.tests.launchers")


def waitFromSpec(tmp_path: Path, launcher: Launcher):
    builder = launcher.processbuilder()

    started = tmp_path / "started"
    semaphore = tmp_path / "semaphore"
    scriptfile = Path(__file__).parent / ".." / "scripts" / "notifyandwait.py"
    assert scriptfile.exists()

    builder.command = [
        sys.executable,
        scriptfile,
        started,
        semaphore,
    ]
    builder.detach = True
    builder.stdout = Redirect.file(tmp_path / "stdout")
    builder.stderr = Redirect.file(tmp_path / "stderr")

    logging.info("Starting job")
    process = builder.start()  # type: BatchSlurmProcess
    spec = process.tospec()
    logging.info("Got job ID %s", spec["pid"])

    while not started.exists():
        time.sleep(0.1)

    logging.info("Job started")

    restored = Process.fromDefinition(launcher.connector, spec)

    assert asyncio.run(restored.aio_isrunning()), "Process is not running"

    logging.info("Waiting for job to end")
    semaphore.touch()
    restored.wait()

    # Should not be running anymore
    assert not asyncio.run(restored.aio_isrunning()), "Process is running"

    return restored


# --- Take back code


class WaitUntilTouched(Task):
    touching: Param[Path]
    waiting: Param[Path]

    def execute(self):
        self.touching.touch()

        while not self.waiting.exists():
            time.sleep(0.01)


def takeback(launcher, datapath, txp1, txp2):
    datapath.mkdir()
    touching = datapath / "touching"
    waiting = datapath / "waiting"

    with txp1 as xp1:  # flake8: noqa: F841
        WaitUntilTouched(touching=touching, waiting=waiting).submit(launcher=launcher)

        logger.debug("Waiting for task to create 'touching' file")
        while not touching.is_file():
            time.sleep(0.01)

        with txp2 as xp2:  # flake8: noqa: F841
            result = WaitUntilTouched(touching=touching, waiting=waiting).submit(
                launcher=launcher
            )

            logger.debug("Waiting for job to be in running mode")
            while result.__xpm__.job.state != JobState.RUNNING:
                time.sleep(0.1)

            waiting.touch()
