from pathlib import Path
import sys
import time
from experimaestro.connectors import Process, Redirect
from experimaestro.launchers import Launcher
import logging


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

    restored = Process.fromDefinition(launcher, spec)
    logging.info("Waiting for job to end")
    semaphore.touch()
    restored.wait()
