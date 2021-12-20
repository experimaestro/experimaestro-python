# Test that progress notification work
from pathlib import Path
import time
import threading
from experimaestro import Task, Annotated, pathgenerator, progress
from experimaestro.core.objects import TaskOutput, logger
from experimaestro.scheduler import Job, JobState, Listener
from queue import Queue
from .utils import TemporaryDirectory, TemporaryExperiment, get_times


class ProgressingTask(Task):
    path: Annotated[Path, pathgenerator("progress.txt")]

    def execute(self):
        _progress = 0.0

        while _progress < 1:
            time.sleep(1e-4)
            if self.path.exists():
                try:
                    _progress = float(self.path.read_text())
                    progress(_progress)
                except:
                    pass


class ProgressListener(Listener):
    def __init__(self):
        self.current = 0.0
        self.progresses = Queue()

    def job_state(self, job: Job):
        if self.current != job.progress:
            logger.info("Got some progress: %f", job.progress)
            self.current = job.progress
            self.progresses.put(job.progress)

        if job.state.finished():
            self.progresses.put(-1 if job.state == JobState.DONE else -2)


def test_progress_basic():
    """Test that when submitting the task, the computed idenfitier is the one of the new class"""
    with TemporaryExperiment("progress-basic", maxwait=5, port=0) as xp:
        assert xp.server is not None
        assert xp.server.port > 0

        listener = ProgressListener()
        xp.scheduler.addlistener(listener)

        out = ProgressingTask().submit()  # type: TaskOutput
        path = out.path  # type: Path
        job = out.__xpm__.job

        logger.info("Waiting for job to start")
        while job.state.notstarted():
            time.sleep(1e-2)

        logger.info("Checking job progress")
        progresses = [(i + 1.0) / 10.0 for i in range(10)]
        for v in progresses:
            path.write_text(f"{v}")
            if v < 1:
                assert listener.progresses.get() == v


def test_progress_multiple():
    """Test that even with two schedulers, we get notified"""
    with TemporaryExperiment("progress-progress-multiple-1", maxwait=5, port=0) as xp1:
        assert xp1.server is not None
        assert xp1.server.port > 0

        listener1 = ProgressListener()
        xp1.scheduler.addlistener(listener1)

        out = ProgressingTask().submit()  # type: TaskOutput
        path = out.path  # type: Path
        job = out.__xpm__.job

        logger.info("Waiting for job to start (1)")
        while job.state.notstarted():
            time.sleep(1e-2)

        with TemporaryExperiment(
            "progress-progress-multiple-2", workdir=xp1.workdir, maxwait=5, port=0
        ) as xp2:
            assert xp2.server is not None
            assert xp2.server.port > 0
            listener2 = ProgressListener()
            xp2.scheduler.addlistener(listener2)

            out = ProgressingTask().submit()
            job = out.__xpm__.job
            logger.info("Waiting for job to start (2)")
            while job.state.notstarted():
                time.sleep(1e-2)

            # Both schedulers should receive the job progress information
            logger.info("Checking job progress")
            progresses = [(i + 1.0) / 10.0 for i in range(10)]
            for v in progresses:
                path.write_text(f"{v}")
                if v < 1:
                    assert listener1.progresses.get() == v
                    assert listener2.progresses.get() == v
