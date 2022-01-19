# Test that progress notification work
from copy import copy
import logging
from pathlib import Path
import time
from typing import Dict, List, OrderedDict, Tuple, Union
from experimaestro import Task, Annotated, pathgenerator, progress, tqdm
from experimaestro.core.arguments import Param
from experimaestro.core.objects import TaskOutput, logger
from experimaestro.notifications import LevelInformation
from experimaestro.scheduler import Job, Listener
from queue import Queue
from .utils import TemporaryDirectory, TemporaryExperiment, get_times


class ProgressingTask(Task):
    path: Annotated[Path, pathgenerator("progress.txt")]

    def execute(self):
        _progress = 0.0

        while True:
            time.sleep(1e-4)
            if self.path.exists():
                try:
                    _level, _progress, _desc = self.path.read_text().split(
                        " ", maxsplit=2
                    )
                    _progress = float(_progress)
                    _level = int(_level)
                    progress(_progress, level=_level, desc=_desc or None)
                    if _progress == 1.0 and _level == 0:
                        break
                except:
                    pass


def writeprogress(path: Path, progress, level=0, desc=None):
    path.write_text(f"{level} {progress:.3f} {desc if desc else ''}")


class ProgressListener(Listener):
    def __init__(self):
        self.current = []
        self.progresses: Queue[List[LevelInformation]] = Queue()

    def job_state(self, job: Job):
        if (len(self.current) != len(job.progress)) or any(
            l1 != l2 for l1, l2 in zip(self.current, job.progress)
        ):
            logger.info("Got some progress: %s", job.progress)
            self.current = [copy(level) for level in job.progress]
            self.progresses.put(self.current)

        # if job.state.finished():
        #     self.progresses.put(-1 if job.state == JobState.DONE else -2)


def test_progress_basic():
    """Test that we get all the progress reports"""
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
            writeprogress(path, v)
            if v < 1:
                l = listener.progresses.get()[0]
                logging.info("Got %s", l)
                assert l.progress == v


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
                writeprogress(path, v)
                if v < 1:
                    assert listener1.progresses.get()[0].progress == v
                    assert listener2.progresses.get()[0].progress == v


NestedTasks = Tuple[str, Union[int, List["NestedTasks"]]]


class NestedProgressingTask(Task):
    PROGRESSES: NestedTasks = (
        "Task 1",
        [
            ("Task 1.1", 2),
            ("Task 1.2", [("Task 1.2.1", 3), ("Task 1.2.2", 4)]),
            ("Task 1.3", 1),
        ],
    )

    path: Annotated[Path, pathgenerator("progress.txt")]

    def execute(self):
        self._execute(NestedProgressingTask.PROGRESSES)

    def wait(self):
        while not self.path.exists():
            time.sleep(1e-4)
        self.path.unlink()

    def _execute(self, tasks: NestedTasks):
        self.wait()
        name, subtasks = tasks
        if isinstance(subtasks, list):
            for subtasks in tqdm(subtasks, desc=name, miniters=1, mininterval=0):
                self._execute(subtasks)
            self.wait()
        else:
            for _ in tqdm(range(subtasks), desc=name, miniters=1, mininterval=0):
                self.wait()
            self.wait()


def check_nested(
    path: Path, listener: ProgressListener, tasks: NestedTasks, level_ix=0
):
    name, subtasks = tasks

    def check_self(progress: float, nlevel: int = None):
        path.touch()
        levels = listener.progresses.get()
        logger.info(
            "Got %s (checking %d, %f, %d)",
            levels,
            level_ix,
            progress,
            nlevel or (level_ix + 1),
        )
        assert len(levels) == nlevel or (level_ix + 1)
        level = levels[level_ix]
        assert level.desc == name
        assert level.progress == progress

    if isinstance(subtasks, list):
        for ix, subsubtasks in enumerate(subtasks):
            logger.info(
                "Checking nested for %s (%d, %d/%d)", name, level_ix, ix, len(subtasks)
            )
            check_self(ix / len(subtasks), level_ix + 2)
            check_nested(path, listener, subsubtasks, level_ix + 1)
        check_self(1, level_ix + 2)
    else:
        for ix in range(subtasks):
            check_self(ix / float(subtasks))
        check_self(1)


def test_progress_nested():
    """Test that we get all the progress reports"""
    with TemporaryExperiment("progress-nested", maxwait=10, port=0) as xp:
        assert xp.server is not None
        assert xp.server.port > 0

        listener = ProgressListener()
        xp.scheduler.addlistener(listener)

        out = NestedProgressingTask().submit()  # type: TaskOutput
        job = out.__xpm__.job
        path = out.path  # type: Path

        logger.info("Waiting for job to start")
        while job.state.notstarted():
            time.sleep(1e-2)

        logger.info("Checking job progress")

        levels = listener.progresses.get()
        assert levels == [LevelInformation(0, None, 0.0)]
        check_nested(path, listener, NestedProgressingTask.PROGRESSES)
        logger.info("All good!")
        path.touch()
