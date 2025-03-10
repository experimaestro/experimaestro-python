# Test for future task outputs handling
# https://github.com/experimaestro/experimaestro-python/issues/90

from functools import partial
import sys
import time
from experimaestro import (
    Config,
    Param,
    Task,
    DependentMarker,
    LightweightTask,
    FailedExperiment,
)
from experimaestro.core.arguments import Meta
from experimaestro.tests.utils import TemporaryDirectory, TemporaryExperiment


class Model(Config):
    pass


class Checkpoint(Config):
    step: Param[int]
    model: Param[Model]


class CheckpointLoader(LightweightTask):
    checkpoint: Param[Checkpoint]

    def execute(self):
        pass


class Evaluate(Task):
    model: Param[Model]

    def execute(self):
        pass


class Validation(Config):
    model: Param[Model]

    def checkpoint(self, dep: DependentMarker, *, step: int) -> Checkpoint:
        return dep(Checkpoint(model=self.model, step=step))

    def compute(self, step: int):
        self.register_task_output(self.checkpoint, step=step)


class Learn(Task):
    model: Param[Model]
    validation: Param[Validation]

    #: Maximum step before we fail
    start_step: Meta[int] = 0

    #: Maximum step before we fail
    max_step: Meta[int] = sys.maxsize

    def execute(self):
        for step in [15, 30]:
            if step > self.max_step:
                sys.exit(1)
            if step <= self.start_step:
                continue
            self.validation.compute(step)


def evaluate(evaluations, checkpoint: Checkpoint):
    # Makes this harder...
    time.sleep(0.05)
    task = Evaluate(model=checkpoint.model)
    checkpoint_loader = CheckpointLoader(checkpoint=checkpoint)
    evaluations.append(task.submit(init_tasks=[checkpoint_loader]))


def test_task_dynamic_simple():
    evaluations = []

    def evaluate(checkpoint: Checkpoint):
        # Makes this harder...
        time.sleep(0.05)
        task = Evaluate(model=checkpoint.model)
        checkpoint_loader = CheckpointLoader(checkpoint=checkpoint)
        evaluations.append(task.submit(init_tasks=[checkpoint_loader]))

    with TemporaryDirectory() as workdir:
        with TemporaryExperiment("dynamic", maxwait=2, workdir=workdir):
            model = Model()
            validation = Validation(model=model)
            learn = Learn(model=model, validation=validation)
            learn.watch_output(validation.checkpoint, partial(evaluate, evaluations))

            learn.submit()

        assert len(evaluations) == 2

        # Re-run the learning process – it should
        with TemporaryExperiment("dynamic", maxwait=2, workdir=workdir):
            learn = Learn(model=model, validation=validation)
            validation = Validation(model=model)
            learn.watch_output(validation.checkpoint, partial(evaluate, evaluations))
            learn.submit()

        assert len(evaluations) == 4


def test_task_dynamic_checkpointing():
    """Test dynamic task output when the task is re-launched from a checkpoint"""
    evaluations = []

    with TemporaryDirectory() as workdir:
        try:
            with TemporaryExperiment("dynamic", maxwait=2, workdir=workdir):
                model = Model()
                validation = Validation(model=model)
                learn = Learn(model=model, validation=validation, max_step=15)
                learn.watch_output(
                    validation.checkpoint, partial(evaluate, evaluations)
                )

                learn.submit()
        except FailedExperiment:
            pass

        assert len(evaluations) == 1

        # Re-run the learning process – it should
        with TemporaryExperiment("dynamic", maxwait=2, workdir=workdir):
            model = Model()
            learn = Learn(model=model, validation=validation, start_step=15)
            validation = Validation(model=model)
            learn.watch_output(validation.checkpoint, partial(evaluate, evaluations))

            learn.submit()

        assert len(evaluations) == 3
