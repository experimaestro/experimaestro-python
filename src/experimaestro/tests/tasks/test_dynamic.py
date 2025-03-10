# Test for future task outputs handling
# https://github.com/experimaestro/experimaestro-python/issues/90

import time
from experimaestro import Config, Param, Task, DependentMarker, LightweightTask
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

    def compute(self):
        self.register_task_output(self.checkpoint, step=15)


class Learn(Task):
    model: Param[Model]
    validation: Param[Validation]

    def execute(self):
        self.validation.compute()


def test_task_dynamic_simple():
    model = Model()
    validation = Validation(model=model)
    learn = Learn(model=model, validation=validation)
    evaluations = []

    def evaluate(checkpoint: Checkpoint):
        # Makes this harder...
        time.sleep(0.05)
        task = Evaluate(model=model)
        checkpoint_loader = CheckpointLoader(checkpoint=checkpoint)
        evaluations.append(task.submit(init_tasks=[checkpoint_loader]))

    with TemporaryDirectory() as workdir:
        with TemporaryExperiment("dynamic", maxwait=2, workdir=workdir):
            learn.watch_output(validation.checkpoint, evaluate)
            learn.submit()

        assert len(evaluations) == 1

        with TemporaryExperiment("dynamic", maxwait=2, workdir=workdir):
            learn.watch_output(validation.checkpoint, evaluate)

        assert len(evaluations) == 2
