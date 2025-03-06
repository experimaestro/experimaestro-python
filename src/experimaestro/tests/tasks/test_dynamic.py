# Test for future task outputs handling
# https://github.com/experimaestro/experimaestro-python/issues/90

from experimaestro import Config, Param, Task, DependentMarker
from experimaestro.scheduler.base import JobState
from experimaestro.tests.utils import TemporaryExperiment


class Model(Config):
    pass


class Checkpoint(Config):
    step: Param[int]
    model: Param[Model]


class CheckpointLoader(Config):
    checkpoint: Param[Checkpoint]


class Evaluate(Task):
    model: Param[Model]

    def execute(self):
        pass


class Validation(Config):
    model: Param[Model]

    def checkpoint(self, dep: DependentMarker, *, step: int) -> Checkpoint:
        return dep(Checkpoint(model=self.model, step=step))

    def validation(self):
        self.register_task_output(self.checkpoint, step=15)


class Learn(Task):
    model: Param[Model]
    validation: Param[Validation]

    def execute(self):
        self.validation()


def test_task_dynamic_simple():
    model = Model()
    validation = Validation(model=model)
    learn = Learn(model=model, validation=validation)
    evaluations = []

    def evaluate(checkpoint: Checkpoint):
        evaluations.append(Evaluate().submit(init_task=[CheckpointLoader(checkpoint=checkpoint)]))
    learn.watch_output(validation.checkpoint, evaluate)

    with TemporaryExperiment("dynamic"):
        assert learn.submit().__xpm__.job.wait() == JobState.DONE

    assert len(evaluations) == 2
