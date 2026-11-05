from experimaestro import tag
from experimaestro.experiments import (
    ExperimentSpecification,
    ExperimentHelper,
    specification,
)
from experimaestro.launcherfinder import find_launcher

from tasks import Dataset, Evaluate, Learn


@specification
class Specification(ExperimentSpecification):
    learning_rates: list[float]
    """Learning rates to compare"""

    epochs: int = 100
    """Number of training epochs"""

    launcher: str = "cpu(cores=1)"
    """Resources needed by each task"""


def run(helper: ExperimentHelper, spec: Specification):
    launcher = find_launcher(spec.launcher)
    train = Dataset.C(size=100, seed=1)
    test = Dataset.C(size=100, seed=2)

    for lr in spec.learning_rates:
        learn = Learn.C(data=train, lr=tag(lr), epochs=spec.epochs)
        model = learn.submit(launcher=launcher)
        Evaluate.C(model=model, data=test).submit(launcher=launcher)
