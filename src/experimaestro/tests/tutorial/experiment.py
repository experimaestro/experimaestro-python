from experimaestro import tag
from experimaestro.experiments import ConfigurationBase, ExperimentHelper, configuration
from experimaestro.launcherfinder import find_launcher

from tasks import Dataset, Evaluate, Learn


@configuration
class Configuration(ConfigurationBase):
    learning_rates: list[float]
    """Learning rates to compare"""

    epochs: int = 100
    """Number of training epochs"""

    launcher: str = "cpu(cores=1)"
    """Resources needed by each task"""


def run(helper: ExperimentHelper, cfg: Configuration):
    launcher = find_launcher(cfg.launcher)
    train = Dataset.C(size=100, seed=1)
    test = Dataset.C(size=100, seed=2)

    for lr in cfg.learning_rates:
        learn = Learn.C(data=train, lr=tag(lr), epochs=cfg.epochs)
        model = learn.submit(launcher=launcher)
        Evaluate.C(model=model, data=test).submit(launcher=launcher)
