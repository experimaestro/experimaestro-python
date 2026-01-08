"""Simple experiment that does nothing."""

from experimaestro.experiments import ExperimentHelper, configuration, ConfigurationBase


@configuration()
class Configuration(ConfigurationBase):
    pass


def run(helper: ExperimentHelper, cfg: Configuration):
    pass
