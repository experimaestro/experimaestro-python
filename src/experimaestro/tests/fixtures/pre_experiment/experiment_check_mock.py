"""Experiment that checks pre_experiment mocked a module."""

import xpm_fake_module  # noqa: F401 - this import will fail without pre_experiment
from experimaestro.experiments import ExperimentHelper, configuration, ConfigurationBase


@configuration()
class Configuration(ConfigurationBase):
    pass


def run(helper: ExperimentHelper, cfg: Configuration):
    assert xpm_fake_module.value == 42, "Mock module should have value 42"
    print("MOCK_MODULE_TEST_PASSED")  # noqa: T201
