"""Experiment that checks pre_experiment set an environment variable."""

import os
from experimaestro.experiments import (
    ExperimentHelper,
    specification,
    ExperimentSpecification,
)


@specification()
class Specification(ExperimentSpecification):
    pass


def run(helper: ExperimentHelper, spec: Specification):
    assert os.environ.get("XPM_TEST_PRE_EXPERIMENT") == "executed", (
        "Pre-experiment script was not executed"
    )
    print("PRE_EXPERIMENT_TEST_PASSED")  # noqa: T201
