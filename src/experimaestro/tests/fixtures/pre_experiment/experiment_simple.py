"""Simple experiment that does nothing."""

from experimaestro.experiments import (
    ExperimentHelper,
    specification,
    ExperimentSpecification,
)


@specification()
class Specification(ExperimentSpecification):
    pass


def run(helper: ExperimentHelper, spec: Specification):
    pass
