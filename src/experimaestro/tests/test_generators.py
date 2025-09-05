from experimaestro import Config, Task, Param, Meta, Path, field, PathGenerator
from experimaestro.scheduler.workspace import Workspace
from experimaestro.settings import Settings, WorkspaceSettings
import pytest
from experimaestro.scheduler import RunMode


class Validation(Config):
    best_checkpoint: Meta[Path] = field(default_factory=PathGenerator("index"))


class Learner(Task):
    validation: Param[Validation]
    x: Param[int]


def test_generators_reuse():
    # We have one way to select the best model
    validation = Validation.C()

    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_reuse", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )

    # OK, the path is generated depending on Learner with x=1
    Learner.C(x=1, validation=validation).submit(workspace=workspace)

    with pytest.raises((AttributeError)):
        # Here we have a problem...
        # the path is still the previous one
        Learner.C(x=2, validation=validation).submit(workspace=workspace)
