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
    validation = Validation()

    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_reuse", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )

    with pytest.raises((ValueError, KeyError)):
        # OK, the path is generated depending on Learner with x=1
        Learner(x=1, validation=validation).submit(workspace=workspace)

        # Here we may have a problem...
        # if the path is still the previous one
        Learner(x=2, validation=validation).submit(workspace=workspace)

    # But if we use a different validation, it should be fine
    other_validation = Validation()
    Learner(x=3, validation=other_validation).submit(workspace=workspace)

    assert validation.best_checkpoint != other_validation.best_checkpoint
