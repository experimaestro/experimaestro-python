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


def test_generators_reuse_on_submit():
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


def test_generators_delayed_submit():
    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_simple", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )
    validation = Validation.C()
    task1 = Learner.C(x=1, validation=validation)
    task2 = Learner.C(x=2, validation=validation)
    task1.submit(workspace=workspace)
    with pytest.raises((AttributeError)):
        task2.submit(workspace=workspace)


def test_generators_reuse_on_set():
    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_simple", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )
    validation = Validation.C()
    Learner.C(x=1, validation=validation).submit(workspace=workspace)
    with pytest.raises((AttributeError)):
        # We should not be able to *create* a second task with the same validation,
        # even without submitting it
        Learner.C(x=2, validation=validation)
