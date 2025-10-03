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

    @staticmethod
    def create(x: int, validation: Param[Validation]):
        return Learner.C(x=x, validation=validation)


class LearnerList(Task):
    validation: Param[list[Validation]]
    x: Param[int]

    @staticmethod
    def create(x: int, validation: Param[Validation]):
        return LearnerList.C(x=x, validation=[validation])


class LearnerDict(Task):
    validation: Param[dict[str, Validation]]
    x: Param[int]

    @staticmethod
    def create(x: int, validation: Param[Validation]):
        return LearnerDict.C(x=x, validation={"key": validation})


class ModuleLoader(Task):
    validation: Param[Validation] = field(ignore_generated=True)


@pytest.mark.parametrize("cls", [Learner, LearnerDict, LearnerList])
def test_generators_reuse_on_submit(cls):
    # We have one way to select the best model
    validation = Validation.C()

    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_reuse", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )

    # OK, the path is generated depending on Learner with x=1
    cls.create(1, validation).submit(workspace=workspace)

    with pytest.raises((AttributeError)):
        # Here we have a problem...
        # the path is still the previous one
        cls.create(2, validation).submit(workspace=workspace)


@pytest.mark.parametrize("cls", [Learner, LearnerDict, LearnerList])
def test_generators_delayed_submit(cls):
    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_simple", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )
    validation = Validation.C()
    task1 = cls.create(1, validation)
    task2 = cls.create(2, validation)
    task1.submit(workspace=workspace)
    with pytest.raises((AttributeError)):
        task2.submit(workspace=workspace)


@pytest.mark.parametrize("cls", [Learner, LearnerDict, LearnerList])
def test_generators_reuse_on_set(cls):
    workspace = Workspace(
        Settings(),
        WorkspaceSettings("test_generators_simple", path=Path("/tmp")),
        run_mode=RunMode.DRY_RUN,
    )
    validation = Validation.C()
    cls.create(1, validation).submit(workspace=workspace)
    with pytest.raises((AttributeError)):
        # We should not be able to *create* a second task with the same validation,
        # even without submitting it
        cls.create(2, validation)

    # This should run OK
    ModuleLoader.C(validation=validation)
