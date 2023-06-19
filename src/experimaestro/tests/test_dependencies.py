from typing import Any, Callable
import pytest
from experimaestro import Config, Param, Task, RunMode
from experimaestro.scheduler.base import JobDependency
from experimaestro.tests.utils import TemporaryExperiment


@pytest.fixture()
def xp():
    with TemporaryExperiment("deprecated", maxwait=0, run_mode=RunMode.DRY_RUN) as xp:
        yield xp


def check_dependencies(task: Task, *tasks: Task):
    deps = [
        id(dep.origin.config)
        for dep in task.__xpm__.job.dependencies
        if isinstance(dep, JobDependency)
    ]

    assert len(deps) == len(tasks)
    for task in tasks:
        assert (
            id(task) in deps
        ), f"Task {task.__xpmtype__} with ID {task.__identifier__()}"
        " is not not in the dependencies"


class TaskA(Task):
    pass


class TaskB(Task):
    a: Param[TaskA]


def test_dependencies_simple(xp):
    a = TaskA().submit()
    b = TaskB(a=a).submit()
    check_dependencies(b, a)


def test_dependencies_implicit(xp):
    a = TaskA().submit()
    b = TaskB(a=a)
    b.submit()
    check_dependencies(b, a)


class TaskC(Task):
    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return dep(ConfigC(param_c=self))


class ConfigC(Config):
    param_c: Param[TaskC]


class TaskD(Task):
    param_c: Param[ConfigC]


def test_dependencies_task_output(xp):
    task_c = TaskC()
    c = task_c.submit()
    d = TaskD(param_c=c).submit()
    check_dependencies(d, task_c)


class Inner_TaskA(Task):
    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return dep(Inner_OutputTaskA())


class Inner_OutputTaskA(Config):
    pass


class Inner_TaskB(Task):
    param_a: Param[Inner_OutputTaskA]


def test_dependencies_inner_task_output(xp):
    task_a = Inner_TaskA()
    a = task_a.submit()
    b = Inner_TaskB(param_a=a).submit()
    check_dependencies(b, task_a)


def test_dependencies_pre_task(xp):
    a = TaskA().submit()
    a2 = TaskA().add_pretasks(a).submit()
    check_dependencies(a2, a)
