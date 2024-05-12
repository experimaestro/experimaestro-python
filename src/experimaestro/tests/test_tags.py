from typing import Dict
from pathlib import Path
from experimaestro import (
    tag,
    LightweightTask,
    config,
    argument,
    Config,
    Task,
    Param,
)
from experimaestro.scheduler.workspace import RunMode
from experimaestro.xpmutils import DirectoryContext


@argument("x", type=int)
@config()
class Config1:
    pass


@argument("x", type=int)
@argument("c", type=Config1)
@config()
class Config2:
    pass


def test_tag():
    c = Config1(x=5)
    c.tag("x", 5)
    assert c.tags() == {"x": 5}


def test_taggedvalue():
    c = Config1(x=tag(5))
    assert c.tags() == {"x": 5}


def test_tagcontain():
    """Test that tags are not propagated to the upper configurations"""
    c1 = Config1(x=5)
    c2 = Config2(c=c1, x=tag(3)).tag("out", 1)
    assert c1.tags() == {}
    assert c2.tags() == {"x": 3, "out": 1}


def test_inneroutput():
    class Output(Config):
        pass

    class MyTask(Task):
        mainoutput: Param[Output]
        outputs: Param[Dict[str, Output]]

    class Evaluate(Task):
        task: Param[MyTask]

    output = Output().tag("hello", "world")
    task = MyTask(outputs={}, mainoutput=output)
    task.submit(run_mode=RunMode.DRY_RUN)
    assert output.tags() == {"hello": "world"}

    output = Output().tag("hello", "world")
    task = MyTask(outputs={"a": output}, mainoutput=Output())
    task.submit(run_mode=RunMode.DRY_RUN)
    assert output.tags() == {"hello": "world"}

    evaluate = Evaluate(task=task).submit(run_mode=RunMode.DRY_RUN)
    assert evaluate.__xpm__.tags() == {"hello": "world"}


def test_tags_init_tasks():
    """Test tags within init tasks"""

    class MyTask(Task):
        pass

    class InitTask(LightweightTask):
        pass

    init_task = InitTask().tag("hello", "world")
    task = MyTask()
    task.submit(run_mode=RunMode.DRY_RUN, init_tasks=[init_task])

    assert task.__xpm__.tags() == {"hello": "world"}


class TaskDirectoryContext(DirectoryContext):
    def __init__(self, task, path):
        super().__init__(path)
        self._task = task

    @property
    def task(self):
        return self._task


def test_objects_tags():
    """Test tags"""

    class A(Config):
        x: Param[int]

    context = DirectoryContext(Path("/__fakepath__"))
    a = A(x=tag(1))
    a.__xpm__.seal(context)
    assert a.__xpm__.tags() == {"x": 1}
