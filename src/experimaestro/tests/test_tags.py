from typing import Dict
from pathlib import Path
from experimaestro import (
    tag,
    LightweightTask,
    Config,
    Task,
    Param,
)
from experimaestro.scheduler.workspace import RunMode
from experimaestro.xpmutils import DirectoryContext


class Config1(Config):
    x: Param[int]


class Config2(Config):
    x: Param[int]
    c: Param[Config1]


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

    class MyConfig(Config):
        pass

    class TaskWithOutput(Task):
        x: Param[MyConfig]

        def task_outputs(self, dep) -> MyConfig:
            return dep(MyConfig())

    init_task = InitTask().tag("hello", "world")
    task = MyTask()
    result = task.submit(run_mode=RunMode.DRY_RUN, init_tasks=[init_task])
    assert result.tags() == {"hello": "world"}

    other_task = TaskWithOutput(x=MyConfig().tag("hello", "world"))
    assert other_task.tags() == {"hello": "world"}

    result = other_task.submit(run_mode=RunMode.DRY_RUN)
    assert isinstance(result, MyConfig)
    assert result.tags() == {"hello": "world"}

    result = MyTask().submit(run_mode=RunMode.DRY_RUN, init_tasks=[result])
    assert result.tags() == {"hello": "world"}


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
