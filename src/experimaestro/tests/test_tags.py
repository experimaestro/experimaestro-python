from typing import Dict
from pathlib import Path
import logging
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
    c = Config1.C(x=5)
    c.tag("x", 5)
    assert c.tags() == {"x": 5}


def test_taggedvalue():
    c = Config1.C(x=tag(5))
    assert c.tags() == {"x": 5}


def test_tagcontain():
    """Test that tags are not propagated to the upper configurations"""
    c1 = Config1.C(x=5)
    c2 = Config2.C(c=c1, x=tag(3)).tag("out", 1)
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

    output = Output.C().tag("hello", "world")
    task = MyTask.C(outputs={}, mainoutput=output)
    task.submit(run_mode=RunMode.DRY_RUN)
    assert output.tags() == {"hello": "world"}

    output = Output.C().tag("hello", "world")
    task = MyTask.C(outputs={"a": output}, mainoutput=Output.C())
    task.submit(run_mode=RunMode.DRY_RUN)
    assert output.tags() == {"hello": "world"}

    evaluate = Evaluate.C(task=task).submit(run_mode=RunMode.DRY_RUN)
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
            return dep(MyConfig.C())

    init_task = InitTask.C().tag("hello", "world")
    task = MyTask.C()
    result = task.submit(run_mode=RunMode.DRY_RUN, init_tasks=[init_task])
    assert result.tags() == {"hello": "world"}

    other_task = TaskWithOutput.C(x=MyConfig.C().tag("hello", "world"))
    assert other_task.tags() == {"hello": "world"}

    result = other_task.submit(run_mode=RunMode.DRY_RUN)
    assert isinstance(result, MyConfig)
    assert result.tags() == {"hello": "world"}

    result = MyTask.C().submit(run_mode=RunMode.DRY_RUN, init_tasks=[result])
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
    a = A.C(x=tag(1))
    a.__xpm__.seal(context)
    assert a.__xpm__.tags() == {"x": 1}


def test_conflicting_tags_warning(caplog):
    """Test that conflicting tag values produce a warning"""

    class Inner(Config):
        value: Param[int]

    class Outer(Config):
        inner: Param[Inner]
        x: Param[int]

    # Create inner config with tag "mytag" = 1
    inner = Inner.C(value=10).tag("mytag", 1)

    # Create outer config with same tag "mytag" = 2 (conflicting)
    outer = Outer.C(inner=inner, x=5).tag("mytag", 2)

    # Getting tags should warn about conflict
    with caplog.at_level(logging.WARNING):
        tags = outer.tags()

    # The warning should mention the conflicting tag
    assert any("mytag" in record.message for record in caplog.records)
    assert any("conflicting" in record.message.lower() for record in caplog.records)

    # The last value should win
    assert tags["mytag"] == 2


def test_same_tag_same_value_no_warning(caplog):
    """Test that same tag with same value does not produce a warning"""

    class Inner(Config):
        value: Param[int]

    class Outer(Config):
        inner: Param[Inner]

    # Create inner config with tag "mytag" = 1
    inner = Inner.C(value=10).tag("mytag", 1)

    # Create outer config with same tag "mytag" = 1 (same value)
    outer = Outer.C(inner=inner).tag("mytag", 1)

    # Getting tags should NOT warn (same value)
    with caplog.at_level(logging.WARNING):
        tags = outer.tags()

    # No warning for same values
    assert not any("mytag" in record.message for record in caplog.records)
    assert tags["mytag"] == 1


def test_tag_source_tracking():
    """Test that tag source locations are tracked"""

    class MyConfig(Config):
        x: Param[int]

    config = MyConfig.C(x=tag(5))

    # Check that tags have source info stored internally
    assert "x" in config.__xpm__._tags
    value, source = config.__xpm__._tags["x"]
    assert value == 5
    # Source should contain file path and line number
    assert ":" in source
    assert "test_tags.py" in source


def test_tag_method_source_tracking():
    """Test that tag() method also tracks source location"""

    class MyConfig(Config):
        x: Param[int]

    config = MyConfig.C(x=5)
    config.tag("mytag", "myvalue")

    # Check that tag has source info
    assert "mytag" in config.__xpm__._tags
    value, source = config.__xpm__._tags["mytag"]
    assert value == "myvalue"
    assert ":" in source
    assert "test_tags.py" in source


def test_tag_via_setattr():
    """Test that config.key = tag(value) works and tracks source"""

    class MyConfig(Config):
        x: Param[int]

    config = MyConfig.C(x=5)
    config.x = tag(10)

    # Check that tag was set correctly
    assert config.tags() == {"x": 10}
    assert config.x == 10

    # Check that source is tracked
    value, source = config.__xpm__._tags["x"]
    assert value == 10
    assert "test_tags.py" in source


def test_tag_setattr_conflict_warning(caplog):
    """Test that setting conflicting tag via setattr produces warning"""

    class Inner(Config):
        value: Param[int]

    class Outer(Config):
        inner: Param[Inner]
        x: Param[int]

    # Create with tag via constructor
    inner = Inner.C(value=tag(1))

    # Create outer with same tag name
    outer = Outer.C(inner=inner, x=5)
    outer.x = tag(2)  # Set tag on x

    # Add a conflicting value tag
    outer.tag("value", 99)

    # Getting tags should warn about conflict
    with caplog.at_level(logging.WARNING):
        tags = outer.tags()

    # The warning should mention the conflicting tag
    assert any("value" in record.message for record in caplog.records)
    assert tags["value"] == 99  # Last value wins
    assert tags["x"] == 2
