from typing import Optional
from experimaestro import (
    Config,
    Param,
    Task,
    state_dict,
    from_state_dict,
)
from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import ConfigMixin, ConfigInformation, TaskStub


class Object1(Config):
    pass


class Object2(Config):
    object: Param[Object1]


def test_serializers_serialization():
    context = SerializationContext(save_directory=None)

    obj1 = Object1.C()
    obj2 = Object2.C(object=obj1)

    data = state_dict(context, [obj1, obj2])

    [obj1, obj2] = from_state_dict(data)
    assert isinstance(obj1, Object1) and isinstance(obj1, ConfigMixin)
    assert isinstance(obj2, Object2)
    assert obj2.object is obj1

    [obj1, obj2] = from_state_dict(data, as_instance=True)
    assert isinstance(obj1, Object1) and not isinstance(obj1, ConfigMixin)
    assert isinstance(obj2, Object2)
    assert obj2.object is obj1


class SubConfig(Config):
    pass


class MultiParamObject(Config):
    opt_a: Param[Optional[int]]

    x: Param[dict[str, Optional[SubConfig]]]


def test_serializers_types():
    context = SerializationContext(save_directory=None)

    config = MultiParamObject.C(x={"a": None})
    config.__xpm__.seal(context)
    state_dict(context, config)


# --- Tests for partial_loading feature ---


class MyTask(Task):
    """A task for testing partial_loading"""

    value: Param[int]

    def execute(self):
        pass


class ConfigWithTask(Config):
    """A config that references a task"""

    name: Param[str]


class TaskDependentConfig(Config):
    """A config that is only used by a task"""

    data: Param[str]


class TaskWithDependency(Task):
    """A task that uses another config"""

    dep: Param[TaskDependentConfig]

    def execute(self):
        pass


def test_partial_loading_skips_task_reference():
    """Test that partial_loading skips loading task references"""
    context = SerializationContext(save_directory=None)

    # Create a config with a task
    task = MyTask.C(value=42)
    config = ConfigWithTask.C(name="test")
    config.__xpm__.task = task

    # Seal both
    task.__xpm__.seal(context)
    config.__xpm__.seal(context)

    # Serialize
    data = state_dict(context, [task, config])

    # Load without partial_loading - task should be loaded
    [loaded_task, loaded_config] = from_state_dict(data, partial_loading=False)
    assert isinstance(loaded_config, ConfigWithTask)
    assert isinstance(loaded_config.__xpm__.task, MyTask)

    # Load with partial_loading - task should be a stub
    [loaded_task2, loaded_config2] = from_state_dict(data, partial_loading=True)
    assert isinstance(loaded_config2, ConfigWithTask)
    assert isinstance(loaded_config2.__xpm__.task, TaskStub)
    # Check that typename contains the task class name
    assert "MyTask" in loaded_config2.__xpm__.task.typename


def test_partial_loading_skips_task_dependencies():
    """Test that partial_loading skips configs only used by tasks"""
    context = SerializationContext(save_directory=None)

    # Create a config that is only used by a task
    dep = TaskDependentConfig.C(data="some data")
    task = TaskWithDependency.C(dep=dep)
    config = ConfigWithTask.C(name="main")
    config.__xpm__.task = task

    # Seal all
    dep.__xpm__.seal(context)
    task.__xpm__.seal(context)
    config.__xpm__.seal(context)

    # Serialize
    data = state_dict(context, [dep, task, config])
    definitions = data["objects"]

    # Load with partial_loading - both task and its dependency should be stubs
    objects = ConfigInformation.load_objects(
        definitions, as_instance=False, partial_loading=True
    )

    # The main config should be loaded
    main_obj = objects[definitions[-1]["id"]]
    assert isinstance(main_obj, ConfigWithTask)

    # The task should be a stub
    task_obj = objects[definitions[1]["id"]]
    assert isinstance(task_obj, TaskStub)

    # The task dependency should also be a stub
    dep_obj = objects[definitions[0]["id"]]
    assert isinstance(dep_obj, TaskStub)


def test_partial_loading_preserves_shared_configs():
    """Test that configs used by both main object and task are not skipped"""
    context = SerializationContext(save_directory=None)

    # Create a shared config
    shared = Object1.C()

    # Create task and main config that both use the shared config
    task = MyTask.C(value=1)
    main = Object2.C(object=shared)
    main.__xpm__.task = task

    # Seal all
    shared.__xpm__.seal(context)
    task.__xpm__.seal(context)
    main.__xpm__.seal(context)

    # Serialize
    data = state_dict(context, [shared, task, main])
    definitions = data["objects"]

    # Load with partial_loading
    objects = ConfigInformation.load_objects(
        definitions, as_instance=False, partial_loading=True
    )

    # The shared config should be loaded (not a stub) since main uses it
    shared_obj = objects[definitions[0]["id"]]
    assert isinstance(shared_obj, Object1)
    assert not isinstance(shared_obj, TaskStub)

    # The main config should be loaded
    main_obj = objects[definitions[-1]["id"]]
    assert isinstance(main_obj, Object2)

    # The task should be a stub
    task_obj = objects[definitions[1]["id"]]
    assert isinstance(task_obj, TaskStub)
