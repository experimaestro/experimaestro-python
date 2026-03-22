from pathlib import Path
from typing import Optional

import pytest

from experimaestro import (
    Config,
    DataPath,
    Param,
    Task,
    state_dict,
    from_state_dict,
)
from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import ConfigMixin, ConfigInformation, TaskStub

# Mark all tests in this module as serialization tests (depends on identifier)
pytestmark = [
    pytest.mark.serialization,
    pytest.mark.dependency(depends=["mod_identifier"], scope="session"),
]


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


# --- Tests for DataPath serialization ---


class ConfigWithDataPath(Config):
    name: Param[str]
    data: DataPath


def test_datapath_serialization(tmp_path):
    """Test that DataPath fields are serialized (copied) into save_directory"""
    src_file = tmp_path / "source_data.txt"
    src_file.write_text("hello data")

    save_dir = tmp_path / "saved"
    save_dir.mkdir()

    config = ConfigWithDataPath.C(name="test", data=src_file)
    context = SerializationContext(save_directory=save_dir)
    config.__xpm__.seal(context)

    data = state_dict(context, config)

    # The data field should be serialized as path.serialized
    fields = data["objects"][0]["fields"]
    assert fields["data"]["type"] == "path.serialized"

    # The file should have been copied into save_dir
    serialized_path = Path(fields["data"]["value"])
    assert (save_dir / serialized_path).exists()
    assert (save_dir / serialized_path).read_text() == "hello data"


def test_datapath_deserialization(tmp_path):
    """Test round-trip: serialize then deserialize a DataPath config"""
    src_file = tmp_path / "source_data.txt"
    src_file.write_text("round trip")

    save_dir = tmp_path / "saved"
    save_dir.mkdir()

    config = ConfigWithDataPath.C(name="test", data=src_file)
    config.__xpm__.serialize(save_dir)

    loaded = ConfigInformation.deserialize(save_dir, as_instance=True)
    assert isinstance(loaded, ConfigWithDataPath)
    assert loaded.name == "test"
    assert isinstance(loaded.data, Path)
    assert loaded.data.read_text() == "round trip"


# --- Tests for custom __xpm_serialize__ ---


class ConfigWithCustomSerialize(Config):
    name: Param[str]
    data: DataPath

    def __xpm_serialize__(self, context):
        """Custom serialization: rename the data path"""
        result = {}
        for argument, value in self.__xpm__.xpmvalues():
            if argument.is_data and value is not None:
                # Serialize under a custom name instead of field name
                result[argument.name] = context.serialize(
                    context.var_path + ["custom_name"], value, self
                )
        return result


def test_custom_xpm_serialize(tmp_path):
    """Test that __xpm_serialize__ override changes the serialized path"""
    src_file = tmp_path / "source.txt"
    src_file.write_text("custom path")

    save_dir = tmp_path / "saved"
    save_dir.mkdir()

    config = ConfigWithCustomSerialize.C(name="test", data=src_file)
    context = SerializationContext(save_directory=save_dir)
    config.__xpm__.seal(context)

    data = state_dict(context, config)

    # The serialized path should use "custom_name" not "data"
    fields = data["objects"][0]["fields"]
    assert fields["data"]["type"] == "path.serialized"
    assert "custom_name" in fields["data"]["value"]

    # The file should exist at the custom path
    serialized_path = Path(fields["data"]["value"])
    assert (save_dir / serialized_path).exists()
    assert (save_dir / serialized_path).read_text() == "custom path"


class ConfigWithExtraData(Config):
    name: Param[str]

    def __xpm_serialize__(self, context):
        """Add extra data entries beyond declared fields"""
        result = super().__xpm_serialize__(context)
        # Add an extra entry that doesn't correspond to any field
        extra_path = Path(__file__)  # Use this test file as data
        result["extra_file"] = context.serialize(
            context.var_path + ["extra_file"], extra_path, self
        )
        return result


def test_custom_xpm_serialize_extra_entries(tmp_path):
    """Test that __xpm_serialize__ can add data entries beyond declared fields"""
    save_dir = tmp_path / "saved"
    save_dir.mkdir()

    config = ConfigWithExtraData.C(name="test")
    context = SerializationContext(save_directory=save_dir)
    config.__xpm__.seal(context)

    data = state_dict(context, config)

    fields = data["objects"][0]["fields"]
    # The extra entry should appear in the serialized fields
    assert "extra_file" in fields
    assert fields["extra_file"]["type"] == "path.serialized"

    # The file should have been copied
    serialized_path = Path(fields["extra_file"]["value"])
    assert (save_dir / serialized_path).exists()


class ConfigWithPathClash(Config):
    name: Param[str]
    data1: DataPath
    data2: DataPath

    def __xpm_serialize__(self, context):
        """Intentionally serialize two fields to the same path"""
        result = {}
        for argument, value in self.__xpm__.xpmvalues():
            if argument.is_data and value is not None:
                # Both fields serialize to the same destination
                result[argument.name] = context.serialize(
                    context.var_path + ["same_name"], value, self
                )
        return result


def test_serialization_path_clash(tmp_path):
    """Test that serializing two data entries to the same path raises an error"""
    src1 = tmp_path / "file1.txt"
    src1.write_text("one")
    src2 = tmp_path / "file2.txt"
    src2.write_text("two")

    save_dir = tmp_path / "saved"
    save_dir.mkdir()

    config = ConfigWithPathClash.C(name="test", data1=src1, data2=src2)
    context = SerializationContext(save_directory=save_dir)
    config.__xpm__.seal(context)

    with pytest.raises(ValueError, match="Serialization path conflict"):
        state_dict(context, config)
