import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from experimaestro.core.context import (
    SerializationContext,
    SerializedPath,
    SerializedPathLoader,
)
from experimaestro.core.objects import ConfigInformation

if TYPE_CHECKING:
    from experimaestro import LightweightTask


def json_object(context: SerializationContext, value: Any, objects=[]):
    from experimaestro import Config

    if isinstance(value, Config):
        value.__xpm__.__get_objects__(objects, context)
    elif isinstance(value, (list, tuple)):
        for el in value:
            ConfigInformation.__collect_objects__(el, objects, context)
    elif isinstance(value, dict):
        for el in value.values():
            ConfigInformation.__collect_objects__(el, objects, context)
    else:
        raise NotImplementedError("Cannot serialize objects of type %s", type(value))

    # Finally, returns the output configuration
    return ConfigInformation._outputjsonvalue(value, context)


def state_dict(context: SerializationContext, obj: Any):
    """Convert an object to a state dictionary for serialization.

    Returns a dictionary representation that can be serialized to JSON
    and later restored with :func:`from_state_dict`.

    :param context: The serialization context
    :param obj: The object to serialize
    :return: A dictionary with 'objects' and 'data' keys
    """
    objects: list[Any] = []
    data = json_object(context, obj, objects)
    return {"objects": objects, "data": data}


def save_definition(obj: Any, context: SerializationContext, path: Path):
    data = state_dict(context, obj)
    with path.open("wt") as out:
        json.dump(data, out)


def save(obj: Any, save_directory: Optional[Path]):
    """Save a configuration to a directory.

    The serialization process stores the configuration in "definition.json"
    and copies any files or folders registered as DataPath parameters.

    Example::

        config = MyConfig.C(data_path=Path("/data/file.txt"))
        save(config, Path("/output/saved_config"))

    :param obj: The configuration to save
    :param save_directory: The directory in which the object and its data will
        be saved (object is saved in "definition.json")
    """
    context = SerializationContext(save_directory=save_directory)
    save_definition(
        obj, context, save_directory / "definition.json" if save_directory else None
    )


def get_data_loader(path: Union[str, Path, SerializedPathLoader]):
    if path is None:

        def data_loader(_: Union[str, Path, SerializedPathLoader]):
            # Just raise an exception
            raise RuntimeError("No serialization path was given")

        return data_loader

    if callable(path):
        return path

    path = Path(path)

    def data_loader(s: Union[Path, str, SerializedPath]):
        if isinstance(s, SerializedPath):
            return path / Path(s.path)
        return path / Path(s)

    return data_loader


def from_state_dict(
    state: Dict[str, Any],
    path: Union[None, str, Path, SerializedPathLoader] = None,
    *,
    as_instance: bool = False,
    partial_loading: Optional[bool] = None,
):
    """Load an object from a state dictionary.

    Restores a configuration from a dictionary previously created by
    :func:`state_dict`.

    :param state: The state dictionary to load from
    :param path: Directory containing data files, or a function that resolves
        relative paths to absolute ones
    :param as_instance: If True, return an instance instead of a config
    :param partial_loading: If True, skip loading task references. If None
        (default), partial_loading is enabled when as_instance is True.
    :return: The loaded configuration or instance
    """
    # Determine effective partial_loading: as_instance implies partial_loading
    effective_partial_loading = (
        partial_loading if partial_loading is not None else as_instance
    )

    objects = ConfigInformation.load_objects(
        state["objects"],
        as_instance=as_instance,
        data_loader=get_data_loader(path),
        partial_loading=effective_partial_loading,
    )

    return ConfigInformation._objectFromParameters(state["data"], objects)


def load(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
    partial_loading: Optional[bool] = None,
) -> Tuple[Any, List["LightweightTask"]]:
    """Load a configuration from a directory.

    Restores a configuration previously saved with :func:`save`.

    Example::

        config = load(Path("/output/saved_config"))

    :param path: Directory containing the saved configuration, or a function
        that resolves relative paths to absolute ones
    :param as_instance: If True, return an instance instead of a config
    :param partial_loading: If True, skip loading task references. If None
        (default), partial_loading is enabled when as_instance is True.
    :return: The loaded configuration or instance
    """
    data_loader = get_data_loader(path)

    with data_loader("definition.json").open("rt") as fh:
        content = json.load(fh)
    return from_state_dict(
        content, as_instance=as_instance, partial_loading=partial_loading
    )


def from_task_dir(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
    partial_loading: Optional[bool] = None,
):
    """Load a task configuration from a task directory.

    Loads the task parameters from a job directory (containing params.json).
    This is useful for reloading task configurations after execution.

    :param path: Task directory containing params.json, or a function that
        resolves relative paths to absolute ones
    :param as_instance: If True, return an instance instead of a config
    :param partial_loading: If True, skip loading task references. If None
        (default), partial_loading is enabled when as_instance is True.
    :return: The loaded task configuration or instance
    """
    data_loader = get_data_loader(path)
    with data_loader("params.json").open("rt") as fh:
        content = json.load(fh)

    content["data"] = {"type": "python", "value": content["objects"][-1]["id"]}

    return from_state_dict(
        content, as_instance=as_instance, partial_loading=partial_loading
    )


def serialize(
    obj: Any, save_directory: Path, *, init_tasks: list["LightweightTask"] = []
):
    """Serialize a configuration to a directory with initialization tasks.

    Similar to :func:`save`, but also stores lightweight initialization tasks
    that should be run when the configuration is deserialized.

    :param obj: The configuration to serialize
    :param save_directory: The directory in which the object and its data will
        be saved (object is saved in "definition.json")
    :param init_tasks: List of lightweight tasks to run on deserialization
    """
    context = SerializationContext(save_directory=save_directory)
    save_definition((obj, init_tasks), context, save_directory / "definition.json")


def deserialize(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
    partial_loading: Optional[bool] = None,
) -> tuple[Any, List["LightweightTask"]] | Any:
    """Deserialize a configuration from a directory.

    Restores a configuration previously saved with :func:`serialize`.
    When ``as_instance=True``, runs any stored initialization tasks.

    :param path: Directory containing the serialized configuration, or a function
        that resolves relative paths to absolute ones
    :param as_instance: If True, return an instance and run init tasks
    :param partial_loading: If True, skip loading task references. If None
        (default), partial_loading is enabled when as_instance is True.
    :return: The configuration/instance (if as_instance), or tuple of
        (configuration, init_tasks)
    """
    data_loader = get_data_loader(path)

    with data_loader("definition.json").open("rt") as fh:
        content = json.load(fh)

    object, init_tasks = from_state_dict(
        content, data_loader, as_instance=as_instance, partial_loading=partial_loading
    )

    if as_instance:
        for init_task in init_tasks:
            init_task.execute()

        return object

    return object, init_tasks
