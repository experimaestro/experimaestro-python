import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
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
    elif isinstance(value, list):
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
    """Returns a state dictionary of the object

    :param context: The serialization context
    :param obj: the object to serialize
    """
    objects = []
    data = json_object(context, obj, objects)
    return {"objects": objects, "data": data}


def save(obj: Any, save_directory: Path):
    """Saves an object into a disk file

    :param save_directory: The directory in which the object and its data will
        be saved (by default, the object is saved in "definition.json")
    """
    context = SerializationContext(save_directory=save_directory)

    data = state_dict(context, obj)
    with (save_directory / "definition.json").open("wt") as out:
        json.dump(data, out)


def get_data_loader(path: Union[str, Path, SerializedPathLoader]):
    if path is None:

        def data_loader():
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
    as_instance: bool = False
):
    """Load an object from a state dictionary

    :param state: The state
    :param path: A directory or a function that transforms relative file path
        into absolute ones
    :param as_instance: returns instances instead of configuration objects
    """
    objects = ConfigInformation.load_objects(
        state["objects"],
        as_instance=as_instance,
        data_loader=get_data_loader(path),
    )

    return ConfigInformation._objectFromParameters(state["data"], objects)


def load(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
) -> Tuple[Any, List["LightweightTask"]]:
    """Load data from disk

    :param path: A directory or a function that transforms relative file path
        into absolute ones
    :param as_instance: returns instances instead of configuration objects
    """
    data_loader = get_data_loader(path)

    with data_loader("definition.json").open("rt") as fh:
        content = json.load(fh)
    return from_state_dict(content, as_instance=as_instance)


def from_task_dir(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
):
    """Loads a task object"""
    data_loader = get_data_loader(path)
    with data_loader("params.json").open("rt") as fh:
        content = json.load(fh)

    content["data"] = {"type": "python", "value": content["objects"][-1]["id"]}

    return from_state_dict(content, as_instance=as_instance)
