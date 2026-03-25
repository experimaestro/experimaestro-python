import json
from dataclasses import dataclass
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
    from experimaestro.core.objects import ConfigMixin

    if isinstance(value, ConfigMixin):
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


def save(
    obj: Any,
    save_directory: Optional[Path],
    definition_filename: str = "experimaestro.json",
):
    """Save a configuration to a directory.

    The serialization process stores the configuration in the definition file
    and copies any files or folders registered as DataPath parameters.

    Example::

        config = MyConfig.C(data_path=Path("/data/file.txt"))
        save(config, Path("/output/saved_config"))

    :param obj: The configuration to save
    :param save_directory: The directory in which the object and its data will
        be saved
    :param definition_filename: The filename for the definition file
        (default: "experimaestro.json")
    """
    context = SerializationContext(save_directory=save_directory)
    save_definition(
        obj,
        context,
        save_directory / definition_filename if save_directory else None,
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


def _resolve_definition_filename(data_loader, definition_filename: str | None) -> str:
    """Resolve the definition filename, trying experimaestro.json first if not specified."""
    if definition_filename is not None:
        return definition_filename
    xpm_path = data_loader("experimaestro.json")
    if xpm_path.exists():
        return "experimaestro.json"
    return "definition.json"


def load(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
    partial_loading: Optional[bool] = None,
    definition_filename: str | None = None,
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
    :param definition_filename: The definition filename. If None, tries
        "experimaestro.json" first, then falls back to "definition.json".
    :return: The loaded configuration or instance
    """
    data_loader = get_data_loader(path)
    definition_filename = _resolve_definition_filename(data_loader, definition_filename)

    with data_loader(definition_filename).open("rt") as fh:
        content = json.load(fh)
    return from_state_dict(
        content, path=path, as_instance=as_instance, partial_loading=partial_loading
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
    obj: Any,
    save_directory: Path,
    *,
    init_tasks: list["LightweightTask"] = [],
    definition_filename: str = "experimaestro.json",
):
    """Serialize a configuration to a directory with initialization tasks.

    Similar to :func:`save`, but also stores lightweight initialization tasks
    that should be run when the configuration is deserialized.

    :param obj: The configuration to serialize
    :param save_directory: The directory in which the object and its data will
        be saved
    :param init_tasks: List of lightweight tasks to run on deserialization
    :param definition_filename: The filename for the definition file
        (default: "experimaestro.json")
    """
    context = SerializationContext(save_directory=save_directory)
    save_definition((obj, init_tasks), context, save_directory / definition_filename)


def deserialize(
    path: Union[str, Path, SerializedPathLoader],
    as_instance: bool = False,
    partial_loading: Optional[bool] = None,
    definition_filename: str | None = None,
) -> tuple[Any, List["LightweightTask"]] | Any:
    """Deserialize a configuration from a directory.

    Restores a configuration previously saved with :func:`serialize`.
    When ``as_instance=True``, runs any stored initialization tasks.

    :param path: Directory containing the serialized configuration, or a function
        that resolves relative paths to absolute ones
    :param as_instance: If True, return an instance and run init tasks
    :param partial_loading: If True, skip loading task references. If None
        (default), partial_loading is enabled when as_instance is True.
    :param definition_filename: The definition filename. If None, tries
        "experimaestro.json" first, then falls back to "definition.json".
    :return: The configuration/instance (if as_instance), or tuple of
        (configuration, init_tasks)
    """
    data_loader = get_data_loader(path)
    definition_filename = _resolve_definition_filename(data_loader, definition_filename)

    with data_loader(definition_filename).open("rt") as fh:
        content = json.load(fh)

    object, init_tasks = from_state_dict(
        content, data_loader, as_instance=as_instance, partial_loading=partial_loading
    )

    if as_instance:
        for init_task in init_tasks:
            init_task.execute()

        return object

    return object, init_tasks


@dataclass
class ExperimentInfo:
    """Structured result from loading experiment objects.

    Contains deserialized job configs and actions from an experiment run.
    """

    jobs: Dict[str, Any]
    """Mapping of job_id to Config objects"""

    actions: Dict[str, Any]
    """Mapping of action_id to Action objects"""


def load_xp_info(
    path: Union[str, Path],
) -> ExperimentInfo:
    """Load all serialized objects from an experiment run directory.

    Reads ``objects.jsonl`` (streaming format) to reconstruct job configs
    and actions. Uses ``jobs.jsonl`` for job IDs and ``status.json`` for
    action IDs to classify entries.

    Falls back to ``configs.json`` for experiments created before the
    ``objects.jsonl`` format was introduced.

    This is a standalone function -- no experiment context or
    ``WorkspaceStateProvider`` is required.

    :param path: Path to the experiment run directory
    :return: ExperimentInfo with .jobs and .actions dictionaries
    :raises FileNotFoundError: If neither objects.jsonl nor configs.json exists
    """
    path = Path(path)
    objects_path = path / "objects.jsonl"

    if objects_path.exists():
        return _load_from_objects_jsonl(path)

    # Backward compat: fall back to configs.json
    configs_path = path / "configs.json"
    if configs_path.exists():
        return _load_from_configs_json(configs_path)

    raise FileNotFoundError(f"Neither objects.jsonl nor configs.json found at {path}")


def _load_from_objects_jsonl(run_dir: Path) -> ExperimentInfo:
    """Load experiment info from objects.jsonl format.

    Object IDs are Python id() values (memory addresses) that are globally
    unique within the same writer session. Since ObjectsWriter uses a shared
    SerializationContext, shared objects only appear once across all entries.
    We accumulate all objects into a single list and deserialize them together.
    """
    objects_path = run_dir / "objects.jsonl"

    # Read job IDs from jobs.jsonl
    job_ids: set[str] = set()
    jobs_path = run_dir / "jobs.jsonl"
    if jobs_path.exists():
        with jobs_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    job_ids.add(entry["job_id"])

    # Read action IDs from status.json
    action_ids: set[str] = set()
    status_path = run_dir / "status.json"
    if status_path.exists():
        with status_path.open() as f:
            status = json.load(f)
            for action_id in status.get("actions", {}):
                action_ids.add(action_id)

    # Read objects.jsonl: accumulate all serialized objects and track entries
    all_objects: list = []
    entries: list[tuple[str, Any]] = []  # (id, data_ref)

    with objects_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry_id = entry["id"]
            new_objects = entry.get("objects", [])

            # No remapping needed — IDs are globally unique id() values
            all_objects.extend(new_objects)
            entries.append((entry_id, entry["data"]))

    # Deserialize all objects at once (shared references preserved)
    deserialized = ConfigInformation.load_objects(all_objects, as_instance=False)

    # Classify entries into jobs and actions
    jobs: Dict[str, Any] = {}
    actions: Dict[str, Any] = {}

    for entry_id, data_ref in entries:
        obj = ConfigInformation._objectFromParameters(data_ref, deserialized)
        if entry_id in job_ids:
            jobs[entry_id] = obj
        elif entry_id in action_ids:
            actions[entry_id] = obj
        else:
            # Unknown ID — store in jobs by default
            jobs[entry_id] = obj

    return ExperimentInfo(jobs=jobs, actions=actions)


def _load_from_configs_json(configs_path: Path) -> ExperimentInfo:
    """Backward-compatible loading from old configs.json format."""
    with configs_path.open() as f:
        data = json.load(f)

    tags = data.pop("tags", {})
    configs = from_state_dict(data)

    for job_id, job_tags in tags.items():
        if job_id in configs:
            for tag_name, tag_value in job_tags.items():
                configs[job_id].tag(tag_name, tag_value)

    return ExperimentInfo(jobs=configs, actions={})


def load_configs(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """Load all job configs from an experiment run directory.

    .. deprecated::
        Use :func:`load_xp_info` instead, which returns both jobs and actions.

    :param path: Path to the experiment run directory
    :return: Dictionary mapping job identifiers to their Config objects
    """
    import warnings

    warnings.warn(
        "load_configs() is deprecated, use load_xp_info() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_xp_info(path).jobs
