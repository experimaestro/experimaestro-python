"""Configuration and tasks"""

import json

from attr import define
import fasteners

from experimaestro import taskglobals

from termcolor import cprint
from pathlib import Path
import hashlib
import logging
import io
from enum import Enum
import inspect
import importlib
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING,
)
import sys
import experimaestro
from experimaestro.utils import logger
from experimaestro.core.types import DeprecatedAttribute, ObjectType, TypeVarType
from ..context import SerializationContext, SerializedPath, SerializedPathLoader

if TYPE_CHECKING:
    from ..callbacks import TaskEventListener
    from ..identifier import Identifier
    from experimaestro.scheduler.base import Job
    from experimaestro.scheduler.workspace import RunMode
    from experimaestro.launchers import Launcher
    from experimaestro.scheduler import Workspace

from .config_walk import ConfigWalk, ConfigWalkContext
from .config_utils import (
    getqualattr,
    add_to_path,
    SealedError,
    TaggedValue,
    ObjectStore,
    classproperty,
)

T = TypeVar("T", bound="Config")


DependentMarker = Callable[["Config"], None]


def updatedependencies(
    dependencies, value: "Config", path: List[str], taskids: Set[int]
):
    """Search recursively jobs to add them as dependencies

    Arguments:
        dependencies: The current set of dependencies
        value: The current inspected configuration
        path: The current path (for error tracing)
        taskids: Sets of added tasks (ids) to avoid repeated depencies
    """

    # Adds pre-tasks
    if isinstance(value, Config):
        value.__xpm__.updatedependencies(dependencies, path, taskids)
    elif isinstance(value, (list, set)):
        for el in value:
            updatedependencies(dependencies, el, path, taskids)
    elif isinstance(value, (dict,)):
        for key, val in value.items():
            updatedependencies(dependencies, key, path, taskids)
            updatedependencies(dependencies, val, path, taskids)
    elif isinstance(value, (str, int, float, Path, Enum)):
        pass
    else:
        raise NotImplementedError("update dependencies for type %s" % type(value))


NOT_SET = object()


@define()
class WatchedOutput:
    #: The enclosing job
    job: "Job"

    #: The configuration containing the watched output
    config: "ConfigInformation"

    #: The watched output (name)
    method_name: str

    #: The watched output method (called with the JSON event)
    method: Callable

    #: The callback to call (with the output of the previous method)
    callback: Callable


class ConfigInformation:
    """Holds experimaestro information for a config (or task) instance"""

    _meta: Optional[bool]
    """Forces this configuration to be a meta-parameter"""

    # Set to true when loading from JSON
    LOADING: ClassVar[bool] = False

    def __init__(self, pyobject: "ConfigMixin"):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject
        self.xpmtype: "ObjectType" = pyobject.__xpmtype__
        self.values = {}

        # Meta-informations
        self._tags: dict[str, Any] = {}
        self._initinfo = ""

        self._taskoutput = None
        """Task output (caches the value of a submit)"""

        self.task: Optional["Config"] = None
        """The task this configuration depends upon (or None)"""

        # State information
        self.job = None
        self._job_listener: "TaskEventListener" | None = None

        #: True when this configuration was loaded from disk
        self.loaded = False

        # Explicitly added dependencies
        self.dependencies = []

        # Concrete type variables resolutions
        # This is used to check typevars coherence
        self.concrete_typevars: Dict[TypeVar, type] = {}

        # Lightweight tasks
        self.pre_tasks: List["LightweightTask"] = []

        # Initialization tasks
        self.init_tasks: List["LightweightTask"] = []

        # Watched outputs
        self.watched_outputs: List[WatchedOutput] = []

        # Cached information

        self._full_identifier = None
        """The full identifier (with pre-tasks)"""

        self._raw_identifier = None
        """The identifier without taking into account pre-tasks"""

        self._validated = False
        self._sealed = False
        self._meta = None

        #: This flags is True when a value in this configuration,
        #: or any sub-configuration, is generated. This prevents problem
        #: when a configuration with generated values is re-used
        self._has_generated_value = False

    def set_meta(self, value: Optional[bool]):
        """Sets the meta flag"""
        assert not self._sealed, "Configuration is sealed"
        self._meta = value

    @property
    def meta(self):
        return self._meta

    def get(self, name):
        """Get an XPM managed value"""
        if name in self.xpmtype.arguments:
            return self.values[name]

        # Not an argument, bypass
        return object.__getattribute__(self.pyobject, name)

    def set(self, k, v, bypass=False):
        from experimaestro.generators import Generator

        # Not an argument, bypass
        if k not in self.xpmtype.arguments:
            setattr(self.pyobject, k, v)
            return

        if self._sealed and not bypass:
            raise AttributeError(f"Object is read-only (trying to set {k})")

        if not isinstance(v, ConfigMixin) and isinstance(v, Config):
            raise AttributeError(
                "Configuration (and not objects) should be used. Consider using .C(...)"
            )

        if (
            isinstance(v, ConfigMixin)
            and v.__xpm__._has_generated_value
            and v.__xpm__.task is None
        ):
            raise AttributeError(
                f"Cannot set {k} to a configuration with generated values"
            )

        try:
            argument = self.xpmtype.arguments.get(k, None)
            if argument:
                if not bypass and (
                    (isinstance(argument.generator, Generator)) or argument.constant
                ):
                    raise AttributeError("Property %s is read-only" % (k))
                if v is not None:
                    self.values[k] = argument.validate(v)
                    # Check for type variables
                    if type(argument.type) is TypeVarType:
                        self.check_typevar(argument.type.typevar, type(v))
                    if isinstance(v, Config):
                        # If the value is a Config, fuse type variables
                        v.__xpm__.fuse_concrete_typevars(self.concrete_typevars)
                        self.fuse_concrete_typevars(v.__xpm__.concrete_typevars)
                elif argument.required:
                    raise AttributeError("Cannot set required attribute to None")
                else:
                    self.values[k] = None
            else:
                raise AttributeError(
                    "Cannot set non existing attribute %s in %s" % (k, self.xpmtype)
                )
        except Exception:
            logger.error("Error while setting value %s in %s", k, self.xpmtype)
            raise

    def fuse_concrete_typevars(self, typevars: Dict[TypeVar, type]):
        """Fuses concrete type variables with the current ones"""
        for typevar, v in typevars.items():
            self.check_typevar(typevar, v)

    def check_typevar(self, typevar: TypeVar, v: type):
        """Check if a type variable is coherent with the current typevars bindings,
        updates the bindings if necessary"""
        if typevar not in self.concrete_typevars:
            self.concrete_typevars[typevar] = v
            return

        concrete_typevar = self.concrete_typevars[typevar]
        bound = typevar.__bound__
        # Check that v is a subclass of the typevar OR that typevar is a subclass of v
        # Then set the concrete type variable to the most generic type

        # First, limiting to the specified bound
        if bound is not None:
            if not issubclass(v, bound):
                raise TypeError(
                    f"Type variable {typevar} is bound to {bound}, but tried to set it to {v}"
                )

        if issubclass(v, concrete_typevar):
            # v is a subclass of the typevar, keep the typevar
            return
        if issubclass(concrete_typevar, v):
            # typevar is a subclass of v, keep v
            self.concrete_typevars[typevar] = v
            return
        raise TypeError(
            f"Type variable {typevar} is already set to {self.concrete_typevars[typevar]}, "
            f"but tried to set it to {v}"
            f" (current typevars bindings: {self.concrete_typevars})"
        )

    def addtag(self, name, value):
        self._tags[name] = value

    def xpmvalues(self, generated=False):
        """Returns an iterarator over arguments and associated values"""
        for argument in self.xpmtype.arguments.values():
            if argument.name in self.values or (generated and argument.generator):
                yield argument, self.values[argument.name]

    def tags(self):
        class TagFinder(ConfigWalk):
            def __init__(self):
                super().__init__(recurse_task=True)
                self.tags = {}

            def postprocess(self, stub, config: Config, values):
                self.tags.update(config.__xpm__._tags)
                return self.tags

        return TagFinder()(self.pyobject)

    def validate(self):
        """Validate a value"""
        if not self._validated:
            self._validated = True

            # Check each argument
            for k, argument in self.xpmtype.arguments.items():
                value = self.values.get(k)
                if value is not None:
                    if isinstance(value, Config):
                        value.__xpm__.validate()
                elif argument.required:
                    if not argument.generator:
                        raise ValueError(
                            "Value %s is required but missing when building %s at %s"
                            % (k, self.xpmtype, self._initinfo)
                        )

            # Validate pre-tasks
            for pre_task in self.pre_tasks:
                pre_task.__xpm__.validate()

            # Validate init tasks
            for init_task in self.init_tasks:
                init_task.__xpm__.validate()

            # Use __validate__ method
            if hasattr(self.pyobject, "__validate__"):
                try:
                    self.pyobject.__validate__()
                except Exception:
                    logger.error(
                        "Error while validating %s at %s", self.xpmtype, self._initinfo
                    )
                    raise

    def seal(self, context: ConfigWalkContext):
        """Seals the object and generate values when needed

        Arguments:
            - context: the generation context
        """
        subconfigs = [
            v.__xpm__
            for v in self.values.values()
            if isinstance(v, Config) and v.__xpm__.task is None
        ]

        if any(v._has_generated_value for v in subconfigs):
            raise AttributeError("Cannot seal a configuration with generated values")

        class Sealer(ConfigWalk):
            def preprocess(self, config: ConfigMixin):
                return not config.__xpm__._sealed, config

            def postprocess(self, stub, config: ConfigMixin, values):
                # Generate values
                from experimaestro.generators import Generator

                for k, argument in config.__xpmtype__.arguments.items():
                    try:
                        if argument.generator:
                            if not isinstance(argument.generator, Generator):
                                # Don't set if already set
                                if config.__xpm__.values.get(k) is not None:
                                    continue
                                value = argument.generator()
                            else:
                                # Generate a value
                                sig = inspect.signature(argument.generator)
                                if len(sig.parameters) == 0:
                                    value = argument.generator()
                                elif len(sig.parameters) == 2:
                                    value = argument.generator(self.context, config)
                                else:
                                    assert (
                                        False
                                    ), "generator has either two parameters (context and config), or none"
                            config.__xpm__.set(k, value, bypass=True)
                            config.__xpm__._has_generated_value = True
                        else:
                            value = config.__xpm__.values.get(k)
                    except Exception:
                        logger.error(
                            "While setting %s of %s", argument.name, config.__xpmtype__
                        )
                        raise

                    # Propagate the generated value flag
                    if (
                        (value is not None)
                        and isinstance(value, ConfigMixin)
                        and value.__xpm__._has_generated_value
                    ):
                        self._has_generated_value = True

                config.__xpm__._sealed = True

        Sealer(context, recurse_task=True)(self.pyobject)

    def __unseal__(self):
        """Unseal this configuration and its descendant

        Internal API - do not use
        """
        context = ConfigWalkContext()

        class Unsealer(ConfigWalk):
            def preprocess(self, config: Config):
                return config.__xpm__._sealed, config

            def postprocess(self, stub, config: Config, values):
                config.__xpm__._sealed = False
                config.__xpm__._identifier = None

        Unsealer(context, recurse_task=True)(self.pyobject)

    def collect_pre_tasks(self) -> Iterator["Config"]:
        context = ConfigWalkContext()
        pre_tasks: Dict[int, "Config"] = {}

        class PreTaskCollect(ConfigWalk):
            def preprocess(self, config: Config):
                # Do not cross tasks
                return not isinstance(config.__xpm__, Task), config

            def postprocess(self, stub, config: Config, values):
                pre_tasks.update(
                    {id(pre_task): pre_task for pre_task in config.__xpm__.pre_tasks}
                )

        PreTaskCollect(context, recurse_task=True)(self.pyobject)
        return pre_tasks.values()

    def identifiers(self, only_raw: bool):
        """Computes the unique identifier"""
        from ..identifier import IdentifierComputer, Identifier

        raw_identifier = self._raw_identifier
        full_identifier = self._full_identifier

        # Computes raw identifier if needed
        if raw_identifier is None or not self._sealed:
            # Get the main identifier
            raw_identifier = IdentifierComputer.compute(self.pyobject)
            if self._sealed:
                self._raw_identifier = raw_identifier

        if only_raw:
            return raw_identifier, full_identifier

        # OK, let's compute the full identifier
        if full_identifier is None or not self._sealed:
            # Compute the full identifier by including the pre-tasks
            hasher = hashlib.sha256()
            hasher.update(raw_identifier.all)
            pre_tasks_ids = [
                pre_task.__xpm__.raw_identifier.all
                for pre_task in self.collect_pre_tasks()
            ]
            for task_id in sorted(pre_tasks_ids):
                hasher.update(task_id)

            # Adds init tasks
            if self.init_tasks:
                hasher.update(IdentifierComputer.INIT_TASKS)
                for init_task in self.init_tasks:
                    hasher.update(init_task.__xpm__.raw_identifier.all)

            full_identifier = Identifier(hasher.digest())
            full_identifier.has_loops = raw_identifier.has_loops

            # Only cache the identifier if sealed
            if self._sealed:
                self._full_identifier = full_identifier

        return raw_identifier, full_identifier

    @property
    def raw_identifier(self) -> "Identifier":
        """Computes the unique identifier (without task modifiers)"""
        raw_identifier, _ = self.identifiers(True)
        return raw_identifier

    @property
    def full_identifier(self) -> "Identifier":
        """Computes the unique identifier (with task modifiers)"""
        _, full_identifier = self.identifiers(False)
        return full_identifier

    identifier = full_identifier
    """Deprecated: use full_identifier"""

    def dependency(self):
        """Returns a dependency"""
        from experimaestro.scheduler import JobDependency

        assert self.job, f"{self.xpmtype} is a task but was not submitted"
        return JobDependency(self.job)

    def updatedependencies(
        self,
        dependencies: Set["experimaestro.dependencies.Dependency"],
        path: List[str],
        taskids: Set[int],
    ):
        # Add pre-tasks
        for pre_task in self.pre_tasks:
            pre_task.__xpm__.updatedependencies(
                dependencies, path + ["__pre_tasks__"], taskids
            )

        # Add initialization tasks
        for init_task in self.init_tasks:
            init_task.__xpm__.updatedependencies(
                dependencies, path + ["__init_tasks__"], taskids
            )

        # Check for an associated task (and not loaded)
        if self.task and not self.loaded:
            if id(self.task) not in taskids:
                taskids.add(id(self.task))
                dependencies.add(self.task.__xpm__.dependency())
        else:
            # Look arguments
            for argument, value in self.xpmvalues():
                try:
                    if value is not None:
                        updatedependencies(
                            dependencies, value, path + [argument.name], taskids
                        )
                except Exception:
                    logger.error("While setting %s", path + [argument.name])
                    raise

    def apply_submit_hooks(self, job: "Job", launcher: "Launcher"):
        """Apply configuration hooks"""
        context = ConfigWalkContext()

        class HookGatherer(ConfigWalk):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.hooks = set()

            def postprocess(self, stub, config: "Config", values: Dict[str, Any]):
                self.hooks.update(config.__xpmtype__.submit_hooks)

        gatherer = HookGatherer(context, recurse_task=False)
        gatherer(self.pyobject)

        # Apply hooks
        for hook in gatherer.hooks:
            hook.process(job, launcher)

    def validate_and_seal(self, context: ConfigWalkContext):
        try:
            self.validate()
        except Exception as e:
            logger.error(
                "Error while validating object of type %s, defined %s",
                self.xpmtype,
                self._initinfo,
            )
            raise e

        # Now, seal the object
        self.seal(context)

    def watch_output(self, method, callback):
        """Watch the task output linked with a given method

        :param method: The method to watch
        :param callback: The callback
        """
        watched = WatchedOutput(
            self, method.__self__, method.__name__, method, callback
        )
        self.watched_outputs.append(watched)
        if self.job:
            self.job.watch_output(watched)

    def on_completed(self, callback: Callable[[], None]):
        """Call a method when the task is completed successfully

        :param callback: _description_
        """
        from ..callbacks import TaskEventListener

        TaskEventListener.on_completed(self, callback)

    def submit(
        self,
        workspace: "Workspace",
        launcher: "Launcher",
        *,
        run_mode=None,
        init_tasks: List["LightweightTask"] = [],
    ):
        from experimaestro.scheduler import experiment, JobContext
        from experimaestro.scheduler.workspace import RunMode
        from ..callbacks import TaskEventListener

        # --- Prepare the object

        if self.job:
            raise Exception("task %s was already submitted" % self)
        if not self.xpmtype.task:
            raise ValueError("%s is not a task" % self.xpmtype)

        # --- Submit the job

        # Sets the init tasks
        self.init_tasks = init_tasks

        # Creates a new job
        self.job = self.xpmtype.task(
            self.pyobject, launcher=launcher, workspace=workspace, run_mode=run_mode
        )

        # Validate the object
        job_context = JobContext(self.job)
        self.validate_and_seal(job_context)

        # --- Workspace

        workspace = workspace or (
            experiment.CURRENT.workspace if experiment.CURRENT else None
        )

        # --- Launcher

        launcher = (
            launcher
            or (workspace and workspace.launcher)
            or (experiment.CURRENT and experiment.CURRENT.workspace.launcher)
        )

        if launcher:
            launcher.onSubmit(self.job)

        # Apply submit hooks
        self.apply_submit_hooks(self.job, launcher)

        # Add job dependencies
        self.updatedependencies(self.job.dependencies, [], set([id(self.pyobject)]))

        # Add predefined dependencies
        self.job.dependencies.update(self.dependencies)

        run_mode = (
            workspace.run_mode if run_mode is None else run_mode
        ) or RunMode.NORMAL
        if run_mode == RunMode.NORMAL:
            TaskEventListener.connect(experiment.CURRENT)
            other = experiment.CURRENT.submit(self.job)
            if other:
                # Just returns the other task
                return other.config.__xpm__._taskoutput
        else:
            # Show a warning
            if run_mode == RunMode.GENERATE_ONLY:
                experiment.CURRENT.prepare(self.job)

            # Check if job is done
            tags = ", ".join(f"{k}={v}" for k, v in self.job.tags.items())
            s = f"""Simulating {self.job.relpath} {f"({tags})" if tags else ""}"""

            color = "white"
            if self.job.workspace is not None:
                if self.job.donepath.is_file():
                    color = "light_green"
                    cprint(f"[done] {s}", color, file=sys.stderr)
                elif self.job.failedpath.is_file():
                    color = "light_red"
                    cprint(f"[failed] {s}", color, file=sys.stderr)
                elif self.job.pidpath.is_file():
                    color = "blue"
                    cprint(f"[running] {s}", color, file=sys.stderr)
                else:
                    color = "light_blue"
                    cprint(f"[not run] {s}", color, file=sys.stderr)

                if launcher:
                    cprint(f"   [Launcher] {launcher}", color, file=sys.stderr)

                if not self.job.dependencies:
                    cprint("   [No dependencies]", color, file=sys.stderr)

                for dep in self.job.dependencies:
                    cprint(f"   [Dependency] {dep}", color, file=sys.stderr)

                print(file=sys.stderr)  # noqa: T201

        # Mark this configuration also
        self.task = self.pyobject

        if hasattr(self.pyobject, "task_outputs"):
            self._taskoutput = self.pyobject.task_outputs(self.mark_output)
        else:
            self._taskoutput = self.task = self.pyobject

        return self._taskoutput

    def mark_output(self, config: "Config"):
        """Sets a dependency on the job"""
        assert not isinstance(config, Task), "Cannot set a dependency on a task"
        assert isinstance(
            config, ConfigMixin
        ), "Only configurations can be marked as dependent on a task"
        config.__xpm__.task = self.pyobject
        return config

    # --- Serialization

    @staticmethod
    def _outputjsonvalue(value, context):
        """Serialize a value"""
        if value is None:
            return None

        elif isinstance(value, (list, tuple)):
            return [ConfigInformation._outputjsonvalue(el, context) for el in value]

        elif isinstance(value, dict):
            return {
                name: ConfigInformation._outputjsonvalue(el, context)
                for name, el in value.items()
            }

        elif isinstance(value, Path):
            return {"type": "path", "value": str(value)}

        elif isinstance(value, SerializedPath):
            return {
                "type": "path.serialized",
                "value": str(value.path),
                "is_folder": value.is_folder,
            }

        elif isinstance(value, (int, float, str)):
            return value

        elif isinstance(value, Enum):
            return {
                "type": "enum",
                "module": value.__class__.__module__,
                "enum": value.__class__.__qualname__,
                "value": value.name,
            }

        elif isinstance(value, Config):
            return {
                "type": "python",
                "value": id(value),
            }

        raise NotImplementedError("Cannot serialize objects of type %s", type(value))

    def __get_objects__(
        self,
        objects: List[Dict],
        context: SerializationContext,
    ) -> List[Dict]:
        """Returns the list of objects necessary to deserialize ourself

        :param objects: the already output objects
        :param context: The serialization context (e.g. useful to create files)
        """

        # Skip if already serialized
        if id(self.pyobject) in context.serialized:
            return objects

        context.serialized.add(id(self.pyobject))

        # Serialize sub-objects
        for argument, value in self.xpmvalues():
            if value is not None:
                ConfigInformation.__collect_objects__(value, objects, context)

        # Adds task
        if self.task is not None and self.task is not self:
            ConfigInformation.__collect_objects__(self.task, objects, context)

        # Serialize pre-tasks
        ConfigInformation.__collect_objects__(self.pre_tasks, objects, context)

        # Serialize initialization tasks
        ConfigInformation.__collect_objects__(self.init_tasks, objects, context)

        # Serialize ourselves
        state_dict = {
            "id": id(self.pyobject),
            "module": self.xpmtype._module,
            "type": self.xpmtype.value_type.__qualname__,
            "typename": self.xpmtype.name(),
            "identifier": self.identifier.state_dict(),
        }

        # Add pre/init tasks
        if self.pre_tasks:
            state_dict["pre-tasks"] = [id(pre_task) for pre_task in self.pre_tasks]
        if self.init_tasks:
            state_dict["init-tasks"] = [id(init_task) for init_task in self.init_tasks]

        if self.meta:
            state_dict["meta"] = self.meta

        if self.task is not None:
            state_dict["task"] = id(self.task)

        if not self.xpmtype._package:
            state_dict["file"] = str(self.xpmtype._file)

        # Serialize identifier and typename
        jsonfields = state_dict["fields"] = {}
        for argument, value in self.xpmvalues():
            with context.push(argument.name) as var_path:
                if argument.is_data and value is not None:
                    assert isinstance(
                        value, Path
                    ), f"Data arguments should be paths (type is {type(value)})"
                    value = context.serialize(var_path, value)

                jsonfields[argument.name] = ConfigInformation._outputjsonvalue(
                    value, context
                )

        objects.append(state_dict)
        return objects

    @staticmethod
    def __collect_objects__(value, objects: List[Dict], context: SerializationContext):
        """Serialize all needed configuration objects, looking at sub
        configurations if necessary"""
        if value is None:
            return

        if isinstance(value, Config):
            value.__xpm__.__get_objects__(objects, context)
        elif isinstance(value, (list, tuple)):
            for el in value:
                ConfigInformation.__collect_objects__(el, objects, context)
        elif isinstance(value, dict):
            for el in value.values():
                ConfigInformation.__collect_objects__(el, objects, context)
        elif isinstance(value, (Path, int, float, str, Enum)):
            pass
        else:
            raise NotImplementedError(
                "Cannot serialize objects of type %s", type(value)
            )

    def outputjson(self, out: io.TextIOBase, context: SerializationContext):
        """Outputs the json of this object

        The format is an array of objects
        {
            "tags: [ LIST_OF_TAGS ],
            "workspace": FOLDERPATH,
            "version": 2,
            "objects": [
                {
                    "id": <ID of the object>,
                    "filename": <filename>, // if in a file
                    "module": <module>, // if in a module
                    "type": <type>, // the type within the module or file
                    "fields":
                        { "key":  {"type": <type>, "value": <value>} }
                }
            ]

        }

        <type> is either a base type or a "python"

        The last object is the one that is serialized

        Arguments:
            out {io.TextIOBase} -- The output stream
            context {[type]} -- the command context
        """
        json.dump(
            {
                "workspace": str(context.workspace.path.absolute()),
                "tags": {key: value for key, value in self.tags().items()},
                "version": 2,
                "objects": self.__get_objects__([], context),
            },
            out,
        )

    def __json__(self) -> str:
        """Returns the JSON representation of the object itself"""
        return json.dumps(self.__get_objects__([], SerializationContext()))

    def serialize(self, save_directory: Path):
        """Serialize the configuration and its data files into a directory"""
        context = SerializationContext(save_directory=save_directory)

        with (save_directory / "definition.json").open("wt") as out:
            objects = self.__get_objects__([], context)
            json.dump(objects, out)

    @staticmethod
    def deserialize(
        path: Union[str, Path, SerializedPathLoader],
        as_instance: bool = False,
        return_tasks: bool = False,
    ) -> "Config":
        """Deserialize a configuration

        :param path: The filesystem Path to use, or a way to download the
            information through a function taking two arguments
        :param as_instance: Return an instance
        :return: a Config object, its instance or a tuple (instance, init_tasks) is return_tasks is True
        """
        # Load
        assert not (
            as_instance and return_tasks
        ), "Cannot set as_instance and return_tasks to True"
        if callable(path):
            data_loader = path
        else:
            path = Path(path)

            def data_loader(s: Union[Path, str, SerializedPath]):
                if isinstance(s, SerializedPath):
                    return path / Path(s.path)
                return path / Path(s)

        with data_loader("definition.json").open("rt") as fh:
            config = json.load(fh)

        return ConfigInformation.fromParameters(
            config,
            as_instance=as_instance,
            data_loader=data_loader,
            return_tasks=return_tasks,
        )

    @staticmethod
    def _objectFromParameters(value: Any, objects: Dict[str, Any]):
        # A list
        if isinstance(value, list):
            return [ConfigInformation._objectFromParameters(x, objects) for x in value]

        # A dictionary
        if isinstance(value, dict):
            if "type" not in value:
                # Just a plain dictionary
                return {
                    ConfigInformation._objectFromParameters(
                        key, objects
                    ): ConfigInformation._objectFromParameters(value, objects)
                    for key, value in value.items()
                }

            # The value is an object (that has been serialized first)
            if value["type"] == "python":
                return objects[value["value"]]

            # A path
            if value["type"] == "path":
                return Path(value["value"])

            if value["type"] == "path.serialized":
                return SerializedPath(value["value"], value["is_folder"])

            if value["type"] == "enum":
                module = importlib.import_module(value["module"])
                enumClass = getqualattr(module, value["enum"])
                return enumClass[value["value"]]

            raise Exception("Unhandled type: %s", value["type"])

        # Just a simple value
        return value

    @overload
    @staticmethod
    def fromParameters(
        definitions: List[Dict],
        as_instance=True,
        save_directory: Optional[Path] = None,
        discard_id: bool = False,
    ) -> "ConfigMixin":
        ...

    @overload
    @staticmethod
    def fromParameters(
        definitions: List[Dict],
        as_instance=False,
        return_tasks=True,
        save_directory: Optional[Path] = None,
        discard_id: bool = False,
    ) -> Tuple["Config", List["LightweightTask"]]:
        ...

    @overload
    @staticmethod
    def fromParameters(
        definitions: List[Dict],
        as_instance=False,
        save_directory: Optional[Path] = None,
        discard_id: bool = False,
    ) -> "Config":
        ...

    @staticmethod
    def load_objects(  # noqa: C901
        definitions: List[Dict],
        as_instance=True,
        data_loader: Optional[SerializedPathLoader] = None,
        discard_id: bool = False,
    ):
        """Load the objects"""
        o = None
        objects = {}
        import experimaestro.taskglobals as taskglobals
        from ..identifier import Identifier

        # Loop over all the definitions and create objects
        for definition in definitions:
            module_name = definition["module"]

            # Avoids problem when runing module
            if module_name == "__main__":
                module_name = "_main_"

            if "file" in definition:
                path = definition["file"]
                with add_to_path(str(Path(path).parent)):
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
            else:
                try:
                    logger.debug("Importing module %s", definition["module"])
                    mod = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    # More hints on the nature of the error
                    logging.warning(
                        "(1) Either the python path is wrong – %s", ":".join(sys.path)
                    )
                    logging.warning("(2) There is not __init__.py in your module")
                    raise

            cls = getqualattr(mod, definition["type"])

            # Creates an object (or a config)
            if as_instance:
                o = cls.__new__(cls)
            else:
                o = cls.XPMConfig.__new__(cls.XPMConfig)
            assert definition["id"] not in objects, "Duplicate id %s" % definition["id"]
            objects[definition["id"]] = o

        # Now that objects have been created, fill in the fields
        for definition in definitions:
            o = objects[definition["id"]]
            xpmtype = o.__getxpmtype__()  # type: ObjectType

            # If instance...
            if as_instance:
                # ... calls the parameter-less initialization
                o.__init__()

                # ... sets potentially useful properties
                if "typename" in definition:
                    o.__xpmtypename__ = definition["typename"]
                    if not discard_id:
                        o.__xpmidentifier__ = Identifier.from_state_dict(
                            definition["identifier"]
                        )

                    if "." in o.__xpmtypename__:
                        _, name = o.__xpmtypename__.rsplit(".", 1)
                    else:
                        name = o.__xpmtypename__

                    if taskglobals.Env.instance().wspath is not None:
                        basepath = (
                            taskglobals.Env.instance().wspath
                            / "jobs"
                            / o.__xpmtypename__.lower()
                            / o.__xpmidentifier__.all.hex().lower()
                        )
                        o.__xpm_stdout__ = basepath / f"{name.lower()}.out"
                        o.__xpm_stderr__ = basepath / f"{name.lower()}.err"
            else:
                o.__init__()
                xpminfo = o.__xpm__  # type: ConfigInformation
                xpminfo.loaded = True

                meta = definition.get("meta", None)
                if meta:
                    xpminfo._meta = meta
                if xpminfo.xpmtype.task is not None:
                    xpminfo.job = object()

            # Set the fields
            for name, value in definition["fields"].items():
                v = ConfigInformation._objectFromParameters(value, objects)

                # Transform serialized paths arguments
                argument = xpmtype.arguments[name]
                if argument.is_data and v is not None:
                    if isinstance(v, SerializedPath):
                        if data_loader is None:
                            v = v.path
                        else:
                            v = data_loader(v)
                    else:
                        assert isinstance(v, Path), "Excepted Path, got {type(v)}"

                if as_instance:
                    # Unwrap the value if needed
                    setattr(o, name, v)

                    assert (
                        getattr(o, name) is v
                    ), f"Problem with deserialization {name} of {o.__class__}"
                else:
                    o.__xpm__.set(name, v, bypass=True)

            if as_instance:
                # Calls post-init
                o.__post_init__()

            else:
                # Sets pre-tasks
                o.__xpm__.pre_tasks = [
                    objects[pre_task_id]
                    for pre_task_id in definition.get("pre-tasks", [])
                ]

                if task_id := definition.get("task", None):
                    o.__xpm__.task = objects[task_id]

                # Seal and set the identifier
                if not discard_id:
                    xpminfo._identifier = Identifier.from_state_dict(
                        definition["identifier"]
                    )
                xpminfo._sealed = True

        return objects

    @staticmethod
    def fromParameters(  # noqa: C901
        definitions: List[Dict],
        as_instance=True,
        data_loader: Optional[SerializedPathLoader] = None,
        discard_id: bool = False,
        return_tasks: bool = False,
    ):
        # Get the objects
        objects = ConfigInformation.load_objects(
            definitions,
            as_instance=as_instance,
            data_loader=data_loader,
            discard_id=discard_id,
        )

        # Get the last one
        o = objects[definitions[-1]["id"]]

        # Run pre-task (or returns them)
        if as_instance or return_tasks:
            # Collect pre-tasks (just once)
            completed_pretasks = set()
            pre_tasks = []
            for definition in definitions:
                for pre_task_id in definition.get("pre-tasks", []):
                    if pre_task_id not in completed_pretasks:
                        completed_pretasks.add(pre_task_id)
                        pre_tasks.append(objects[pre_task_id])

            # Collect init tasks
            init_tasks = []
            for init_task_id in definitions[-1].get("init-tasks", []):
                init_task = objects[init_task_id]
                init_tasks.append(init_task)

            if as_instance:
                for pre_task in pre_tasks:
                    logger.info("Executing pre-task %s", type(pre_task))
                    pre_task.execute()
                for init_task in init_tasks:
                    logger.info("Executing init task %s", type(init_task))
                    init_task.execute()
            else:
                return o, pre_tasks, pre_task + init_tasks

        return o

    class FromPython(ConfigWalk):
        def __init__(self, context: ConfigWalkContext, *, objects: ObjectStore = None):
            super().__init__(context)
            self.objects = ObjectStore() if objects is None else objects
            self.pre_tasks = {}

        def preprocess(self, config: "Config"):
            if self.objects.is_constructed(id(config)):
                return False, self.objects.retrieve(id(config))
            return True, None

        def stub(self, config: "Config"):
            o = self.objects.retrieve(id(config))

            if o is None:
                # Creates an object (and not a config)
                o = config.__xpmtype__.value_type()

                # Store in cache
                self.objects.add_stub(id(config), o)

            return o

        def postprocess(self, stub, config: "Config", values: Dict[str, Any]):
            # Copy values from the
            for key, value in values.items():
                setattr(stub, key, value)

            # Call __post_init__
            stub.__post_init__()

            # Gather pre-tasks
            for pre_task in config.__xpm__.pre_tasks:
                self.pre_tasks[id(pre_task)] = self.stub(pre_task)

            self.objects.set_constructed(id(config))
            return stub

    def fromConfig(self, context: ConfigWalkContext, *, objects: ObjectStore = None):
        """Generate an instance given the current configuration"""

        # Validate and seal
        self.validate()
        self.seal(context)

        processor = ConfigInformation.FromPython(context, objects=objects)
        last_object = processor(self.pyobject)

        # Execute pre-tasks
        for pre_task in processor.pre_tasks.values():
            pre_task.execute()

        return last_object

    def add_dependencies(self, *dependencies):
        self.dependencies.extend(dependencies)


def clone(v):
    """Clone a value"""
    if v is None:
        return v

    if isinstance(v, (str, float, int, Path)):
        return v

    if isinstance(v, list):
        return [clone(x) for x in v]

    if isinstance(v, dict):
        return {clone(key): clone(value) for key, value in v.items()}

    if isinstance(v, Enum):
        return v

    if isinstance(v, tuple):
        return tuple(clone(x) for x in v)

    if isinstance(v, Config):
        # Create a new instance
        kwargs = {
            argument.name: clone(value)
            for argument, value in v.__xpm__.xpmvalues()
            if argument.generator is None and not argument.constant
        }

        config = type(v)(**kwargs)
        return config

    raise NotImplementedError("Clone not implemented for type %s" % type(v))


class ConfigMixin:
    """Class for configuration objects"""

    __xpmtype__: ObjectType
    """The associated XPM type"""

    __xpm__: ConfigInformation
    """The __xpm__ object contains all instance specific information about a
    configuration/task"""

    def __init__(self, **kwargs):
        """Initialize the configuration with the given parameters"""

        # Add configuration
        xpmtype = self.__xpmtype__

        # Create the config information object
        xpm = ConfigInformation(self)

        # Get the line where the object was created (error reporting)
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        xpm._initinfo = "%s:%s" % (str(Path(caller.filename).absolute()), caller.lineno)

        self.__xpm__ = xpm

        # Initialize with default arguments (or None)
        for name, value in xpmtype.arguments.items():
            if name not in kwargs:
                if value.default is not None:
                    self.__xpm__.set(name, clone(value.default), bypass=True)
                elif not value.required:
                    self.__xpm__.set(name, None, bypass=True)

        # Initialize with arguments
        for name, value in kwargs.items():
            # Check if argument is OK
            if name not in xpmtype.arguments:
                attribute = getattr(self.__class__, name, None)
                if isinstance(attribute, DeprecatedAttribute):
                    attribute.__set__(self, value)
                    continue
                raise ValueError("%s is not an argument for %s" % (name, xpmtype))

            # Special case of a tagged value
            if isinstance(value, TaggedValue):
                value = value.value
                self.__xpm__._tags[name] = value

            # Really set the value
            xpm.set(name, value)

    def __repr__(self):
        return f"Config[{self.__xpmtype__.identifier}]"

    def __str__(self):
        params = ", ".join(
            [f"{key}={value}" for key, value in self.__xpm__.values.items()]
        )
        return (
            f"{self.__xpmtype__.value_type.__module__}."
            f"{self.__xpmtype__.value_type.__qualname__}({params})"
        )

    def tag(self, name, value):
        self.__xpm__.addtag(name, value)
        return self

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        for argument, value in self.__xpm__.xpmvalues():
            if value != getattr(other, argument.name, None):
                return False
        return True

    def __arguments__(self):
        """Returns a map containing all the arguments"""
        return {arg.name: value for arg, value in self.__xpm__.xpmvalues()}

    def tags(self):
        """Returns the tag associated with this object (and below)"""
        return self.__xpm__.tags()

    def add_dependencies(self, *dependencies):
        """Adds tokens to the task"""
        self.__xpm__.add_dependencies(*dependencies)
        return self

    def instance(
        self,
        context: ConfigWalkContext = None,
        *,
        objects: ObjectStore = None,
        keep: bool = True,
    ) -> T:
        """Return an instance with the current values

        :param context: The context when computing the instance
        :param objects: The previously built objects (so that we avoid
            re-creating instances of past configurations)
        :param keep: register a configuration in the __config__ field of the
            instance
        """
        if context is None:
            from experimaestro.xpmutils import EmptyContext

            context = EmptyContext()
        else:
            assert isinstance(
                context, ConfigWalkContext
            ), f"{context.__class__} is not an instance of ConfigWalkContext"

        instance = self.__xpm__.fromConfig(context, objects=objects)  # type: ignore
        if keep:
            instance.__config__ = self
        return instance

    def submit(
        self,
        *,
        workspace=None,
        launcher=None,
        run_mode: "RunMode" = None,
        init_tasks: List["LightweightTask"] = [],
    ):
        """Submit this task

        :param workspace: the workspace, defaults to None
        :param launcher: The launcher, defaults to None
        :param run_mode: Run mode (if None, uses the workspace default)
        :return: an object object
        """
        return self.__xpm__.submit(
            workspace, launcher, run_mode=run_mode, init_tasks=init_tasks
        )

    def stdout(self):
        return self.__xpm__.job.stdout

    def stderr(self):
        return self.__xpm__.job.stderr

    def wait(self):
        return self.__xpm__.job.wait()

    @property
    def job(self):
        return self.__xpm__.job

    @property
    def jobpath(self):
        if self.__xpm__.job:
            return self.__xpm__.job.jobpath
        raise AssertionError("Cannot ask the job path of a non submitted task")

    def copy(self):
        """Returns a copy of this configuration (ignores other non parameters
        attributes)"""
        return clone(self)

    def add_pretasks(self, *tasks: "LightweightTask"):
        assert all(
            [isinstance(task, ConfigMixin) for task in tasks]
        ), "One of the parameters is not a configuration object"
        assert all(
            [isinstance(task, LightweightTask) for task in tasks]
        ), "One of the pre-tasks are not lightweight tasks"
        if self.__xpm__._sealed:
            raise SealedError("Cannot add pre-tasks to a sealed configuration")
        self.__xpm__.pre_tasks.extend(tasks)
        return self

    def add_pretasks_from(self, *configs: "Config"):
        assert all(
            [isinstance(config, ConfigMixin) for config in configs]
        ), "One of the parameters is not a configuration object"
        for config in configs:
            self.add_pretasks(*config.__xpm__.pre_tasks)
        return self

    @property
    def pre_tasks(self) -> List["LightweightTask"]:
        """Access pre-tasks"""
        return self.__xpm__.pre_tasks

    def copy_dependencies(self, other: "Config"):
        """Add all the dependencies from other configuration"""

        # Add task dependency
        if other.__xpm__.task is not None:
            assert self.__xpm__.task is None
            self.__xpm__.task = other.__xpm__.task

        # Add other dependencies
        self.__xpm__.add_dependencies(*other.__xpm__.dependencies)


class Config:
    """Base type for all objects in python interface"""

    __xpmid__: ClassVar[Optional[str]]
    """Optional configuration ID, mostly useful when moving a class to another
    package to avoid changes in computed task identifiers"""

    __xpmtype__: ClassVar[ObjectType]
    """The object type holds all the information about a specific subclass
    experimaestro metadata"""

    @classproperty
    def XPMConfig(cls):
        if issubclass(cls, ConfigMixin):
            return cls
        return cls.__getxpmtype__().config_type

    @classproperty
    def C(cls):
        """Alias for XPMConfig"""
        return cls.XPMConfig

    @classmethod
    def __getxpmtype__(cls) -> "ObjectType":
        """Get (and create if necessary) the Object type associated
        with thie Config object"""
        xpmtype = cls.__dict__.get("__xpmtype__", None)
        if xpmtype is None:
            from experimaestro.core.types import ObjectType

            # Will set __xpmtype__
            try:
                return ObjectType(cls)
            except Exception:
                logger.error("Error while creating object type for %s", cls)
                raise
        return xpmtype

    def __validate__(self):
        """Validate the values"""
        pass

    def __post_init__(self):
        """Called after the object  __init__() and with properties set"""
        # Default implementation is to do nothing
        pass

    def __json__(self):
        """Returns a JSON version of the object (if possible)"""
        return self.__xpm__.__json__()

    def __identifier__(self) -> "Identifier":
        return self.__xpm__.full_identifier

    def add_pretasks(self, *tasks: "LightweightTask"):
        """Add pre-tasks"""
        raise AssertionError("This method can only be used during configuration")

    def add_pretasks_from(self, *configs: "Config"):
        """Add pre-tasks from the listed configurations"""
        raise AssertionError(
            "The 'add_pretasks_from' can only be used during configuration"
        )

    def copy_dependencies(self, other: "Config"):
        """Add pre-tasks from the listed configurations"""
        raise AssertionError(
            "The 'copy_dependencies' method can only be used during configuration"
        )

    @property
    def pre_tasks(self) -> List["LightweightTask"]:
        """Access pre-tasks"""
        raise AssertionError("Pre-tasks can be accessed only during configuration")

    def register_task_output(self, method, *args, **kwargs):
        # Determine the path for this...
        path = taskglobals.Env.instance().xpm_path / "task-outputs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = json.dumps(
            {
                "key": f"{self.__xpmidentifier__}/{method.__name__}",
                "args": args,
                "kwargs": kwargs,
            }
        )
        with path.open("at") as fp:
            fp.writelines([data, "\n"])
            fp.flush()


class LightweightTask(Config):
    """A task that can be run before or after a real task to modify its behaviour"""

    def execute(self):
        raise NotImplementedError()


class Task(LightweightTask):
    """Base class for tasks"""

    __tags__: Dict[str, str]
    """Tags associated with class"""

    def submit(self):
        raise AssertionError("This method can only be used during configuration")

    def watch_output(self, method, callback):
        """Sets up a callback

        :param method: a method within a configuration
        :param callback: the callback
        """
        self.__xpm__.watch_output(method, callback)

    def on_completed(self, callback: Callable[[], None]):
        self.__xpm__.on_completed(callback)


# --- Utility functions


def copyconfig(config: Config, **kwargs):
    """Copy a configuration

    Useful to modify a configuration that can be potentially
    wrapped into a task output (i.e., the configuration can be
    a task output). If the configuration is sealed, the copy
    will be unsealed.

    :param config: _description_
    :param kwargs: Modify the configuration by assigning the values

    :return: _description_
    """

    # Builds a new configuration object
    copy = config.__class__()

    fullkwargs = {name: value for name, value in config.__xpm__.values.items()}
    fullkwargs.update(kwargs)
    for name, value in fullkwargs.items():
        copy.__xpm__.set(name, value, True)

    # Remove generated attributes
    for argument, value in copy.__xpm__.xpmvalues(generated=True):
        if argument.generator is not None and value is not None:
            del copy.__xpm__.values[argument.name]

    return copy


def setmeta(config: Config, flag: bool):
    """Flags the configuration as a meta-parameter"""
    config.__xpm__.set_meta(flag)
    return config


def cache(fn, name: str):
    def __call__(config, *args, **kwargs):
        import experimaestro.taskglobals as taskglobals

        # Get path and create directory if needed
        hexid = config.__xpmidentifier__  # type: Identifier
        typename = config.__xpmtypename__  # type: str
        dir = taskglobals.Env.instance().wspath / "config" / typename / hexid.all.hex()

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        path = dir / name
        ipc_lock = fasteners.InterProcessLock(path.with_suffix(path.suffix + ".lock"))
        with ipc_lock:
            r = fn(config, path, *args, **kwargs)
            return r

    return __call__
