"""Configuration and tasks"""

from functools import cached_property
import json
from termcolor import cprint
import os
from pathlib import Path
import hashlib
import struct
import io
import fasteners
from enum import Enum
import inspect
import importlib
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    overload,
    TYPE_CHECKING,
)
import sys
import experimaestro
from experimaestro.utils import logger
from contextlib import contextmanager
from experimaestro.core.types import DeprecatedAttribute, ObjectType
from .context import SerializationContext, SerializedPath, SerializedPathLoader

if TYPE_CHECKING:
    from experimaestro.scheduler.workspace import RunMode
    from experimaestro.launchers import Launcher
    from experimaestro.scheduler import Workspace

T = TypeVar("T", bound="Config")


class Identifier:
    def __init__(self, main: bytes, sub: Optional[bytes] = None):
        self.main = main
        self.sub = sub

    @cached_property
    def all(self):
        if self.sub is None:
            return self.main

        h = hashlib.sha256()
        h.update(self.main)
        h.update(self.sub)
        return h.digest()

    def state_dict(self):
        if self.sub is None:
            return self.main.hex()

        return {"main": self.main.hex(), "sub": self.sub.hex()}

    def __eq__(self, other: "Identifier"):
        return self.main == other.main and self.sub == other.sub

    @staticmethod
    def from_state_dict(data: Union[Dict[str, str], str]):
        if isinstance(data, str):
            return Identifier(bytes.fromhex(data))

        return Identifier(bytes.fromhex(data["main"]), bytes.fromhex(data["sub"]))


def is_ignored(value):
    """Returns True if the value should be ignored by itself"""
    return value is not None and isinstance(value, Config) and (value.__xpm__.meta)


def remove_meta(value):
    """Cleanup a dict/list by removing ignored values"""
    if isinstance(value, list):
        return [el for el in value if not is_ignored(el)]
    if isinstance(value, dict):
        return {key: value for key, value in value.items() if not is_ignored(value)}
    return value


class HashComputer:
    """This class is in charge of computing a config/task identifier"""

    OBJECT_ID = b"\x00"
    INT_ID = b"\x01"
    FLOAT_ID = b"\x02"
    STR_ID = b"\x03"
    PATH_ID = b"\x04"
    NAME_ID = b"\x05"
    NONE_ID = b"\x06"
    LIST_ID = b"\x07"
    TASK_ID = b"\x08"
    DICT_ID = b"\x09"
    ENUM_ID = b"\x0a"

    def __init__(self, version=None):
        # Hasher for parameters
        self._hasher = hashlib.sha256()
        self._subhasher = None
        self._tasks = set()
        self.version = version or int(os.environ.get("XPM_HASH_COMPUTER", 2))

    def identifier(self) -> Identifier:
        main = self._hasher.digest()
        sub = self._subhasher.digest() if self._subhasher is not None else None
        return Identifier(main, sub)

    def _hashupdate(self, bytes, subparam):
        if subparam:
            # If subparam, creates a specific sub-hasher
            if self._subhasher is None:
                self._subhasher = hashlib.sha256()
            self._subhasher.update(bytes)
        else:
            self._hasher.update(bytes)

    def update(self, value, subparam=False, myself=False):
        """Update the hash

        :param value: The value to add to the hash
        :param subparam: True if this value is a sub-parameter, defaults to
            False
        :param myself: True if the value is the configuration for which we wish
            to compute the identifier, defaults to False
        :raises NotImplementedError: If the value cannot be processed
        """
        if value is None:
            self._hashupdate(HashComputer.NONE_ID, subparam=subparam)
        elif isinstance(value, float):
            self._hashupdate(HashComputer.FLOAT_ID, subparam=subparam)
            self._hashupdate(struct.pack("!d", value), subparam=subparam)
        elif isinstance(value, int):
            self._hashupdate(HashComputer.INT_ID, subparam=subparam)
            self._hashupdate(struct.pack("!q", value), subparam=subparam)
        elif isinstance(value, str):
            self._hashupdate(HashComputer.STR_ID, subparam=subparam)
            self._hashupdate(value.encode("utf-8"), subparam=subparam)
        elif isinstance(value, list):
            values = [el for el in value if not is_ignored(el)]
            self._hashupdate(HashComputer.LIST_ID, subparam=subparam)
            self._hashupdate(struct.pack("!d", len(values)), subparam=subparam)
            for x in values:
                self.update(x, subparam=subparam)
        elif isinstance(value, Enum):
            self._hashupdate(HashComputer.ENUM_ID, subparam=subparam)
            k = value.__class__
            self._hashupdate(
                f"{k.__module__}.{k.__qualname__ }:{value.name}".encode("utf-8"),
                subparam=subparam,
            )
        elif isinstance(value, dict):
            self._hashupdate(HashComputer.DICT_ID, subparam=subparam)
            items = [
                (key, value) for key, value in value.items() if not is_ignored(value)
            ]
            items.sort(key=lambda x: x[0])
            for key, value in items:
                self.update(key, subparam=subparam)
                self.update(value, subparam=subparam)
        elif isinstance(value, TaskOutput):
            # Add the task ID...
            self.update(value.__xpm__.task, subparam=subparam)

            # ... as well as the configuration
            if self.version > 1:
                self.update(value.__unwrap__(), subparam=subparam)

        # Handles configurations
        elif isinstance(value, Config):
            # Encodes the identifier
            self._hashupdate(HashComputer.OBJECT_ID, subparam=subparam)
            if not myself and self.version > 1:
                # Just use the object identifier
                value_id = value.__xpm__.identifier
                self._hashupdate(value_id.all, subparam=subparam)

                # If the config has sub-parameters, also update this way
                if value_id.sub is not None and not subparam:
                    self._hashupdate(value_id.sub, subparam=True)

                # And that's it!
                return

            xpmtype = value.__xpmtype__
            self._hashupdate(xpmtype.identifier.name.encode("utf-8"), subparam=subparam)

            # Process arguments (sort by name to ensure uniqueness)
            arguments = sorted(xpmtype.arguments.values(), key=lambda a: a.name)
            for argument in arguments:
                arg_subparam = subparam or argument.subparam

                # Ignored argument
                if argument.ignored:
                    argvalue = value.__xpm__.values.get(argument.name, None)

                    # ... unless meta is set to false
                    if (
                        argvalue is None
                        or not isinstance(argvalue, Config)
                        or (argvalue.__xpm__.meta is not False)
                    ):
                        continue

                if argument.generator:
                    continue

                # Argument value
                # Skip if the argument is not a constant, and
                # - optional argument: both value and default are None
                # - the argument value is equal to the default value
                argvalue = getattr(value, argument.name, None)
                if not argument.constant and (
                    (
                        not argument.required
                        and argument.default is None
                        and argvalue is None
                    )
                    or (
                        argument.default is not None
                        and argument.default == remove_meta(argvalue)
                    )
                ):
                    # No update if same value (and not constant)
                    continue

                if (
                    argvalue is not None
                    and isinstance(argvalue, Config)
                    and argvalue.__xpm__.meta
                ):
                    continue

                # Hash name
                self.update(argument.name, subparam=arg_subparam)

                # Hash value
                self._hashupdate(HashComputer.NAME_ID, subparam=arg_subparam)
                self.update(argvalue, subparam=arg_subparam)

        else:
            raise NotImplementedError("Cannot compute hash of type %s" % type(value))


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

    def add(task):
        if id(task) not in taskids:
            taskids.add(id(task))
            dependencies.add(task.__xpm__.dependency())

    if isinstance(value, Config):
        if value.__xpmtype__.task:
            add(value)
        else:
            value.__xpm__.updatedependencies(dependencies, path, taskids)
    elif isinstance(value, (list, set)):
        for el in value:
            updatedependencies(dependencies, el, path, taskids)
    elif isinstance(value, TaskOutput):
        dependencies.add(value.__xpm__.task.__xpm__.dependency())
    elif isinstance(value, (dict,)):
        for key, val in value.items():
            updatedependencies(dependencies, key, path, taskids)
            updatedependencies(dependencies, val, path, taskids)
    elif isinstance(value, (str, int, float, Path, Enum)):
        pass
    else:
        raise NotImplementedError("update dependencies for type %s" % type(value))


class TaggedValue:
    def __init__(self, value):
        self.value = value


@contextmanager
def add_to_path(p):
    """Temporarily add a path to sys.path"""
    import sys

    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


class GenerationContext:
    """Context when generating values in configurations"""

    @property
    def path(self):
        """Returns the path of the job directory"""
        raise NotImplementedError()

    def __init__(self):
        self._configpath = None

    @property
    def task(self):
        return None

    def currentpath(self) -> Path:
        """Returns the configuration folder"""
        if self._configpath:
            return self.path / self._configpath
        return self.path

    @contextmanager
    def push(self, key: str):
        """Push a new key to contextualize paths"""
        p = self._configpath
        try:
            self._configpath = (Path("out") if p is None else p) / key
            yield key
        finally:
            self._configpath = p


NOT_SET = object()


class ConfigProcessing:
    """Allows to perform an operation on all nested configurations"""

    def __init__(self, recurse_task=False):
        """

        Parameters:
            recurse_task: Recurse into linked tasks
        """
        self.recurse_task = recurse_task

        # Stores already visited nodes
        self.visited = {}

    def preprocess(self, config: "Config"):
        return True, None

    def postprocess(self, config: "Config", values: Dict[str, Any]):
        return config

    @contextmanager
    def list(self, i: int):
        yield i

    @contextmanager
    def map(self, k: str):
        yield k

    def __call__(self, x):
        if isinstance(x, Config):
            info = x.__xpm__  # type: ConfigInformation

            # Avoid loops
            xid = id(x)
            if xid in self.visited:
                return self.visited[xid]
            self.visited[xid] = NOT_SET

            # Pre-process
            flag, value = self.preprocess(x)

            if not flag:
                # Stop processing and returns value
                return value

            result = {}
            for arg, v in info.xpmvalues():
                if v is not None:
                    with self.map(arg.name):
                        result[arg.name] = self(v)

            processed = self.postprocess(x, result)
            self.visited[xid] = processed
            return processed

        if isinstance(x, list):
            result = []
            for i, sv in enumerate(x):
                with self.list(i):
                    result.append(self(sv))
            return result

        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                assert isinstance(key, (str, float, int))
                with self.map(key):
                    result[key] = self(value)
            return result

        if isinstance(x, TaskOutput):
            # Process task if different
            if self.recurse_task and x.__xpm__.task is not x.__unwrap__():
                self(x.__xpm__.task)

            # Processed the wrapped config
            return self(x.__unwrap__())

        if isinstance(x, (float, int, str, Path, Enum)):
            return x

        raise NotImplementedError(f"Cannot handle a value of type {type(x)}")


class GenerationConfigProcessing(ConfigProcessing):
    def __init__(self, context: GenerationContext, recurse_task=False):
        super().__init__(recurse_task=recurse_task)
        self.context = context

    def list(self, i: int):
        return self.context.push(str(i))

    def map(self, k: str):
        return self.context.push(k)


def getqualattr(module, qualname):
    """Get a qualified attributed value"""
    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls


class ConfigInformation:
    """Holds experimaestro information for a config (or task) instance"""

    _meta: Optional[bool]
    """Forces this configuration to be a meta-parameter"""

    # Set to true when loading from JSON
    LOADING = False

    def __init__(self, pyobject: "TypeConfig"):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject
        self.xpmtype = pyobject.__xpmtype__  # type: ObjectType
        self.values = {}

        # Meta-informations
        self._tags = {}
        self._initinfo = ""

        # Generated task
        self._taskoutput = None

        # State information
        self.job = None

        # Explicitely added dependencies
        self.dependencies = []

        # Cached information
        self._identifier = None
        self._validated = False
        self._sealed = False
        self._meta = None

    def set_meta(self, value: Optional[bool]):
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
        # Not an argument, bypass
        if k not in self.xpmtype.arguments:
            setattr(self.pyobject, k, v)
            return

        if self._sealed and not bypass:
            raise AttributeError(f"Object is read-only (trying to set {k})")

        try:
            argument = self.xpmtype.arguments.get(k, None)
            if argument:
                if not bypass and (argument.generator or argument.constant):
                    raise AttributeError("Property %s is read-only" % (k))
                if v is not None:
                    self.values[k] = argument.validate(v)
                elif argument.required:
                    raise AttributeError("Cannot set required attribute to None")
                else:
                    self.values[k] = None
            else:
                raise AttributeError(
                    "Cannot set non existing attribute %s in %s" % (k, self.xpmtype)
                )
        except Exception:
            logger.exception("Error while setting value %s" % k)
            raise

    def addtag(self, name, value):
        self._tags[name] = value

    def xpmvalues(self, generated=False):
        """Returns an iterarator over arguments and associated values"""
        for argument in self.xpmtype.arguments.values():
            if argument.name in self.values or (generated and argument.generator):
                yield argument, self.values[argument.name]

    def tags(self):
        class TagFinder(ConfigProcessing):
            def __init__(self):
                super().__init__(recurse_task=True)
                self.tags = {}

            def postprocess(self, config: Config, values):
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
                    if value is None and argument.required:
                        raise ValueError(
                            "Value %s is required but missing when building %s at %s"
                            % (k, self.xpmtype, self._initinfo)
                        )
                    if isinstance(value, Config):
                        value.__xpm__.validate()
                elif argument.required:
                    if not argument.generator:
                        raise ValueError(
                            "Value %s is required but missing when building %s at %s"
                            % (k, self.xpmtype, self._initinfo)
                        )

            # Use __validate__ method
            if hasattr(self.pyobject, "__validate__"):
                try:
                    self.pyobject.__validate__()
                except Exception:
                    logger.error(
                        "Error while validating %s at %s", self.xpmtype, self._initinfo
                    )
                    raise

    def seal(self, context: GenerationContext):
        """Seal the object, generating values when needed,
        before scheduling the associated job(s)

        Arguments:
            - context: the generation context
        """

        class Sealer(GenerationConfigProcessing):
            def preprocess(self, config: Config):
                return not config.__xpm__._sealed, config

            def postprocess(self, config: Config, values):
                # Generate values
                for k, argument in config.__xpmtype__.arguments.items():
                    if argument.generator:
                        config.__xpm__.set(
                            k, argument.generator(self.context, config), bypass=True
                        )

                config.__xpm__._sealed = True

        Sealer(context, recurse_task=True)(self.pyobject)

    def __unseal__(self):
        """Unseal this configuration and its descendant

        Internal API - do not use
        """
        context = GenerationContext()

        class Unsealer(GenerationConfigProcessing):
            def preprocess(self, config: Config):
                return config.__xpm__._sealed, config

            def postprocess(self, config: Config, values):
                config.__xpm__._sealed = False
                config.__xpm__._identifier = None

        Unsealer(context, recurse_task=True)(self.pyobject)

    @property
    def identifier(self) -> Identifier:
        """Computes the unique identifier"""
        if self._identifier is None or not self._sealed:
            hashcomputer = HashComputer()
            hashcomputer.update(self.pyobject, myself=True)
            identifier = hashcomputer.identifier()

            if self._sealed:
                # Only cache the identifier if sealed
                self._identifier = identifier

            return identifier

        return self._identifier

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
        for argument, value in self.xpmvalues():
            try:
                if value is not None:
                    updatedependencies(
                        dependencies, value, path + [argument.name], taskids
                    )
            except Exception:
                logger.error("While setting %s", path + [argument.name])
                raise

    def submit(
        self, workspace: "Workspace", launcher: "Launcher", run_mode=None
    ) -> "TaskOutput":
        from experimaestro.scheduler import experiment, JobContext
        from experimaestro.scheduler.workspace import RunMode

        # --- Prepare the object
        if self.job:
            raise Exception("task %s was already submitted" % self)
        if not self.xpmtype.task:
            raise ValueError("%s is not a task" % self.xpmtype)

        # --- Submit the job

        self.job = self.xpmtype.task(
            self.pyobject, launcher=launcher, workspace=workspace, run_mode=run_mode
        )

        # Validate the object
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
        self.seal(JobContext(self.job))

        # --- Search for dependencies

        workspace = workspace or (
            experiment.CURRENT.workspace if experiment.CURRENT else None
        )

        # Call onSubmit
        launcher = (
            launcher
            or (workspace and workspace.launcher)
            or (experiment.CURRENT and experiment.CURRENT.workspace.launcher)
        )
        if launcher:
            launcher.onSubmit(self.job)

        # Add job dependencies
        self.updatedependencies(self.job.dependencies, [], set([id(self.pyobject)]))

        # Add predefined dependencies
        self.job.dependencies.update(self.dependencies)

        run_mode = (
            workspace.run_mode if run_mode is None else run_mode
        ) or RunMode.NORMAL
        if run_mode == RunMode.NORMAL:
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
            if self.job.donepath.is_file():
                cprint(f"[done] {s}", "light_green", file=sys.stderr)
            elif self.job.failedpath.is_file():
                cprint(f"[failed] {s}", "light_red", file=sys.stderr)
            else:
                cprint(f"[not run] {s}", "light_blue", file=sys.stderr)

        # Handle an output configuration
        if hasattr(self.pyobject, "config"):
            config = self.pyobject.config()
            if isinstance(config, dict):
                # Converts to the specified configuration
                hints = get_type_hints(self.pyobject.config)
                config = hints["return"](**config)
            config.__xpm__.validate()
            self._taskoutput = TaskOutput(config, self.pyobject)

        # New way to handle outputs
        elif hasattr(self.pyobject, "taskoutputs"):
            value = self.pyobject.taskoutputs()
            self._taskoutput = TaskOutput(value, self.pyobject)

        # Otherwise, the output is just the config
        else:
            self._taskoutput = TaskOutput(self.pyobject, self.pyobject)

        return self._taskoutput

    # --- Serialization

    @staticmethod
    def _outputjsonvalue(value, context):
        """Serialize a value"""
        if value is None:
            return None

        elif isinstance(value, list):
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

        elif isinstance(value, SerializedTaskOutput):
            # Reference to a serialized object
            return {
                "type": "serialized",
                "value": id(value.__xpm__.serialized.loader),
                "path": [c.toJSON() for c in value.__xpm__.path],
            }

        elif isinstance(value, TaskOutput):
            return {
                "type": "python",
                "value": id(value.__unwrap__()),
                # We add the task for identifier computation
                "task": id(value.__xpm__.task),
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

        # Serialize ourselves
        state_dict = {
            "id": id(self.pyobject),
            "module": self.xpmtype._module,
            "type": self.xpmtype.objecttype.__qualname__,
            "typename": self.xpmtype.name(),
            "identifier": self.identifier.state_dict(),
        }

        if self.meta:
            state_dict["meta"] = self.meta

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
        # objects
        if isinstance(value, SerializedTaskOutput):
            loader = value.__xpm__.serialized.loader
            if id(loader) not in context.serialized:
                objects.append(
                    {
                        "id": id(loader),
                        "serialized": True,
                        "module": loader.__class__.__module__,
                        "type": loader.__class__.__qualname__,
                        "value": loader.toJSON(),
                    }
                )
            return

        # Unwrap if needed
        if isinstance(value, TaskOutput):
            # We will need to output the task configuration objects
            ConfigInformation.__collect_objects__(value.__xpm__.task, objects, context)

            # Unwrap the value to output it
            value = value.__unwrap__()

        if isinstance(value, Config):
            value.__xpm__.__get_objects__(objects, context)
        elif isinstance(value, list):
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
                "has_subparam": self.identifier.sub is not None,
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
        path: Union[str, Path, SerializedPathLoader], as_instance: bool = False
    ) -> "Config":
        """Deserialize a configuration

        :param path: The filesystem Path to use, or a way to download the
            information through a function taking two arguments
        :return: A Config object
        """
        # Load
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
            config, as_instance=as_instance, data_loader=data_loader
        )

    @staticmethod
    def _objectFromParameters(value: Any, objects: Dict[str, Any], as_instance: bool):
        # A list
        if isinstance(value, list):
            return [
                ConfigInformation._objectFromParameters(x, objects, as_instance)
                for x in value
            ]

        # A dictionary
        if isinstance(value, dict):
            if "type" not in value:
                # Just a plain dictionary
                return {
                    ConfigInformation._objectFromParameters(
                        key, objects, as_instance
                    ): ConfigInformation._objectFromParameters(
                        value, objects, as_instance
                    )
                    for key, value in value.items()
                }

            # The value is an object (that should have been serialized first)
            if value["type"] == "python":
                obj = objects[value["value"]]

                # If we have a task
                if not as_instance:
                    if task_id := value.get("task", None):
                        task = objects[task_id]
                        return TaskOutput(obj, task)
                return obj

            if value["type"] == "serialized":
                o = objects[value["value"]]
                for c in value["path"]:
                    if c["type"] == "item":
                        o = o[c["name"]]
                    elif c["type"] == "attr":
                        o = getattr(o, c["name"])
                    else:
                        raise TypeError(f"Cannot handle type {c['type']}")
                return o

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
    ) -> "TypeConfig":
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
    def fromParameters(
        definitions: List[Dict],
        as_instance=True,
        data_loader: Optional[SerializedPathLoader] = None,
        discard_id: bool = False,
    ):
        """Builds config (instances) from a dictionary"""
        o = None
        objects = {}
        import experimaestro.taskglobals as taskglobals

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
                logger.debug("Importing module %s", definition["module"])
                mod = importlib.import_module(module_name)

            cls = getqualattr(mod, definition["type"])

            if definition.get("serialized", False):
                o = cls.fromJSON(definition["value"])
            else:
                # Creates an object (and not a config)
                o = cls.__new__(cls, __xpmobject__=as_instance)
                xpmtype = cls.__getxpmtype__()  # type: ObjectType

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
                    xpminfo = o.__xpm__  # type: ConfigInformation

                    meta = definition.get("meta", None)
                    if meta:
                        xpminfo._meta = meta
                    if xpminfo.xpmtype.task is not None:
                        o.__xpm__.job = object()

                # Set the fields
                for name, value in definition["fields"].items():
                    v = ConfigInformation._objectFromParameters(
                        value, objects, as_instance
                    )

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
                    # Seal and set the identifier
                    if not discard_id:
                        xpminfo._identifier = Identifier.from_state_dict(
                            definition["identifier"]
                        )
                    xpminfo._sealed = True

                assert definition["id"] not in objects, (
                    "Duplicate id %s" % definition["id"]
                )

            objects[definition["id"]] = o

        return o

    class FromPython(GenerationConfigProcessing):
        def __init__(self, context: GenerationContext):
            super().__init__(context)
            self.objects = {}

        def preprocess(self, config: "Config"):
            v = self.objects.get(id(config))
            return v is None, v

        def postprocess(self, config: "Config", values: Dict[str, Any]):
            # Creates an object (and not a config)
            o = config.__xpmtype__.objecttype.__new__(
                config.__xpmtype__.objecttype, __xpmobject__=True
            )

            # And calls the parameter-less initialization
            o.__init__()

            self.objects[id(self)] = o

            # Generate values (in configuration)
            if not config.__xpm__._sealed:
                for arg in config.__xpmtype__.arguments.values():
                    if arg.generator is not None:
                        config.__xpm__.set(
                            arg.name, arg.generator(self.context, o), bypass=True
                        )
                    if not arg.required and not hasattr(o, arg.name):
                        config.__xpm__.set(arg.name, None)

            # Set values
            for key, value in config.__xpm__.values.items():
                setattr(o, key, values.get(key, value))

            # Call __post_init__
            o.__post_init__()

            return o

    def fromConfig(self, context: GenerationContext):
        """Generate an instance given the current configuration"""
        self.validate()
        processor = ConfigInformation.FromPython(context)
        return processor(self.pyobject)

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


class TypeConfig:
    """Class for configuration objects"""

    __xpmtype__: ObjectType

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
            f"{self.__xpmtype__.objecttype.__module__}."
            f"{self.__xpmtype__.objecttype.__qualname__}({params})"
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

    def instance(self, context: GenerationContext = None) -> T:
        """Return an instance with the current values"""
        if context is None:
            from experimaestro.xpmutils import EmptyContext

            context = EmptyContext()
        else:
            assert isinstance(
                context, GenerationContext
            ), f"{context.__class__} is not an instance of GenerationContext"
        return self.__xpm__.fromConfig(context)  # type: ignore

    def submit(self, *, workspace=None, launcher=None, run_mode: "RunMode" = None):
        """Submit this task

        :param workspace: the workspace, defaults to None
        :param launcher: The launcher, defaults to None
        :param run_mode: Run mode (if None, uses the workspace default)
        :return: a :py:class:TaskOutput object
        """
        return self.__xpm__.submit(workspace, launcher, run_mode=run_mode)

    def stdout(self):
        return self.__xpm__.job.stdout

    def stderr(self):
        return self.__xpm__.job.stderr

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


class Config:
    """Base type for all objects in python interface"""

    __xpmtype__: ClassVar[ObjectType]
    """The object type holds all the information about a specific subclass
    experimaestro metadata"""

    __xpm__: ConfigInformation
    """The __xpm__ object contains all instance specific information about a
    configuration/task"""

    @classmethod
    def __getxpmtype__(cls):
        """Get (and create if necessary) the Object type of this"""
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

    def __getnewargs_ex__(self):
        # __new__ will be called with those arguments when unserializing
        return ((), {"__xpmobject__": True})

    @classmethod
    def c(cls: Type[T], **kwargs) -> T:
        """Allows typing to process easily"""
        return cls.__new__(cls, **kwargs)

    def __new__(cls: Type[T], *args, __xpmobject__=False, **kwargs) -> T:
        """Returns an instance of a TypeConfig when called __xpmobject__ is False,
        and otherwise the real object
        """

        if __xpmobject__:
            # __init__ is  called directly
            return object.__new__(cls)

        # We use the configuration type
        o = object.__new__(cls.__getxpmtype__().configtype)
        o.__init__(*args, **kwargs)
        return o

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

    def __identifier__(self) -> Identifier:
        return self.__xpm__.identifier


class Task(Config):
    """Base class for tasks"""

    __tags__: Dict[str, str]
    """Tags associated with class"""

    def execute(self):
        """The main method that should be implemented in all"""
        raise NotImplementedError()


# --- Output proxy


class Proxy:
    """A proxy for a value"""

    def __unwrap__(self) -> Any:
        raise NotImplementedError()


class ItemAccessor:
    def __init__(self, key: Any):
        self.key = key

    def toJSON(self):
        return {"type": "item", "name": self.key}

    def get(self, value):
        return value.__getitem__(self.key)


class AttrAccessor:
    def __init__(self, key: Any, default: Any):
        self.key = key
        self.default = default

    def get(self, value):
        return getattr(value, self.key, self.default)

    def toJSON(self):
        return {"type": "attr", "name": self.key}


class Serialized:
    """Simple serialization object"""

    def __init__(self, value):
        self.value = value

    def toJSON(self):
        return self.value


class SerializedConfig:
    """A serializable configuration

    This can be used to define a loading mechanism when instanciating the
    configuration
    """

    pyobject: Config
    """The configuration that will be serialized"""

    def __init__(self, pyobject: Config, loader: Callable[[Path], Config]):
        self.pyobject = pyobject
        self.loader = loader


class TaskOutputInfo:
    def __init__(self, task: Task):
        self.task = task
        self.value = None
        self.path = None
        self.serialized = None

    @property
    def identifier(self):
        return self.task.__xpm__.identifier

    @property
    def job(self):
        return self.task.__xpm__.job

    def tags(self):
        tags = self.task.__xpm__.tags()
        return tags

    def stdout(self):
        return self.task.__xpm__.job.stdout

    def stderr(self):
        return self.task.__xpm__.job.stderr

    def wait(self):
        from experimaestro.scheduler import JobState

        return self.task.__xpm__.job.wait() == JobState.DONE


class TaskOutput(Proxy):
    """Task proxy

    This is used when accessing properties *after* having submitted a task,
    to keep track of the dependencies
    """

    def __init__(self, value: Any, task: Union[Task, TaskOutputInfo]):
        self.__xpm__ = (
            task if isinstance(task, TaskOutputInfo) else TaskOutputInfo(task)
        )
        self.__xpm__.value = value

    def _wrap(self, value):
        if isinstance(value, SerializedConfig):
            return SerializedTaskOutput(value.pyobject, value, self.__xpm__.task, [])

        if isinstance(value, (str, int, float, Path, bool)):
            # No need to wrap if direct
            return value

        return TaskOutput(value, self.__xpm__.task)

    def __getitem__(self, key: Any):
        return self._wrap(self.__xpm__.value.__getitem__(key))

    def __getattr__(self, key: str, default=None) -> Any:
        return self._wrap(getattr(self.__xpm__.value, key, default))

    def __unwrap__(self):
        return self.__xpm__.value

    def __call__(self, *args, **kwargs):
        assert callable(self.__xpm__.value), "Attribute is not a function"
        __self__ = TaskOutput(self.__xpm__.value.__self__, self.__xpm__.task)
        return self.__xpm__.value.__func__(__self__, *args, **kwargs)


class SerializedTaskOutput(TaskOutput):
    """Used when serializing a configuration

    Here, we need to keep track of the path to the value we need
    """

    def __init__(
        self, value, serialized: SerializedConfig, task: Task, path: List[Any]
    ):
        super().__init__(value, task)
        self.__xpm__.serialized = serialized
        self.__xpm__.path = path

    def __getitem__(self, key: Any):
        value = self.__xpm__.value.__getitem__(key)
        return SerializedTaskOutput(
            value, self.serialized, self.__xpm__.task, self.path + [ItemAccessor(key)]
        )

    def __getattr__(self, key: str, default=None) -> Any:
        value = getattr(self.__xpm__.value, key, default)
        return SerializedTaskOutput(
            value,
            self.__xpm__.serialized,
            self.__xpm__.task,
            self.__xpm__.path + [AttrAccessor(key, default)],
        )


# --- Utility functions


def copyconfig(config_or_output: Union[Config, TaskOutput], **kwargs):
    """Copy a configuration or task output

    Useful to modify a configuration that can be potentially
    wrapped into a task output (i.e., the configuration can be
    a task output).
    """

    if isinstance(config_or_output, TaskOutput):
        output = config_or_output
        config = config_or_output.__unwrap__()
        assert isinstance(config, Config)
    else:
        config = config_or_output
        output = None

    # Builds a new configuration object
    copy = config.__class__()

    fullkwargs = {name: value for name, value in config.__xpm__.values.items()}
    fullkwargs.update(kwargs)
    for name, value in fullkwargs.items():
        copy.__xpm__.set(name, value, True)

    if output is None:
        return copy

    # wrap in Task output
    return TaskOutput(copy, output.__xpm__)


def setmeta(config: Config, flag: bool):
    """Flags the configuration as a meta-parameter"""
    config.__xpm__.set_meta(flag)
    return config
