"""Configuration and tasks"""

from pathlib import Path
import hashlib
import struct
import io
import fasteners
import inspect
import importlib
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union, get_type_hints
from experimaestro.utils import logger
import sys
from contextlib import contextmanager
from experimaestro.core.types import ObjectType


class Identifier:
    def __init__(self, all, main, sub):
        self.all = all
        self.main = main
        self.sub = sub


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

    def __init__(self):
        # Hasher for parameters
        self._hasher = hashlib.sha256()
        self._subhasher = None
        self._tasks = set()

    def identifier(self) -> Identifier:
        sub = None if self._subhasher is None else self._subhasher.digest()
        main = self._hasher.digest()
        if sub:
            h = hashlib.sha256()
            h.update(main)
            h.update(sub)
            all = h.digest()
        else:
            all = main

        return Identifier(all, main, sub)

    def _hashupdate(self, bytes, subparam):
        if subparam:
            # If subparam, creates a specific sub-hasher
            if self._subhasher is None:
                self._subhasher = hashlib.sha256()
            self._subhasher.update(bytes)
        else:
            self._hasher.update(bytes)

    def update(self, value, subparam=False):
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
            self._hashupdate(HashComputer.LIST_ID, subparam=subparam)
            self._hashupdate(struct.pack("!d", len(value)), subparam=subparam)
            for x in value:
                self.update(x, subparam=subparam)
        elif isinstance(value, dict):
            self._hashupdate(HashComputer.DICT_ID, subparam=subparam)
            items = list(value.items())
            items.sort(key=lambda x: x[0])
            for key, value in items:
                self.update(key, subparam=subparam)
                self.update(value, subparam=subparam)
        elif isinstance(value, Config):
            xpmtype = value.__xpmtype__
            self._hashupdate(HashComputer.OBJECT_ID, subparam=subparam)
            self._hashupdate(xpmtype.identifier.name.encode("utf-8"), subparam=subparam)

            # Add task parameters
            if value.__xpm__._task:
                task = value.__xpm__._task
                # Avoid recursion
                if id(task) not in self._tasks:
                    self._tasks.add(id(task))
                    self._hashupdate(HashComputer.TASK_ID, subparam=subparam)
                    self.update(task, subparam=subparam)

            # Process arguments (sort by name to ensure uniqueness)
            arguments = sorted(xpmtype.arguments.values(), key=lambda a: a.name)
            for argument in arguments:
                arg_subparam = subparam or argument.subparam

                # Ignored argument
                if argument.ignored or argument.generator:
                    continue

                # Argument value
                argvalue = getattr(value, argument.name, None)
                if not argument.constant and (
                    argument.default and argument.default == argvalue
                ):
                    # No update if same value (and not constant)
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
        if value.__xpm__._task is not None:
            # Value that depends on a task
            add(value.__xpm__._task)
        elif value.__xpmtype__.task:
            add(value)
        else:
            value.__xpm__.updatedependencies(dependencies, path, taskids)
    elif isinstance(value, (list, set)):
        for el in value:
            updatedependencies(dependencies, el, path, taskids)
    elif isinstance(value, (dict,)):
        for key, val in value.items():
            updatedependencies(dependencies, key, path, taskids)
            updatedependencies(dependencies, val, path, taskids)
    elif isinstance(value, (str, int, float, Path)):
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


class ConfigProcessing:
    """Allows to perform an operation on all nested configurations"""

    def __init__(self):
        pass

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

            flag, value = self.preprocess(x)

            if not flag:
                # Stop processing and returns value
                return value

            result = {}
            for arg, v in info.xpmvalues():
                if v is not None:
                    with self.map(arg.name):
                        result[arg.name] = self(v)
            return self.postprocess(x, result)

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

        if isinstance(x, (float, int, str, Path)):
            return x

        raise NotImplementedError(f"Cannot handle a value of type {type(x)}")


class GenerationConfigProcessing(ConfigProcessing):
    def __init__(self, context: GenerationContext):
        self.context = context

    def list(self, i: int):
        return self.context.push(str(i))

    def map(self, k: str):
        return self.context.push(k)


class ConfigInformation:
    """Holds experimaestro information for a config (or task) instance"""

    # Set to true when loading from JSON
    LOADING = False

    def __init__(self, pyobject: "Config"):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject
        self.xpmtype = pyobject.__xpmtype__  # type: ObjectType
        self.values = {}

        # Meta-informations
        self._tags = {}
        self._initinfo = {}

        # Generated task
        self._task = None  # type: Optional[Task]

        # State information
        self.job = None

        # Explicitely added dependencies
        self.dependencies = []

        # Cached information
        self._identifier = None
        self._validated = False
        self._sealed = False

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

        if self._sealed:
            raise AttributeError("Object is read-only")

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

    def tags(self, tags=None, taskmode=False):
        tags = tags or {}

        # If nested within a task, returns the task tags
        if self._task and not taskmode:
            return self._task.__xpm__.tags(tags=tags, taskmode=True)

        # Recursion within configurations
        tags.update(self._tags)
        for argument, value in self.xpmvalues():
            if isinstance(value, Config):
                value.__xpm__.tags(tags, taskmode=taskmode)
        return tags

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
                self.pyobject.__validate__()

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
                            k, argument.generator(self.context), bypass=True
                        )

                # Set task
                if context.task is not None and config.istaskoutput():
                    assert not config.__xpm__._sealed, "Object with output is sealed"
                    config.__xpm__._task = context.task

                config.__xpm__._sealed = True

        Sealer(context)(self.pyobject)

    @property
    def identifier(self) -> Identifier:
        """Computes the unique identifier"""
        if self._identifier is None:
            hashcomputer = HashComputer()
            hashcomputer.update(self.pyobject)
            self._identifier = hashcomputer.identifier()
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

    def submit(self, workspace, launcher, dryrun=False):
        # --- Prepare the object
        if self.job:
            raise Exception("task %s was already submitted" % self)
        if not self.xpmtype.task:
            raise ValueError("%s is not a task" % self.xpmtype)

        # --- Submit the job
        from experimaestro.scheduler import Job, Scheduler, JobContext

        self.job = self.xpmtype.task(
            self.pyobject, launcher=launcher, workspace=workspace, dryrun=dryrun
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
        self.updatedependencies(self.job.dependencies, [], set([id(self.pyobject)]))

        # Add predefined dependencies
        self.job.dependencies.update(self.dependencies)

        if not dryrun and Scheduler.CURRENT.submitjobs:
            other = Scheduler.CURRENT.submit(self.job)
            if other:
                # Just returns the other task
                return other.config
        else:
            logger.warning("Simulating: not submitting job %s", self.job)

        # Handle an output configuration
        if hasattr(self.pyobject, "config"):
            config = self.pyobject.config()
            if isinstance(config, dict):
                hints = get_type_hints(self.pyobject.config)
                config = hints["return"](**config)
            config.__xpm__._task = self.pyobject
            return config

        return self.pyobject

    # --- Serialization

    @staticmethod
    def _outputjsonobjects(value, jsonout, context, serialized):
        """Output JSON objects"""
        # objects
        if isinstance(value, Config):
            value.__xpm__._outputjson_inner(jsonout, context, serialized)
        elif isinstance(value, list):
            for el in value:
                ConfigInformation._outputjsonobjects(el, jsonout, context, serialized)
        elif isinstance(value, dict):
            for name, el in value.items():
                ConfigInformation._outputjsonobjects(el, jsonout, context, serialized)
        elif isinstance(value, (Path, int, float, str)):
            pass
        else:
            raise NotImplementedError(
                "Cannot serialize objects of type %s", type(value)
            )

    @staticmethod
    def _outputjsonvalue(key, value, jsonout, context):
        """Output JSON value"""
        if value is None:
            jsonout.write(*key, None)
        elif isinstance(value, Config):
            with jsonout.subobject(*key) as obj:
                obj.write("type", "python")
                obj.write("value", id(value))
        elif isinstance(value, list):
            with jsonout.subarray(*key) as arrayout:
                for el in value:
                    ConfigInformation._outputjsonvalue([], el, arrayout, context)
        elif isinstance(value, dict):
            with jsonout.subobject(*key) as objectout:
                for name, el in value.items():
                    ConfigInformation._outputjsonvalue([name], el, objectout, context)
        elif isinstance(value, Path):
            with jsonout.subobject(*key) as objectout:
                objectout.write("type", "path")
                objectout.write("value", str(value))
        elif isinstance(value, (int, float, str)):
            jsonout.write(*key, value)
        else:
            raise NotImplementedError(
                "Cannot serialize objects of type %s", type(value)
            )

    def _outputjson_inner(self, jsonstream, context, serialized: Set[int]):
        # Skip if already serialized
        if id(self.pyobject) in serialized:
            return

        serialized.add(id(self.pyobject))

        # Serialize sub-objects
        for argument, value in self.xpmvalues():
            if value is not None:
                ConfigInformation._outputjsonobjects(
                    value, jsonstream, context, serialized
                )

        with jsonstream.subobject() as objectout:

            # Serialize ourselves
            objectout.write("id", id(self.pyobject))
            if not self.xpmtype._package:
                objectout.write("file", str(self.xpmtype._file))

            objectout.write("module", self.xpmtype._module)
            objectout.write("type", self.xpmtype.objecttype.__qualname__)

            # Serialize identifier and typename
            # TODO: remove when not needed (cache issues)
            objectout.write("typename", self.xpmtype.name())
            objectout.write("identifier", self.identifier.all.hex())

            with objectout.subobject("fields") as jsonfields:
                for argument, value in self.xpmvalues():
                    ConfigInformation._outputjsonvalue(
                        [argument.name], value, jsonfields, value
                    )

    def outputjson(self, out: io.TextIOBase, context: "CommandContext"):
        """Outputs the json of this object

        The format is an array of objects
        {
            "tags: [ LIST_OF_TAGS ],
            "workspace": FOLDERPATH,
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
        import jsonstreams

        serialized: Set[int] = set()
        with jsonstreams.Stream(jsonstreams.Type.object, fd=out, close_fd=True) as out:
            # Write information
            out.write("has_subparam", self.identifier.sub is not None)

            out.write("workspace", str(context.workspace.path.absolute()))

            with out.subobject("tags") as objectout:
                for key, value in self.tags().items():
                    objectout.write(key, value)

            # Write objects
            with out.subarray("objects") as arrayout:
                self._outputjson_inner(arrayout, context, serialized)

    # def _outputjson(self, jsonout, context, key=[]):
    #     with jsonout.subobject(*key) as objectout:
    #         self._outputjson_inner(objectout, context)

    @staticmethod
    def _objectFromParameters(value, objects):
        if isinstance(value, list):
            return [ConfigInformation._objectFromParameters(x, objects) for x in value]
        if isinstance(value, dict):
            if "type" not in value:
                # Just a plain dictionary
                return {
                    ConfigInformation._objectFromParameters(
                        key, objects
                    ): ConfigInformation._objectFromParameters(value, objects)
                    for key, value in value.items()
                }
            elif value["type"] == "python":
                return objects[value["value"]]
            if value["type"] == "path":
                return Path(value["value"])
            else:
                raise Exception("Unhandled type: %s", value["type"])

        return value

    @staticmethod
    def fromParameters(definitions):
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
                    print(spec, module_name, path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
            else:
                logger.debug("Importing module %s", definition["module"])
                mod = importlib.import_module(module_name)

            cls = mod
            for part in definition["type"].split("."):
                cls = getattr(cls, part)

            # Creates an object (and not a config)
            o = cls.__new__(cls, __xpmobject__=True)

            # And calls the parameter-less initialization
            o.__init__()

            if "typename" in definition:
                o.__xpmtypename__ = definition["typename"]
                o.__xpmidentifier__ = definition["identifier"]

                if "." in o.__xpmtypename__:
                    _, name = o.__xpmtypename__.rsplit(".", 1)
                else:
                    name = o.__xpmtypename__
                basepath = (
                    taskglobals.wspath
                    / "jobs"
                    / o.__xpmtypename__
                    / o.__xpmidentifier__
                )
                o.__xpm_stdout__ = basepath / f"{name}.out"
                o.__xpm_stderr__ = basepath / f"{name}.err"

            for name, value in definition["fields"].items():
                v = ConfigInformation._objectFromParameters(value, objects)
                setattr(o, name, v)
                assert (
                    getattr(o, name) is v
                ), f"Problem with deserialization {name} of {o.__class__}"

            # Calls post-init
            postinit = getattr(o, "__postinit__", None)
            if postinit is not None:
                postinit()

            assert definition["id"] not in objects, "Duplicate id %s" % definition["id"]
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

            # Set values
            for key, value in values.items():
                setattr(o, key, value)
                postinit = getattr(o, "__postinit__", None)
                if postinit is not None:
                    postinit()

            # Generate values
            for arg in config.__xpmtype__.arguments.values():
                if arg.generator is not None:
                    setattr(o, arg.name, arg.generator(self.context))

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
    if isinstance(v, (str, float, int, Path)):
        return v

    if isinstance(v, list):
        return [clone(x) for x in v]

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
        hexid = config.__xpmidentifier__  # type: str
        typename = config.__xpmtypename__  # type: str
        dir = taskglobals.wspath / "config" / typename / hexid

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        path = dir / name
        ipc_lock = fasteners.InterProcessLock(path.with_suffix(path.suffix + ".lock"))
        with ipc_lock:
            r = fn(config, path, *args, **kwargs)
            return r

    return __call__


class TypeConfig:
    """Class of configuration objects"""

    def __init__(self, **kwargs):
        """Initialize the configuration with the given parameters"""

        # Add configuration
        xpmtype = self.__xpmtype__

        # Create the config information object
        xpm = ConfigInformation(self)

        caller = inspect.getframeinfo(inspect.stack()[1][0])
        xpm._initinfo = "%s:%s" % (str(Path(caller.filename).absolute()), caller.lineno)

        self.__xpm__ = xpm

        # Initialize with arguments
        for name, value in kwargs.items():
            # Check if argument is OK
            if name not in xpmtype.arguments:
                raise ValueError("%s is not an argument for %s" % (name, xpmtype))

            # Special case of a tagged value
            if isinstance(value, TaggedValue):
                value = value.value
                self.__xpm__._tags[name] = value

            # Really set the value
            xpm.set(name, value)

        # Initialize with default arguments (or None)
        for name, value in xpmtype.arguments.items():
            if name not in kwargs:
                if value.default is not None:
                    self.__xpm__.set(name, clone(value.default), bypass=True)
                elif not value.required:
                    self.__xpm__.set(name, None, bypass=True)

    def tag(self, name, value) -> "Config":
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

    def instance(self, context: GenerationContext = None):
        """Return an instance with the current values"""
        if context is None:
            from experimaestro.xpmutils import EmptyContext

            context = EmptyContext()
        return self.__xpm__.fromConfig(context)

    def submit(self, *, workspace=None, launcher=None, dryrun=False):
        """Submit this task"""
        return self.__xpm__.submit(workspace, launcher, dryrun=dryrun)

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
        """Returns a copy of this configuration (ignores other non parameters attributes)"""
        return clone(self)

    def istaskoutput(self):
        """Returns whether this configuration is a task output"""
        for arg, value in self.__xpm__.xpmvalues(generated=True):
            if arg.isoutput():
                return True
        return False


T = TypeVar("T")


class Config:
    """Base type for all objects in python interface"""

    __xpmtype__: ObjectType
    __xpm__: ConfigInformation

    @classmethod
    def __getxpmtype__(cls):
        xpmtype = cls.__dict__.get("__xpmtype__", None)
        if xpmtype is None:
            from experimaestro.core.types import ObjectType

            return ObjectType(cls)
        return xpmtype

    def __getnewargs_ex__(self):
        # __new__ will be called with those arguments when unserializing
        return ((), {"__xpmobject__": True})

    def __new__(
        cls: Type[T], *args, __xpmobject__=False, **kwargs
    ) -> Union[TypeConfig, T]:
        """Returns an instance of a TypeConfig when called __xpmobject__ set"""

        if __xpmobject__:
            return object.__new__(cls)

        o = object.__new__(cls.__getxpmtype__().configtype)
        o.__init__(*args, **kwargs)
        return o


class Task(Config):
    """base class for tasks"""

    pass
