"""Configuration and tasks"""

from pathlib import Path
import hashlib
import struct
import io
import fasteners
import os
import inspect
from experimaestro.utils import logger
from typing import Set
from experimaestro.constants import CACHEPATH_VARNAME

class HashComputer:
    OBJECT_ID = b"\x00"
    INT_ID = b"\x01"
    FLOAT_ID = b"\x02"
    STR_ID = b"\x03"
    PATH_ID = b"\x04"
    NAME_ID = b"\x05"
    NONE_ID = b"\x06"
    LIST_ID = b"\x06"

    def __init__(self):
        self.hasher = hashlib.sha256()

    def digest(self):
        return self.hasher.digest()

    def update(self, value):
        if value is None:
            self.hasher.update(HashComputer.NONE_ID)
        elif isinstance(value, float):
            self.hasher.update(HashComputer.FLOAT_ID)
            self.hasher.update(struct.pack("!d", value))
        elif isinstance(value, int):
            self.hasher.update(HashComputer.INT_ID)
            self.hasher.update(struct.pack("!q", value))
        elif isinstance(value, str):
            self.hasher.update(HashComputer.STR_ID)
            self.hasher.update(value.encode("utf-8"))
        elif isinstance(value, list):
            self.hasher.update(HashComputer.LIST_ID)
            self.hasher.update(struct.pack("!d", len(value)))
            for x in value:
                self.update(x)
        elif isinstance(value, Config):
            xpmtype = value.__xpmtype__  # type: ObjectType
            self.hasher.update(HashComputer.OBJECT_ID)
            self.hasher.update(xpmtype.identifier.name.encode("utf-8"))
            arguments = sorted(xpmtype.arguments.values(), key=lambda a: a.name)
            for argument in arguments:
                # Ignored argument
                if argument.ignored or argument.generator:
                    continue

                # Argument value
                argvalue = getattr(value, argument.name, None)
                if argument.default and argument.default == argvalue:
                    # No update if same value
                    continue

                # Hash name
                self.update(argument.name)

                # Hash value
                self.hasher.update(HashComputer.NAME_ID)
                self.update(argvalue)

        else:
            raise NotImplementedError("Cannot compute hash of type %s" % type(value))


def outputjson(jsonout, context, value, key=[]):
    if isinstance(value, Config):
        value.__xpm__._outputjson(jsonout, context, key)
    elif isinstance(value, list):
        with jsonout.subarray(*key) as arrayout:
            for el in value:
                outputjson(arrayout, context, el)
    elif isinstance(value, dict):
        with jsonout.subobject(*key) as objectout:
            for name, el in value.items():
                outputjson(objectout, context, el, [name])
    elif isinstance(value, Path):
        with jsonout.subobject(*key) as objectout:
            objectout.write("$type", "path")
            objectout.write("$value", str(value))
    elif isinstance(value, (int, float, str)):
        jsonout.write(*key, value)
    else:
        raise NotImplementedError("Cannot serialize objects of type %s", type(value))


def updatedependencies(dependencies, value: "Config"):
    """Search recursively jobs to add them as dependencies"""
    if isinstance(value, Config):
        if value.__xpmtype__.task:
            dependencies.add(value.__xpm__.dependency())
        else:
            value.__xpm__.updatedependencies(dependencies)
    elif isinstance(value, list):
        for el in value:
            updatedependencies(dependencies, el)
    elif isinstance(value, (str, int, float, Path)):
        pass
    else:
        raise NotImplementedError("update dependencies for type %s" % type(value))


class FakeJob:
    def __init__(self, path):
        self.path = Path(path)
        self.stdout = self.path.with_suffix(".out")
        self.stderr = self.path.with_suffix(".err")


class TaggedValue:
    def __init__(self, value):
        self.value = value


class ConfigInformation:
    """Holds experimaestro information for a Config object (config or task)"""

    # Set to true when loading from JSON
    LOADING = False

    def __init__(self, pyobject):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject
        self.xpmtype = pyobject.__xpmtype__  # type: ObjectType

        # Meta-informations
        self._tags = {}
        self._initinfo = {}

        # State information
        self.job = None
        self.dependencies = []
        self.setting = False

        # Cached information
        self._identifier = None
        self._validated = False
        self._sealed = False


    def objectmodules(self, modules=set()):
        """Returns all the modules"""
        modules.add(self.pyobject.__module__)
        for argument, value in self.xpmvalues():
            if isinstance(value, Config):
                value.__xpm__.objectmodules(modules)
        return modules


    def set(self, k, v, bypass=False):
        if self._sealed:
            raise AttributeError("Object is read-only")

        try:
            argument = self.xpmtype.arguments.get(k, None)
            if argument:
                # If argument, check the value
                if not bypass and argument.generator:
                    raise AssertionError("Property %s is read-only" % (k))
                object.__setattr__(self.pyobject, k, argument.type.validate(v))
            elif k == "$type":
                if not Config.TASKMODE:
                    # Only check type if constructing the XP
                    assert v == str(self.xpmtype.identifier)
            elif k == "$job":
                self.job = FakeJob(v)
            else:
                raise AttributeError(
                    "Cannot set non existing attribute %s in %s" % (k, self.xpmtype)
                )
                # object.__setattr__(self, k, v)
        except:
            logger.error("Error while setting value %s" % k)
            raise

    def addtag(self, name, value):
        self._tags[name] = value

    def xpmvalues(self, generated=False):
        """Returns an iterarator over arguments and associated values"""
        for argument in self.xpmtype.arguments.values():
            if hasattr(self.pyobject, argument.name) or (
                generated and argument.generator
            ):
                yield argument, getattr(self.pyobject, argument.name, None)

    def tags(self, tags=None):
        if tags is None:
            tags = {}
        tags.update(self._tags)
        for argument, value in self.xpmvalues():
            if isinstance(value, Config):
                value.__xpm__.tags(tags)
        return tags

    def run(self):
        self.pyobject.execute()

    def init(self):
        self.pyobject._init()

    def validate(self):
        """Validate a value"""
        if not self._validated:
            self._validated = True

            # Check function
            if inspect.isfunction(self.xpmtype.originaltype):
                # Get arguments from XPM definition
                argnames = set()
                for argument, value in self.xpmvalues(True):
                    argnames.add(argument.name)

                # Get declared arguments from inspect
                spec = inspect.getfullargspec(self.xpmtype.originaltype)
                declaredargs = set(spec.args)

                # Arguments declared but not set
                notset = declaredargs.difference(argnames)
                notdeclared = argnames.difference(declaredargs)

                if notset or notdeclared:
                    raise ValueError(
                        "Some arguments were set but not declared (%s) or declared but not set (%s) at %s"
                        % (",".join(notdeclared), ",".join(notset), self._initinfo)
                    )

            # Check each argument
            for k, argument in self.xpmtype.arguments.items():
                if hasattr(self.pyobject, k):
                    value = getattr(self.pyobject, k)
                    if isinstance(value, Config):
                        value.__xpm__.validate()
                elif argument.required:
                    if not argument.generator:
                        raise ValueError("Value %s is required but missing when building %s at %s" 
                            % (k, self.xpmtype, self._initinfo))

            # Use __validate__ method
            if hasattr(self.pyobject, "__validate__"):
                self.pyobject.__validate__()


    def seal(self, job: "experimaestro.scheduler.Job"):
        """Seal the object, generating values when needed, before scheduling the associated job(s)"""
        if self._sealed:
            return

        for k, argument in self.xpmtype.arguments.items():
            if argument.generator:
                self.set(k, argument.generator(job), bypass=True)
            elif hasattr(self.pyobject, k):
                v = getattr(self.pyobject, k)
                if isinstance(v, Config):
                    v.__xpm__.seal(job)
        self._sealed = True

    @property
    def identifier(self):
        """Computes the unique identifier"""
        if self._identifier is None:
            hashcomputer = HashComputer()
            hashcomputer.update(self.pyobject)
            self._identifier = hashcomputer.digest()
        return self._identifier

    def dependency(self):
        """Returns a dependency"""
        from experimaestro.scheduler import JobDependency

        assert self.job
        return JobDependency(self.job)

    def updatedependencies(
        self, dependencies: Set["experimaestro.dependencies.Dependency"]
    ):
        for argument, value in self.xpmvalues():
            updatedependencies(dependencies, value)

    def submit(self, workspace, launcher, dryrun=None):
        # --- Prepare the object
        if self.job:
            raise Exception("task %s was already submitted" % self)
        if not self.xpmtype.task:
            raise ValueError("%s is not a task" % self.xpmtype)

        self.validate()

        # --- Submit the job
        from experimaestro.scheduler import Job, Scheduler

        self.job = self.xpmtype.task(
            self.pyobject, launcher=launcher, workspace=workspace
        )
        self.seal(self.job)

        # --- Search for dependencies
        self.updatedependencies(self.job.dependencies)
        self.job.dependencies.update(self.dependencies)

        if dryrun == False or (Scheduler.CURRENT.submitjobs and dryrun is None):
            Scheduler.CURRENT.submit(self.job)
        else:
            logger.warning("Simulating: not submitting job %s", self.job)

    def _outputjson_inner(self, objectout, context):
        objectout.write("$type", str(self.xpmtype.identifier))
        if self.job:
            objectout.write(
                "$job",
                str(
                    self.job.launcher.connector.resolve(
                        self.job.jobpath / self.job.name
                    )
                ),
            )
        for argument, value in self.xpmvalues():
            outputjson(objectout, context, value, [argument.name])

    def outputjson(self, out: io.TextIOBase, context):
        import jsonstreams

        with jsonstreams.Stream(jsonstreams.Type.object, fd=out) as objectout:
            self._outputjson_inner(objectout, context)

    def _outputjson(self, jsonout, context, key=[]):
        with jsonout.subobject(*key) as objectout:
            self._outputjson_inner(objectout, context)

    def add_dependencies(self, *dependencies):
        self.dependencies.extend(dependencies)


def clone(v):
    """Clone a value"""
    if isinstance(v, (str, float, int)):
        return v
    if isinstance(v, list):
        return [clone(x) for x in v]

    if isinstance(v, Config):
        # Create a new instance
        kwargs = {argument.name: clone(value) for argument, value in v.__xpm__.xpmvalues()}
        
        config = type(v).__new__(type(v))
        return type(v)(**kwargs)
        
    raise NotImplementedError("For type %s" % v)



def cache(fn, name: str):
    def __call__(config: "Config"):
        # Get path
        hexid = config.__xpm__.identifier.hex()
        typename = str(config.__xpmtype__.identifier.name)
        dir = Path(os.environ[CACHEPATH_VARNAME]) / typename / hexid
        dir.mkdir(parents=True, exist_ok=True)
        
        path = dir / name
        ipc_lock = fasteners.InterProcessLock(
            path.with_suffix(path.suffix + ".lock")
        )
        with ipc_lock:
            try:
                return fn(config, path)
            except:
                # Remove path
                if path.is_file():
                    path.unlink()
                raise
    return __call__


class Config():
    """Base type for all objects in python interface"""

    # Set to true when executing a task to remove all checks
    TASKMODE = False

    def __init__(self, **kwargs):
        # Add configuration
        from .types import ObjectType

        self.__xpmtype__ = self.__class__.__xpm__
        if not isinstance(self.__xpmtype__, ObjectType):
            logger.error("%s is not an object type", self.__xpmtype__)
            assert isinstance(self.__xpmtype__, ObjectType)

        xpm = ConfigInformation(self)
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        xpm._initinfo = "%s:%s" % (str(Path(caller.filename).absolute()), caller.lineno)

        self.__xpm__ = xpm

        # Initialize with arguments
        for name, value in kwargs.items():
            if name not in xpm.xpmtype.arguments and not name in ["$type", "$job"]:
                if Config.TASKMODE:
                    # Do not set this attribute when running a task
                    logger.debug("Do not set %s (not in attributes)", name)
                    continue
                raise ValueError(
                    "%s is not an argument for %s" % (name, self.__xpmtype__)
                )

            # Special case of a tagged value
            if isinstance(value, TaggedValue):
                value = value.value
                self.__xpm__._tags[name] = value

            # Really set the value
            xpm.set(name, value, bypass=ConfigInformation.LOADING)

        # Initialize with default arguments
        for name, value in self.__xpmtype__.arguments.items():
            if name not in kwargs and value.default is not None:
                self.__xpm__.set(name, clone(value.default))

        # call initialize
        if Config.TASKMODE:
            if hasattr(self, "__initialize__"):
                try:
                    self.__initialize__()
                except Exception as e:
                    logger.exception("Error while calling %s.__initialize__()", type(self).__name__)
                    raise


    def __setattr__(self, name, value):
        if not Config.TASKMODE and not name.startswith("__xpm"):
            return self.__xpm__.set(name, value)
        return super().__setattr__(name, value)

    def tag(self, name, value) -> "Config":
        self.__xpm__.addtag(name, value)
        return self

    def tags(self):
        """Returns the tag associated with this object (and below)"""
        return self.__xpm__.tags()

    def add_dependencies(self, *dependencies):
        """Adds tokens to the task"""
        self.__xpm__.add_dependencies(*dependencies)
        return self


class Task(Config):
    """Base type for all tasks"""

    def submit(self, *, workspace=None, launcher=None, dryrun=None):
        """Submit this task"""
        self.__xpm__.submit(workspace, launcher, dryrun=dryrun)
        return self

    def init(self):
        """Prepare object after creation"""
        pass

    def stdout(self):
        return self.__xpm__.job.stdout

    def stderr(self):
        return self.__xpm__.job.stderr

    @property
    def job(self):
        return self.__xpm__.job


# XPM task as a function
def gettaskclass(function, parents):
    class TaskFunction(Task, *parents):
        def execute(self):
            kwargs = {}
            for argument, value in self.__xpm__.xpmvalues():
                kwargs[argument.name] = value
            function(**kwargs)

    return TaskFunction
