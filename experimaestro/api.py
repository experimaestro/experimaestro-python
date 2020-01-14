"""Old layer between the C API and experimaestro

TODO: re-organize
"""

import json
import io
import sys
import inspect
import os.path as op
import os
import logging
from pathlib import Path, PosixPath
import re
from typing import Union, Dict, List, Set, Iterator
import hashlib
import struct
from collections import ChainMap
import typing

from experimaestro.utils import logger

# --- Initialization

modulepath = Path(__file__).parent

class Identifier():
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return self.name.__hash__()
    def __eq__(self, other):
        return other.name.__eq__(self.name)

    def __getattr__(self, key):
        return self(key)
        
    def __getitem__(self, key):
        return self(key)

    def __call__(self, key):
        return Identifier(self.name + "." + key)

    def __str__(self):
        return self.name


def identifier(name: Union[str, Identifier]):
    if isinstance(name, Identifier):
        return name
    return Identifier(str(name))


class Argument:
    def __init__(self, name, type: "Type", required=None, help=None, generator=None, ignored=False, default=None):
        required = (default is None) if required is None else required
        if default is not None and required is not None and required:
            raise Exception("argument '%s' is required but default value is given" % name)

        self.name = name
        self.help = help
        self.type = type
        self.ignored = ignored
        if ignored is None:
            self.ignored = self.type.ignore
        else:
            self.ignored = False
        self.required = required
        self.default = default
        self.generator = generator

    def __repr__(self):
        return "argument[{name}:{type}]".format(**self.__dict__)

class TypeAttribute:
    def __init__(self, type, key):
        self.type = type
        self.path = [key]

class Type():
    """Any experimaestro type is a child class"""

    DEFINED:Dict[type, "Type"] = {}

    def __init__(self, tn: Union[str, Identifier], description=None): 
        if tn is None:
            tn = None
        elif isinstance(tn, str):
            tn = Identifier(tn)
        self.identifier = tn
        self.description = description

    @property
    def ignore(self):
        """Ignore by default"""
        return False

    def __str__(self):
        return "Type({})".format(self.identifier)

    def __repr__(self):
        return "Type({})".format(self.identifier)

    def name(self):
        return str(self.identifier)

    def isArray(self):
        return False


    @staticmethod
    def fromType(key):
        """Returns the type object corresponding to the given type"""
        logger.debug("Searching for type %s", key)

        if key is None:
            return Any

        defined = Type.DEFINED.get(key, None)
        if defined:
            return defined

        if isinstance(key, Type):
            return key

        if isinstance(key, TypeProxy):
            return key()

        if isinstance(key, Config):
            return key.__xpmtype__

        if inspect.isclass(key) and issubclass(key, Config):
            return key.__xpm__

        if type(key) == type(typing.List):
            # Python 3.6, 3.7+ test
            if key.__origin__ in [typing.List, list]:
                return ArrayType(Type.fromType(*key.__args__))

        raise Exception("No type found for %s", key)


class ObjectType(Type):
    """The XPM type of a configuration"""

    REGISTERED:Dict[str, "Config"] = {}
    
    def __init__(self, objecttype: type, identifier, description):
        super().__init__(identifier, description)

        self.objecttype = objecttype
        self.task = None
        self.originaltype = None

        self.arguments = ChainMap({}, *(tp.arguments for tp in self.parents()))

    def addArgument(self, argument: Argument):
        self.arguments[argument.name] = argument

    def getArgument(self, key: str) -> Argument:
        return self.arguments[key]

    def parents(self) -> Iterator["ObjectType"]:
        for tp in self.objecttype.__bases__:
            if issubclass(tp, Config) and tp not in [Config, Task]:
                yield tp.__xpm__
        
    @staticmethod
    def create(configclass: "Config", identifier, description, register=True):
        if register and str(identifier) in ObjectType.REGISTERED:
            _objecttype = ObjectType.REGISTERED[str(identifier)]
            if _objecttype.__xpm__.originaltype != configclass:
                # raise Exception("Experimaestro type %s is already declared" % identifier)
                pass

            logging.error("Experimaestro type %s is already declared" % identifier)
            return _objecttype


        if register:
            ObjectType.REGISTERED[str(identifier)] = configclass
        return ObjectType(configclass, identifier, description)


    def validate(self, value):
        if isinstance(value, dict):
            # This is a unserialized object
            valuetype = value.get("$type", None)
            if valuetype is None:
                raise ValueError("Object has no $type")

            classtype = ObjectType.REGISTERED.get(valuetype, None)
            if classtype:
                return classtype(**value)
            if not Config.TASKMODE:
                raise ValueError("Could not find type %s", valuetype)
        
            logger.debug("Using argument type (not real type)")
            return self.objecttype(**value)

        if not isinstance(value, Config):
            raise ValueError("%s is not an experimaestro type or task", value)
        
        types = self.objecttype

        if not isinstance(value, types):
            raise ValueError("%s is not a subtype of %s" % (value, types))

        if self.task and not value.__xpm__.job:
            raise ValueError("The value must be submitted before giving it")
        return value


class TypeProxy: 
    def __call__(self) -> Type:
        """Returns the real type"""
        raise NotImplementedError()

def definetype(*types):
    def call(typeclass):
        instance = typeclass(types[0].__name__)
        for t in types:
            Type.DEFINED[t] = instance
        return typeclass
    return call

@definetype(int)
class IntType(Type): 
    def validate(self, value):
        return int(value)

@definetype(str)
class StrType(Type): 
    def validate(self, value):
        return str(value)

@definetype(float)
class FloatType(Type): 
    def validate(self, value):
        return float(value)


@definetype(bool)
class BoolType(Type): 
    def validate(self, value):
        return bool(value)


@definetype(Path)
class PathType(Type): 
    def validate(self, value):
        if isinstance(value, dict) and value.get("$type", None) == "path":
            return Path(value.get("$value"))
        return Path(value)
        
    @property
    def ignore(self):
        """Ignore by default"""
        return True

class AnyType(Type):
    def __init__(self):
        super().__init__("any")

    def validate(self, value):
        return value

Any = AnyType()

class ArrayType(Type):  
    def __init__(self, type: Type):
        self.type = type

    def validate(self, value):
        if not isinstance(value, List):
            raise ValueError("value is not a list")

        return [self.type.validate(x) for x in value]


# --- XPM Objects

class HashComputer():
    OBJECT_ID = b'\x00'
    INT_ID = b'\x01'
    FLOAT_ID = b'\x02'
    STR_ID = b'\x03'
    PATH_ID = b'\x04'
    NAME_ID = b'\x05'
    NONE_ID = b'\x06'
    LIST_ID = b'\x06'

    def __init__(self):
        self.hasher = hashlib.sha256()

    def digest(self):
        return self.hasher.digest()

    def update(self, value):
        if value is None:
            self.hasher.update(NONE_ID)
        if isinstance(value, float):
            self.hasher.update(HashComputer.FLOAT_ID)
            self.hasher.update(struct.pack('!d', value))
        elif isinstance(value, int):
            self.hasher.update(HashComputer.INT_ID)
            self.hasher.update(struct.pack('!q', value))
        elif isinstance(value, str):
            self.hasher.update(HashComputer.STR_ID)
            self.hasher.update(value.encode("utf-8"))
        elif isinstance(value, list):
            self.hasher.update(HashComputer.LIST_ID)
            self.hasher.update(struct.pack('!d', len(value)))
            for x in value:
                self.update(x)
        elif isinstance(value, Config):
            xpmtype = value.__xpmtype__ # type: ObjectType
            self.hasher.update(HashComputer.OBJECT_ID)
            self.hasher.update(xpmtype.identifier.name.encode("utf-8"))
            arguments = sorted(xpmtype.arguments.values(), key=lambda a: a.name)
            for argument in arguments:
                # Hash name
                self.update(argument.name)

                # Hash value
                argvalue = getattr(value, argument.name, None)
                if argument.ignored or argument.generator:
                    continue
                if argument.default and argument.default == argvalue:
                    # No update if same value
                    continue
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

class TypeInformation():
    """Holds experimaestro information for a Config (config or task)"""

    # Set to true when loading from JSON
    LOADING = False

    def __init__(self, pyobject):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject       
        self.xpmtype = pyobject.__xpmtype__ # type: ObjectType

        # Meta-informations
        self._tags = {}

        # State information
        self.job = None
        self.dependencies = []
        self.setting = False

        # Cached information
        self._identifier = None
        self._validated = False
        self._sealed = False

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
                raise AttributeError("Cannot set non existing attribute %s in %s" % (k, self.xpmtype))
                # object.__setattr__(self, k, v)
        except:
            logger.error("Error while setting value %s" % k)
            raise

    def addtag(self, name, value):
        self._tags[name] = value

    def xpmvalues(self, generated=False):
        """Returns an iterarator over arguments and associated values"""
        for argument in self.xpmtype.arguments.values():
            if hasattr(self.pyobject, argument.name) or (generated and argument.generator):
                yield argument, getattr(self.pyobject, argument.name, None)

    def tags(self, tags={}):
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
                    raise ValueError("Some arguments were set but not declared (%s) or declared but not set (%s)" %  (",".join(notdeclared), ",".join(notset)))

            
            # Check each argument
            for k, argument in self.xpmtype.arguments.items():
                if hasattr(self.pyobject, k):
                    value = getattr(self.pyobject, k)
                    if isinstance(value, Config):
                        value.__xpm__.validate()
                elif argument.required:
                    if not argument.generator:
                        raise ValueError("Value %s is required but missing", k)

    def seal(self, job: "experimaestro.job.Job"):
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

    def updatedependencies(self, dependencies: Set["experimaestro.dependencies.Dependency"]):
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
        from .scheduler import Job, Scheduler

        self.job = self.xpmtype.task(self.pyobject, launcher=launcher, workspace=workspace)
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
            objectout.write("$job", str(self.job.launcher.connector.resolve(self.job.jobpath / self.job.name)))
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
    raise NotImplementedError("For type %s" % v)

class ConfigMetaclass(type):
    def __getattr__(cls, key):
        """Access to a class field"""
        if not key.startswith("__xpm") and key in cls.__xpm__.arguments:
            return cls.__xpm__.arguments[key]
        return type.__getattribute__(cls, key)

class Config(metaclass=ConfigMetaclass):
    """Base type for all objects in python interface"""

    # Set to true when executing a task to remove all checks
    TASKMODE = False
    
    def __init__(self, **kwargs):
        # Add configuration
        self.__xpmtype__ = self.__class__.__xpm__
        if not isinstance(self.__xpmtype__, ObjectType):
            logging.error("%s is not an object type", self.__xpmtype__)
            assert isinstance(self.__xpmtype__, ObjectType)

        xpm = TypeInformation(self)
        self.__xpm__ = xpm

        # Initialize with arguments
        for name, value in kwargs.items():
            if name not in xpm.xpmtype.arguments and not name in ["$type", "$job"]:
                if Config.TASKMODE:
                    # Do not set this attribute
                    logging.debug("Do not set %s (not in attributes)", name)
                    continue
                raise ValueError("%s is not an argument for %s" % (name, self.__xpmtype__))

            if isinstance(value, TaggedValue):
                value = value.value
                self.__xpm__._tags[name] = value
            xpm.set(name, value, bypass=TypeInformation.LOADING)

        # Initialize with default arguments
        for name, value in self.__xpmtype__.arguments.items():
            if name not in kwargs and value.default is not None:
                self.__xpm__.set(name, clone(value.default))

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
