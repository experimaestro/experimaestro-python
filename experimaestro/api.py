""" Object wrapping the C-api objects and methods
"""

import json
import sys
import inspect
import os.path as op
import os
import logging
from pathlib import Path, PosixPath
import re
from typing import Union, Dict
import hashlib
import struct
from .server import Server

# --- Initialization

logger = logging.getLogger("xpm")
modulepath = Path(__file__).parent

class Typename():
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, key):
        return self(key)

    def __call__(self, key):
        return Typename(self.name + "." + key)

    def __str__(self):
        return self.name


def typename(name: Union[str, Typename]):
    if isinstance(name, Typename):
        return name
    return Typename(str(name))


class BaseArgument:
    def __init__(self, name, type, help=None):
        self.name = name
        self.help = help
        self.type = Type.fromType(type)
        self.ignored = False
        self.required = True
        self.defaultvalue = None

    def __call__(self, t):
        t.__xpm__.addArgument(self)
        return t

class GeneratedArgument:
    """An argument that is generated"""
    def generate(self):
        pass


class TypeAttribute:
    def __init__(self, type, key):
        self.type = type
        self.path = [key]

class Type():
    DEFINED:Dict[type, "Type"] = {}

    def __init__(self, tn: Union[str, Typename], description=None): 
        if tn is None:
            tn = None
        elif isinstance(tn, str):
            tn = Typename(tn)
        self.typename = tn
        self.description = description
        self.arguments:Dict[str, BaseArgument] = {}

    def __str__(self):
        return "Type({})".format(self.typename)

    def name(self):
        return str(self.typename)

    def addArgument(self, argument: BaseArgument):
        self.arguments[argument.name] = argument

    def getArgument(self, key: str) -> BaseArgument:
        return self.arguments[key]

    def isArray(self):
        return False

    @staticmethod
    def fromType(key):
        """Returns the Type object corresponding to the given type"""
        logger.debug("Searching for type %s", key)

        if key is None:
            return AnyType

        defined = Type.DEFINED.get(key, None)
        if defined:
            return defined

        if isinstance(key, Type):
            return key

        if isinstance(key, TypeProxy):
            return key()

        if isinstance(key, PyObject):
            return key.__class__.__xpm__
        
        if inspect.isclass(key) and issubclass(key, PyObject):
            return key.__xpm__

        raise Exception("No type found for %s", key)

class ObjectType(Type):
    """The type of PyObject"""

    REGISTERED:Dict[Typename, "PyObject"] = {}
    def __init__(self, objecttype: "PyObject", typename, description):
        super().__init__(typename, description)
        self.objecttype = objecttype

        if self.typename in ObjectType.REGISTERED:
            raise Exception("Experimaestro type %s is already declared" % self.typename)
        ObjectType.REGISTERED[self.typename] = objecttype

    def validate(self, value):
        if not isinstance(value, PyObject):
            raise ValueError("%s is not an experimaestro type or task", value)
        if not isinstance(value, self.objecttype):
            raise ValueError("%s is not a subtype of %s")
        return value


def definetype(*types):
    def call(typeclass):
        instance = typeclass()
        for t in types:
            Type.DEFINED[t] = instance
    return call

@definetype(int)
class IntType: 
    def validate(self, value):
        return int(value)

@definetype(str)
class StrType: 
    def validate(self, value):
        return str(value)

@definetype(float)
class FloatType: 
    def validate(self, value):
        return float(value)

@definetype(Path)
class PathType: 
    def validate(self, value):
        return Path(value)

class ArrayType(Type):  
    def __init__(self, type: Type):
        self.type = type

class PredefinedType(Type):
    def __init__(self, ptr, pythontypes, topython, frompython):
        self.ptr = ptr
        self.pythontypes = pythontypes
        self.topython = topython
        self.frompython = frompython

        Type.XPM2PYTHON[str(self)] = self
        for pythontype in pythontypes:
            Type.PYTHON2XPM[pythontype] = self


AnyType = Type("any")

class AbstractCommandComponent(): pass

def aspath(path: [Path, str]):
    if isinstance(path, str):
        return Path(path)
    return path

class CommandPath(AbstractCommandComponent):
    def __init__(self, path: [Path, str]):
        self.path = aspath(path)

class CommandString(AbstractCommandComponent):
    def __init__(self, string: str):
        self.string = string

class CommandParameters(AbstractCommandComponent): pass

class AbstractCommand(): pass

class Command(AbstractCommand):
    def __init__(self):
        self.components = []

    def add(self, component: AbstractCommandComponent):
        self.components.append(component)

class CommandLine(Command):
    def __init__(self):
        self.commands = []

    def add(self, command: Command):
        self.commands.append(command)

class Dependency(): pass

class DependencyArray:
    def __init__(self):
        self.dependencies = []

    def add(self, dependency: Dependency):
        self.dependencies.append(dependency)

class Task():
    def __init__(self, tasktype: Type, *, taskId:Typename=None):
        assert tasktype and isinstance(tasktype, Type)
        self.type = tasktype
        self._commandline = None

    def name(self):
        return str(self.type)

    def commandline(self, commandline: CommandLine):
        assert commandline and isinstance(tasktype, CommandLine)
        self._commandline = commandline

    def submit(self, workspace, launcher, value, dependencies: DependencyArray):
        Workspace.SUBMITTED = True
        raise NotImplementedError("Task submit")

def aspath(path: Union[str, Path]):
    if isinstance(path, Path): 
        return path
    return Path(path)


class Job():   
    def state(self):
        raise NotImplementedError()

    def wait(self):
        raise NotImplementedError()

    def codePath(self):
        return self.code

    def stdoutPath(self):
        return self.stdout

    def stderrPath(self):
        return self.stderr


class Connector(): 
    pass

class LocalConnector(Connector): pass

class Launcher():
    def __init__(self):
        self.environ = {}
        self.notificationURL = None

    def setenv(self, key: str, value: str):
        self.environ[key] = value

    def setNotificationURL(self, url: str):
        self.notificationURL = url

class DirectLauncher(Launcher): pass

class Generator(): pass

class TypeProxy: pass

class Token(): pass

class CounterToken(Token):
    def __init__(self, path: Path, tokens: int=-1):
        self.path = path
        self.tokens = tokens

    def createDependency(self, count: int):
        return Dependency()


# --- XPM Objects

class HashComputer():
    def __init__(self):
        self.hasher = hashlib.sha512()
    def digest(self):
        return self.hasher.digest()
    def update(self, value):
        if isinstance(value, float):
            self.hasher.update(struct.pack('!d', value))
        elif isinstance(value, int):
            self.hasher.update(struct.pack('!q', value))
        elif isinstance(value, str):
            self.hasher.update(value.encode("utf-8"))
        elif isinstance(value, PyObject):
            self.hasher.update(value.__xpm__.identifier)
        else:
            raise NotImplementedError("Cannot compute hash of type %s" % type(value))

class Information():
    """Holds experimaestro information for a PyObject (Type or Task)"""

    def __init__(self, pyobject):
        # The underlying pyobject and XPM type
        self.pyobject = pyobject
        self.xpmtype = self.pyobject.__class__.__xpm__

        # Meta-informations
        self.tags = {}
        self.dependencies = DependencyArray()

        # State information
        self.job = None
        self.setting = False
        self.submitted = False

        # Cached information
        self._identifier = None
        self._validated = False

    def set(self, k, v):
        argument = self.xpmtype.arguments.get(k, None)
        if argument:
            # If argument, check the value
            object.__setattr__(self.pyobject, k, argument.type.validate(v))
        else:
            object.__setattr__(k, v)

    def run(self):
        self.pyobject.execute()

    def init(self):
        self.pyobject._init()

    def validate(self):
        """Validate values and generate if needed"""
        if not self._validated:
            self._validated = True
            for k, argument in self.xpmtype.arguments.items():
                if hasattr(self.pyobject, k):
                    value = getattr(self.pyobject, k)
                    if isinstance(value, PyObject):
                        value.__xpm__.validate()
                elif argument.required:
                    raise ValueError("Value %s is required but missing", k)


    
    @property
    def identifier(self):
        """Computes the unique identifier"""
        hashcomputer = HashComputer()
        hashcomputer.update(self.xpmtype.typename.name)
        if not self._identifier:
            for k, v in self.xpmtype.arguments.items():
                value = getattr(self.pyobject, k, None)
                if v.ignored:
                    continue
                if v.defaultvalue and v.defaultvalue == value:
                    # No update if same value
                    continue

                hashcomputer.update(value)
                self._identifier = hashcomputer.digest()
        return self._identifier

def clone(v):
    """Clone a value"""
    if isinstance(v, (str, float, int)):
        return v
    if isinstance(v, list):
        return [clone(x) for x in v]
    raise NotImplementedError("For type %s" % v)

class PyObjectMetaclass(type):
    def __getattr__(cls, key):
        """Access to a class field"""
        return cls.__xpm__.arguments[key]

class PyObject(metaclass=PyObjectMetaclass):
    """Base type for all objects in python interface"""

    def __init__(self, **kwargs):
        assert self.__class__.__xpm__, "No XPM type associated with this XPM object"

        # Add configuration
        self.__xpm__ = Information(self)

        # Initialize with arguments
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # Initialize with default arguments
        for k, v in self.__class__.__xpm__.arguments.items():
            if k not in kwargs and v.defaultvalue is not None:
                self.__setattr__(k, clone(v.defaultvalue))

    def __setattr__(self, name, value):
        if name != "__xpm__":
            return self.__xpm__.set(name, value)
        return super().__setattr__(name, value)

    def submit(self, *, workspace=None, launcher=None):
        """Submit this task"""
        if self.__xpm__.submitted:
            raise Exception("Task %s was already submitted" % self)
        if send:
            workspace = workspace or Workspace.CURRENT
            launcher = launcher or workspace.launcher
            assert workspace is not None, "No experiment has been defined"
            assert launcher is not None, "No launcher has been set"

            self.__class__.__xpm__.submit(workspace, launcher, self, self.__xpm__.dependencies)

        self.__xpm__.submitted = True
        return self

    def tag(self, name, value):
        self.__xpm__.tags[name] = value

    def _init(self):
        """Prepare object after creation"""
        pass

class Register():
    def __init__(self):
        self.tasks = {}

register = Register()


class Workspace():
    """A workspace
    """
    CURRENT = None

    """True if a job was submitted"""
    SUBMITTED = False

    """An experimental workspace"""
    def __init__(self, path: Path):
        # Initialize the base class
        self.path = path
        self.launcher = None

    @staticmethod
    def setcurrent(workspace: "Workspace"):
        """Set this workspace as being the default workspace for all the tasks"""
        Workspace.CURRENT = workspace

    def experiment(self, name):
        """Sets the current experiment name"""
        self.experiment_name = name

    @staticmethod
    def waitUntilTaskCompleted():
        raise NotImplementedError()

