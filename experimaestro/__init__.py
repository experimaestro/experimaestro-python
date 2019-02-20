# Import Python modules

import json
import sys
import inspect
import os.path as op
import os
import logging
import pathlib
from pathlib import Path as BasePath, PosixPath
from typing import Union

# --- Initialization

logger = logging.getLogger("xpm")
modulepath = BasePath(__file__).parent

# --- Import C bindings

from cffi import FFI
import re

ffi = FFI()
with open(modulepath / "api.h", "r") as fp:
    RE_SKIP = re.compile(r"""^\s*(?:#include|#ifn?def|#endif|#define|extern "C") .*""")
    RE_CALLBACK = re.compile(r"""^\s*typedef\s+\w+\s*\(\s*\*.*callback""")
    cdef = ""
    for line in fp:
        if RE_CALLBACK.match(line):
            cdef += "extern \"Python\" %s" % line
        elif not RE_SKIP.match(line):
            cdef += line

    ffi.cdef(cdef)

lib = ffi.dlopen(str(modulepath / "libexperimaestro.so"))

def cstr(s):
    return str(s).encode("utf-8")

def fromcstr(s):
    return ffi.string(s).decode("utf-8")

class FFIObject:
    @classmethod
    def _ptr(cls, self):
        return ffi.cast(f"{cls.__name__} *", self.ptr)

    @classmethod
    def fromcptr(cls, ptr, managed=True, notnull=False):
        """Wraps a returned pointer into an object"""
        if ptr == ffi.NULL:
            if notnull:
                raise ValueError("Null pointer")
            else:
                return None
        self = cls.__new__(cls)
        self.ptr = cls._wrap(ptr) if managed else ptr
        return self

    @classmethod
    def _wrap(cls, ptr):
        return ffi.gc(ptr, getattr(lib, "%s_free" % cls.__name__.lower()))

class Typename(FFIObject):
    def __init__(self, name: str):
        self.ptr = ffi.gc(lib.typename_new(cstr(name)), lib.typename_free)

    def __getattr__(self, key):
        return self(key)

    def __call__(self, key):
        tn = Typename.__new__(Typename)
        tn.ptr = ffi.gc(lib.typename_sub(self.ptr, cstr(key)), lib.typename_free)
        return tn

    def __str__(self):
        return fromcstr(lib.typename_name(self.ptr))


def typename(name: Union[str, Typename]):
    if isinstance(name, Typename):
        return name
    return Typename(str(name))

class Type(FFIObject):
    XPM2PYTHON = {}
    PYTHON2XPM = {}

    def __init__(self, tn: Union[str, Typename], parentType: "Type" = None): 
        self.ptr = ffi.gc(lib.type_new(typename(tn).ptr, parentType.ptr if parentType else ffi.NULL), lib.type_free)

    def __str__(self):
        return self.name()

    def name(self):
        return fromcstr(lib.type_tostring(Type._ptr(self)))

    def addArgument(self, argument: "Argument"):
        lib.type_addargument(Type._ptr(self), argument.ptr)

    def getArgument(self, key: str):
        return Argument.fromcptr(lib.type_getargument(Type._ptr(self), cstr(key)))

    def isArray(self):
        return lib.type_isarray(Type._ptr(self))

    @staticmethod
    def frompython(pythontype):
        if pythontype is None:
            return Type.fromcptr(lib.ANY_TYPE, managed=False)
        return Type.PYTHON2XPM.get(pythontype, None)

    def topython(self):
        return Type.XPM2PYTHON.get(self.name()).pythontypes[0]

class ArrayType(Type):  
    def __init__(self, type: Type):
        self.ptr = ffi.gc(lib.arraytype_new(Type._ptr(type)), lib.arraytype_free)      

class PredefinedType(Type):
    def __init__(self, ptr, pythontypes, topython, frompython):
        self.ptr = ptr
        self.pythontypes = pythontypes
        self.topython = topython
        self.frompython = frompython

        Type.XPM2PYTHON[str(self)] = self
        for pythontype in pythontypes:
            Type.PYTHON2XPM[pythontype] = self

BooleanType = PredefinedType(lib.BOOLEAN_TYPE, [bool],
    lambda v: v.asScalar().asBoolean(), lib.scalarvalue_fromboolean)
StringType = PredefinedType(lib.STRING_TYPE, [str],
    lambda v: v.asScalar().asString(), lambda s: lib.scalarvalue_fromstring(cstr(s)))
IntegerType = PredefinedType(lib.INTEGER_TYPE, [int],
    lambda v: v.asScalar().asInteger(), lib.scalarvalue_frominteger)
RealType = PredefinedType(lib.REAL_TYPE, [float],
    lambda v: v.asScalar().asReal(), lib.scalarvalue_fromreal)
PathType = PredefinedType(lib.PATH_TYPE, [BasePath, PosixPath],
    lambda v: v.asScalar().asPath(), 
    lambda p: lib.scalarvalue_frompathstring(cstr(str(p.absolute()))))

AnyType = Type.fromcptr(lib.ANY_TYPE, managed=False)

class Path(FFIObject):
    def __init__(self, path: str):
        self.ptr =  ffi.gc(lib.path_new(cstr(path)), lib.path_free)
    def str(self):
        return str(String(lib.path_string(self.ptr)))

    def localpath(self):
        s = String.fromcptr(lib.path_localpath(self.ptr))
        if s is None:
            raise ValueError("Path %s is not local", self)
        return str(s)

class AbstractCommandComponent(FFIObject): pass
class CommandPath(AbstractCommandComponent):
    def __init__(self, path: [Path, str]):
        self.ptr =  ffi.gc(lib.commandpath_new(aspath(path).ptr), lib.commandpath_free)

class CommandString(AbstractCommandComponent):
    def __init__(self, string: str):
        self.ptr =  ffi.gc(lib.commandstring_new(cstr(string)), lib.commandstring_free)

class CommandParameters(AbstractCommandComponent):
    def __init__(self):
        self.ptr =  ffi.gc(lib.commandparameters_new(), lib.commandparameters_free)


class AbstractCommand(FFIObject): pass

class Command(AbstractCommand):
    def __init__(self):
        self.ptr = ffi.gc(lib.command_new(), lib.command_free)

    def add(self, component: AbstractCommandComponent):
        lib.command_add(Command._ptr(self), AbstractCommandComponent._ptr(component))


class CommandLine(Command):
    def __init__(self):
        self.ptr = ffi.gc(lib.commandline_new(), lib.commandline_free)

    def add(self, command: Command):
        lib.commandline_add(self.ptr, Command._ptr(command))

class Dependency(FFIObject): pass

class DependencyArray:
    def __init__(self):
        self.ptr = ffi.gc(lib.dependencyarray_new(), lib.dependencyarray_free)

    def add(self, dependency: Dependency):
        lib.dependencyarray_add(self.ptr, Dependency._ptr(dependency))

class StringArray:
    def __init__(self):
        self.ptr = ffi.gc(lib.stringarray_new(), lib.stringarray_free)
    def add(self, string: str):
        lib.stringarray_add(self.ptr, cstr(string))

class Task(FFIObject):
    def __init__(self, tasktype: Type, *, taskId:Typename=None):
        tn_ptr = typename(taskId).ptr if taskId else ffi.NULL
        self.ptr =  ffi.gc(lib.task_new(tn_ptr, tasktype.ptr), lib.task_free)

    def name(self):
        return fromcstr(lib.typename_name(lib.task_name(self.ptr)))

    def commandline(self, commandline: CommandLine):
        lib.task_commandline(self.ptr, commandline.ptr)

    def submit(self, workspace, launcher, value, dependencies: DependencyArray):
        Workspace.SUBMITTED = True
        lib.task_submit(self.ptr, workspace.ptr, Launcher._ptr(launcher), Value._ptr(value), dependencies.ptr)

    @classmethod
    def isRunning(cls):
        return lib.task_isrunning()

def aspath(path: Union[str, Path]):
    if isinstance(path, Path): 
        return path
    return Path(path)

class String(FFIObject): 
    def __str__(self):
        return fromcstr(lib.string_ptr(self.ptr))

class Job(FFIObject):
    def stdoutPath(self):
        return Path.fromcptr(lib.job_stdoutpath(self.ptr))
    def stderrPath(self):
        return Path.fromcptr(lib.job_stderrpath(self.ptr))

class Value(FFIObject): 
    def type(self):
        return Type.fromcptr(lib.value_gettype(Value._ptr(self)))

    def __str__(self):
        return String.fromcptr(lib.value_tostring(Value._ptr(self))).__str__()

    def isMap(self):
        return lib.value_ismap(Value._ptr(self))

    def asMap(self):
        ptr = lib.value_asmap(Value._ptr(self))   
        if ptr == ffi.NULL:
            raise ValueError("Value is not a map: %s" % self)
        return MapValue.fromcptr(ptr)

    def asArray(self):
        ptr = lib.value_asarray(Value._ptr(self))   
        if ptr == ffi.NULL:
            raise ValueError("Value is not an array: %s" % self)
        return ArrayValue.fromcptr(ptr)

    def asScalar(self):
        ptr = lib.value_asscalar(Value._ptr(self))   
        if ptr == ffi.NULL:
            raise ValueError("Value is not a scalar: %s" % self)
        return ScalarValue.fromcptr(ptr)

    def isNull(self):
        return lib.scalarvalue_isnull(ScalarValue._ptr(self))

    def tags(self):
        iterator = ffi.gc(lib.value_tags(Value._ptr(self)), lib.tagvalueiterator_free)
        m = {}
        while lib.tagvalueiterator_next(iterator):
            key = fromcstr(lib.tagvalueiterator_key(iterator))
            value = ScalarValue.fromcptr(lib.tagvalueiterator_value(iterator))
            m[key] = value.toPython()
        return m

    def toPython(self):
        """Converts a value into a Python object"""

        # Returns object if it is one
        object = self.asMap().object if self.isMap() else None
        if object:
            return object.pyobject

        # Returns array
        svtype = self.type()
        if svtype.isArray():
            array  = self.asArray()
            r = []
            for i in range(len(array)):
                r.append(array[i].toPython())
            return r

        if self.isNull(): return None

        return Type.XPM2PYTHON.get(str(svtype), None).topython(self)

    @staticmethod
    def frompython(value):
        """Transforms a Python value into a structured value"""
        # Simple case: it is already a configuration
        if isinstance(value, Value):
            return value

        # It is a PyObject: get the associated configuration
        if isinstance(value, PyObject):
            return value.__xpm__.sv

        # A dictionary: transform
        if isinstance(value, dict):
            v = register.build(JSON_ENCODER.encode(value))
            return v

        # A list
        if isinstance(value, list):
            newvalue = ArrayValue()
            for v in value:
                newvalue.add(Value.frompython(v))

            return newvalue

        # For anything else, we try to convert it to a value
        return ScalarValue(value)

class ComplexValue(Value):
    def setTagContext(self, key: str):
        lib.complexvalue_settagcontext(ComplexValue._ptr(self), cstr(key))



class MapValue(ComplexValue):
    def __init__(self):
        self.ptr = ffi.gc(lib.mapvalue_new(), lib.mapvalue_free)

    @property
    def object(self):
        object = lib.mapvalue_getobjecthandle(self.ptr)
        if object == ffi.NULL: 
            checkexception()
            raise Exception()
        return ffi.from_handle(object)

    @object.setter
    def object(self, object):
        lib.mapvalue_setobject(self.ptr, object.ptr)

    def set(self, key: str, value: Value):
        lib.mapvalue_set(self.ptr, cstr(key), Value._ptr(value))

    @property
    def job(self): 
        return Job.fromcptr(lib.mapvalue_getjob(self.ptr))

    @property
    def type(self): raise NotImplementedError()

    @type.setter
    def type(self, type: Type):
        return lib.mapvalue_settype(self.ptr, Type._ptr(type))

    def addTag(self, key, value):
        lib.mapvalue_addtag(self.ptr, cstr(key), Value.frompython(value).ptr)

class ScalarValue(Value):
    def __init__(self, value):
        if value is None:
            self.ptr = scalarvalue_new()
        else:
            predefinedType = Type.PYTHON2XPM.get(type(value), None)
            if predefinedType is None:
                raise NotImplementedError("Cannot create scalar from %s", type(value))
            self.ptr = predefinedType.frompython(value)

        self.ptr = ffi.gc(self.ptr, lib.scalarvalue_free)       

    def asReal(self):
        return lib.scalarvalue_asreal(self.ptr) 
    def asInteger(self):
        return lib.scalarvalue_asinteger(self.ptr) 
    def asBoolean(self):
        return lib.scalarvalue_asboolean(self.ptr) 
    def asString(self):
        s = String.fromcptr(lib.scalarvalue_asstring(self.ptr))
        return str(s)
    def asPath(self):
        return Path.fromcptr(lib.scalarvalue_aspath(self.ptr))

    def tag(self, key:str):
        lib.scalarvalue_tag(self.ptr, cstr(key))
        

class ArrayValue(Value):
    def __init__(self):
        self.ptr = ffi.gc(lib.arrayvalue_new(), lib.arrayvalue_free)

    def __len__(self):
        return lib.arrayvalue_size(self.ptr)

    def __getitem__(self, index: int):
        return Value.fromcptr(lib.arrayvalue_get(self.ptr, index))

    def add(self, value: Value):
        lib.arrayvalue_add(self.ptr, Value._ptr(value))

class Connector(FFIObject): 
    pass

class LocalConnector(Connector):
    def __init__(self):
        self.ptr = ffi.gc(lib.localconnector_new(), lib.localconnector_free)


class Launcher(FFIObject):
    @property
    def launcherPtr(self):
        return ffi.cast("Launcher *", self.ptr)

    def setenv(self, key: str, value: str):
        lib.launcher_setenv(Launcher._ptr(self), cstr(key), cstr(value))

    def setNotificationURL(self, url: str):
        lib.launcher_setnotificationURL(Launcher._ptr(self), cstr(url))

    @staticmethod
    def defaultLauncher():
        return Launcher.fromcptr(lib.launcher_defaultlauncher())

class DirectLauncher(Launcher):
    def __init__(self, connector: Connector):
        self.ptr = ffi.gc(lib.directlauncher_new(Connector._ptr(connector)), lib.directlauncher_free)

class Generator(FFIObject): pass

class PathGenerator(Generator):
    def __init__(self, path: str):
        self.ptr = PathGenerator._wrap(lib.pathgenerator_new(cstr(path)))

class Token(FFIObject): pass
class CounterToken(Token):
    def __init__(self, tokens: int):
        self.ptr = CounterToken._wrap(lib.countertoken_new(tokens))
    def createDependency(self, count: int):
        return Dependency.fromcptr(lib.countertoken_createdependency(self.ptr, count))

# --- Utilities and constants

class JSONEncoder(json.JSONEncoder):
    """A JSON encoder for Python objects"""
    def default(self, o):
        if type(o) == Typename:
            return str(o)

        if isinstance(o, BasePath):
            return {"$type": "path", "$value": str(o.resolve())}

        return json.JSONEncoder.default(self, o)


# Json encoder
JSON_ENCODER = JSONEncoder()

# Flag for simulating
SUBMIT_TASKS = True

# Default launcher
DEFAULT_LAUNCHER = None


# --- XPM Objects

def callback(args):
    def wrapper(function):
        def _wrapped(*args, **kwargs):
            try:
                function(*args, **kwargs)
                return 0
            except Exception:
                tb.print_exc()
                return 1
        return ffi.callback(args)(_wrapped)
    return wrapper

@callback("int(void *, CString, Value *)")
def object_setvalue_cb(handle, key, value):
    ffi.from_handle(handle).setValue(fromcstr(key), Value.fromcptr(value, managed=False))

@callback("int(void *)")
def object_init_cb(handle):
    ffi.from_handle(handle).init()

@callback("int(void *)")
def object_delete_cb(handle):
    logging.info("Deleting object")
    object = ffi.from_handle(handle)
    logging.debug("Deleting object of type %s [%s]", object.pyobject.__class__.__name__, object)
    del XPMObject.OBJECTS[handle]

class XPMObject(FFIObject):
    OBJECTS = {}

    """Holds XPM information for a PyObject"""
    def __init__(self, pyobject, sv=None):
        handle = ffi.new_handle(self)
        self.ptr = ffi.gc(lib.object_new(handle, object_init_cb, object_delete_cb, object_setvalue_cb), lib.object_free)
        # Keep a copy of the handle
        XPMObject.OBJECTS[handle] = self

        self.pyobject = pyobject
        logging.debug("Created object of type %s [%s]", self.pyobject.__class__.__name__, handle)

        if sv is None:
            self.sv = MapValue()
        else:
            self.sv = sv.asMap()

        self.sv.object = self
        self.sv.type = self.pyobject.__class__.__xpmtype__
        self.setting = False
        self.submitted = False
        self.dependencies = DependencyArray()

    @property
    def job(self):
        job = self.sv.job
        if job: return job
        raise Exception("No job associated with value %s" % self.sv)
        
    def set(self, k, v):
        if self.setting: return

        logger.debug("Called set: %s, %s (%s)", k, v, type(v))
        try:
            self.setting = True
            # Check if the value corresponds to a task; if so,
            # raise an exception if the task was not submitted
            if isinstance(v, PyObject) and hasattr(v.__class__, "__xpmtask__"):
                if not v.__xpm__.submitted:
                    raise Exception("Task for argument '%s' was not submitted" % k)
            pv = Value.frompython(v)
            self.sv.set(k, pv)
        except:
            logger.error("Error while setting %s", k)
            raise
        finally:
            self.setting = False


    def setValue(self, key, sv):
        """Called by XPM when value has been validated"""
        if self.setting: return
        try:
            self.setting = True
            if sv is None:
                value = None
                svtype = None
            else:
                value = sv.toPython()
                svtype = sv.type()

            # Set the value on the object if not setting otherwise
            logger.debug("Really setting %s to %s [%s => %s] on %s", key, value,
                    svtype, type(value), type(self.pyobject))
            setattr(self.pyobject, key, value)
        finally:
            self.setting = False
    
    def run(self):
        self.pyobject.execute()

    def init(self):
        self.pyobject._init()

class PyObject:
    """Base type for all objects in python interface"""

    def __init__(self, **kwargs):
        assert self.__class__.__xpmtype__, "No XPM type associated with this XPM object"

        # Add configuration
        self.__xpm__ = XPMObject(self)

        # Initialize with arguments
        for k, v in kwargs.items():
            self.__xpm__.set(k, v)

    def submit(self, *, workspace=None, launcher=None, send=SUBMIT_TASKS):
        """Submit this task"""
        if self.__xpm__.submitted:
            raise Exception("Task %s was already submitted" % self)
        if send:
            launcher = launcher or DEFAULT_LAUNCHER
            workspace = workspace or Workspace.DEFAULT
            self.__class__.__xpmtask__.submit(workspace, launcher, self.__xpm__.sv, self.__xpm__.dependencies)

        self.__xpm__.submitted = True
        return self

    def __setattr__(self, name, value):
        if not Task.isRunning:
            # If task is not running, we update the structured
            # value
            if name != "__xpm__":
                self.__xpm__.set(name, value)
        super().__setattr__(name, value)

    def _init(self):
        """Prepare object after creation"""
        pass

    
    def _stdout(self):
        return self.__xpm__.job.stdoutPath().localpath()
    def _stderr(self):
        return self.__xpm__.job.stderrPath().localpath()

    def _adddependency(self, dependency):
        self.__xpm__.dependencies.add(dependency)


# Another way to submit if the method is overriden
def submit(*args, **kwargs):
    PyObject.submit(*args, **kwargs)

# Defines a class property
PyObject.__xpmtype__ = lib.ANY_TYPE

class TypeProxy: pass

class ArrayOf(TypeProxy):
    """Array of object"""
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, register):
        type = register.getType(self.cls)
        return ArrayType(type)

class Choice(TypeProxy):
    def __init__(self, *args):
        self.choices = args

    def __call__(self, register):
        return StringType

import traceback as tb
@ffi.callback("Object * (void * handle, Value *)")
def register_create_object_callback(handle, value):
    try:
        object = ffi.from_handle(handle).createObject(Value.fromcptr(value, managed=False))
        return object.ptr
    except Exception as e:
        tb.print_exc()
        logger.error("Error while creating object: %s", e)
        return None

@ffi.callback("int (void * handle, Task *, Value *)")
def register_run_task_callback(handle, task, value):
    try:
        ffi.from_handle(handle).runTask(Task.fromcptr(task, managed=False), Value.fromcptr(value, managed=False))
        return 0
    except Exception as e:
        tb.print_exc()
        logger.error("Error while running task: %s", e)
        return 1



class Register(FFIObject):
    """The register contains a reference"""
    def __init__(self):
        # Initialize the base class
        self.handle = ffi.new_handle(self)
        self.ptr = ffi.gc(lib.register_new(self.handle, register_create_object_callback, register_run_task_callback), 
            lib.register_free)
        self.registered = {}


    def associateType(self, pythonType, xpmType):
        pythonType.__xpmtype__ = xpmType
        self.registered[xpmType.name()] = pythonType

    def addTask(self, task: Task):
        lib.register_addTask(self.ptr, task.ptr)

    def addType(self, pythonType, typeName, parentType, description=None):
        xpmType = Type(typeName, parentType)
        if description is not None:
            xpmType.description(description)

        self.associateType(pythonType, xpmType)
        lib.register_addType(self.ptr, xpmType.ptr)

    def getType(self, key):
        """Returns the Type object corresponding to the given type or None if not found
        """
        logger.debug("Searching for type %s", key)

        if key is None:
            return AnyType

        if isinstance(key, Type):
            return key

        if isinstance(key, TypeProxy):
            return key(self)

        if isinstance(key, type):
            t = Type.frompython(key)
            if t is None:
                return getattr(key, "__xpmtype__", None)
            return t

        if isinstance(key, PyObject):
            return key.__class__.__xpmtype__
        
        if isinstance(key, Typename):
            return self.registered.get(str(key), None)

        return None

    def getTask(self, name):
        lib.register_getTask()


    def runTask(self, task: Task, value: Value):
        logger.info("Running %s", task)
        value.asMap().object.run()

    def build(self, string: str):
        return Value.fromcptr(lib.register_build(self.ptr, cstr(string)))

    def createObject(self, sv: Value):
        type = self.registered.get(sv.type().name(), PyObject)
        logger.debug("Creating object for %s [%s]", sv, type)
        pyobject = type.__new__(type)
        pyobject.__xpm__ = XPMObject(pyobject, sv=sv)
        logger.debug("Preparing object for %s", type)
        return pyobject.__xpm__

    def parse(self, arguments=None, try_parse=False):
        if arguments is None:
            arguments = sys.argv[1:]
        array = StringArray()
        for argument in arguments:
            array.add(argument)
        return lib.register_parse(self.ptr, array.ptr, try_parse)


    def try_parse(self, arguments=None):
        return self.parse(arguments, True)

register = Register()


# --- Annotations to define tasks and types

class RegisterType:
    """Annotations for experimaestro types"""
    def __init__(self, qname, description=None, associate=False):
        if type(qname) == Typename:
            self.qname = qname
        else:
            self.qname = Typename(qname)
        self.description = description
        self.associate = associate

    def __call__(self, t):
        # Check if conditions are fullfilled
        xpmType = None
        if self.qname:
            xpmType = register.getType(self.qname)
            if xpmType is not None and not self.associate:
                raise Exception("XPM type %s is already declared" % self.qname)
            if self.associate and xpmType is None:
                raise Exception("XPM type %s is not already declared" % self.qname)

        # Add XPM object if needed
        if not issubclass(t, PyObject):
            __bases__ = (PyObject, ) + t.__bases__
            t = type(t.__name__, __bases__, dict(t.__dict__))

        # Find first registered ancestor
        parentinfo = None
        for subtype in t.__mro__[1:]:
            if issubclass(subtype, PyObject) and subtype != PyObject:
                parentinfo = register.getType(subtype)
                if parentinfo is not None:
                    logger.debug("Found super info %s for %s", parentinfo, t)
                    break

        # Register
        if self.associate:
            register.associateType(t, xpmType)
        else:
            register.addType(t, self.qname, parentinfo)

        return t


class AssociateType(RegisterType):
    """Annotation to associate one class with an XPM type"""
    def __init__(self, qname, description=None):
        super().__init__(qname, description=description, associate=True)



class RegisterTask(RegisterType):
    """Register a task"""

    def __init__(self, qname, scriptpath=None, pythonpath=None, prefix_args=[], description=None, associate=None):
        super().__init__(qname, description=description, associate=associate)
        self.pythonpath = sys.executable if pythonpath is None else pythonpath
        self.scriptpath = scriptpath
        self.prefix_args = prefix_args

    def __call__(self, t):
        # Register the type
        t = super().__call__(t)
        
        if not issubclass(t, PyObject):
            raise Exception("Only experimaestro objects (annotated with RegisterType or AssociateType) can be tasks")

        if self.scriptpath is None:
            self.scriptpath = inspect.getfile(t)
        else:
            self.scriptpath = op.join(
                op.dirname(inspect.getfile(t)), self.scriptpath)

        self.scriptpath = BasePath(self.scriptpath).absolute()

        logger.debug("Task %s command: %s %s", t, self.pythonpath,
                     self.scriptpath)
        for mro in t.__mro__:
            pyType = register.getType(mro)
            if pyType is not None:
                break
        if pyType is None:
            raise Exception(
                "Class %s has no associated experimaestro type" % t)
        task = Task(pyType)
        t.__xpmtask__ = task
        register.addTask(task)

        command = Command()
        command.add(CommandPath(self.pythonpath))
        command.add(CommandPath(op.realpath(self.scriptpath)))
        for arg in self.prefix_args:
            command.add(CommandString(arg))
        command.add(CommandString("run"))
        command.add(CommandString("--json-file"))
        command.add(CommandParameters())
        command.add(CommandString(task.name()))
        commandLine = CommandLine()
        commandLine.add(command)
        task.commandline(commandLine)

        return t


class Argument(FFIObject):
    """Abstract class for all arguments (standard, path, etc.)"""

    def __init__(self, name: str, argtype, help=""):
        self.ptr = ffi.gc(lib.argument_new(cstr(name)), lib.argument_free)

        if argtype:
            lib.argument_settype(self.ptr, Type._ptr(argtype))
        lib.argument_sethelp(self.ptr, cstr(help))


    def __call__(self, t):
        xpminfo = register.getType(t)
        if xpminfo is None:
            raise Exception("%s is not an XPM type" % t)

        xpminfo.addArgument(self)
        return t

    @property
    def name(self):
        return fromcstr(lib.argument_getname(self.ptr))
    
    @property
    def type(self):
        return Type.fromcptr(lib.argument_gettype(self.ptr))
    
    @property
    def help(self):
        return fromcstr(lib.argument_gethelp(self.ptr))
    
    @property
    def defaultvalue(self):
        return Value.fromcptr(lib.argument_getdefaultvalue(self.ptr))

class TypeArgument(Argument):
    """Defines an argument for an experimaestro type"""
    def __init__(self, name, type=None, default=None, required=None,
                 help=None, ignored=False):
        xpmtype = register.getType(type)
        logger.debug("Registering type argument %s [%s -> %s]", name, type,
                      xpmtype)
        Argument.__init__(self, name, xpmtype, help=help)
        if default is not None and required is not None and required:
            raise Exception("Argument is required but default value is given")

        lib.argument_setignored(self.ptr, ignored)
        
        required = (default is None) if required is None else required
        lib.argument_setrequired(self.ptr, required)
        if default is not None:
            value = Value.frompython(default)
            lib.argument_setdefault(self.ptr, Value._ptr(value))


class PathArgument(Argument):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""
    def __init__(self, name, path, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path
        """
        Argument.__init__(self, name, PathType, help=help)
        generator = PathGenerator(path)
        lib.argument_setgenerator(self.ptr, Generator._ptr(generator))

class ConstantArgument(Argument):
    """
    An constant argument (useful for versionning tasks)
    """
    def __init__(self, name: str, value, help=""):
        value = Value.frompython(value)
        xpmtype = register.getType(value)
        super().__init__(name, xpmtype, help=help)
        lib.argument_setconstant(self.ptr, Value._ptr(value))


# --- Export some useful functions

class _Definitions:
    """Allow easy access to XPM tasks with dot notation to tasks or types"""

    def __init__(self, retriever, path=None):
        self.__retriever = retriever
        self.__path = path

    def __getattr__(self, name):
        if name.startswith("__"):
            return object.__getattr__(self, name)
        if self.__path is None:
            return Definitions(self.__retriever, name)
        return Definitions(self.__retriever, "%s.%s" % (self.__path, name))

    def __call__(self, *args, **options):
        definition = self.__retriever(self.__path)
        if definition is None:
            raise AttributeError("Task/Type %s not found" % self.__path)
        return definition.__call__(*args, **options)


types = _Definitions(register.getType)
tasks = _Definitions(register.getTask)


EXCEPTIONS = {
    lib.ERROR_RUNTIME: RuntimeError
}



def checkexception():
    code = lib.lasterror_code()
    if code != lib.ERROR_NONE:
        raise EXCEPTIONS.get(code, Exception)(fromcstr(lib.lasterror_message()))


class Workspace():
    DEFAULT = None

    """True if a job was submitted"""
    SUBMITTED = False

    """An experimental workspace"""
    def __init__(self, path):
        # Initialize the base class
        self.ptr = ffi.gc(lib.workspace_new(cstr(path)), lib.workspace_free)

    def current(self):
        """Set this workspace as being the default workspace for all the tasks"""
        lib.workspace_current(self.ptr)

    def experiment(self, name):
        """Sets the current experiment name"""
        lib.workspace_experiment(self.ptr, cstr(name))

    def server(self, port: int):
        lib.workspace_server(self.ptr, port, cstr(modulepath / "htdocs"))
        checkexception()

Workspace.waitUntilTaskCompleted = lib.workspace_waitUntilTaskCompleted


def experiment(path, name):
    """Defines an experiment
    
    :param path: The working directory for the experiment
    :param name: The name of the experiment
    """
    if isinstance(path, BasePath):
        path = path.absolute()
    workspace = Workspace(str(path))
    workspace.current()
    workspace.experiment(name)
    Workspace.DEFAULT = workspace
    return workspace

def set_launcher(launcher):
    global DEFAULT_LAUNCHER
    DEFAULT_LAUNCHER = launcher

launcher = DirectLauncher(LocalConnector())
if os.getenv("PYTHONPATH"):
    launcher.setenv("PYTHONPATH", os.getenv("PYTHONPATH"))

set_launcher(launcher)

def tag(name: str, x, object:PyObject=None, context=None):
    """Tag a value"""
    if object:
        if not hasattr(object, "__xpm__"):
            object = sv = Value.frompython(object).asMap()
        else:
            sv = object.__xpm__.sv # type: MapValue
        sv.addTag(name, x)
        if context:
            sv.setTagContext(context)
        return object

    value = ScalarValue(x)
    value.tag(name)
    return value

def tags(value):
    """Return the tags associated with a value"""
    if isinstance(value, Value):
        return value.tags()
    return value.__xpm__.sv.tags()

def tagspath(value: PyObject):
    """Return the tags associated with a value"""
    p = BasePath()
    for key, value in value.__xpm__.sv.tags().items():
        p /= "%s=%s" % (key.replace("/","-"), value)
    return p

# --- Handle signals

import atexit
import signal

EXIT_MODE = False

def handleKill():
    EXIT_MODE = True
    logger.warn("Received SIGINT or SIGTERM")
    sys.exit(0)

signal.signal(signal.SIGINT, handleKill)
signal.signal(signal.SIGTERM, handleKill)
signal.signal(signal.SIGQUIT, handleKill)

@atexit.register
def handleExit():
    if Workspace.SUBMITTED:
        logger.info("End of script: waiting for jobs to be completed")
        lib.workspace_waitUntilTaskCompleted()
    lib.stopping()


LogLevel_TRACE = lib.LogLevel_TRACE
LogLevel_DEBUG = lib.LogLevel_DEBUG
LogLevel_INFO = lib.LogLevel_INFO
LogLevel_WARN = lib.LogLevel_WARN
LogLevel_ERROR = lib.LogLevel_ERROR
def setLogLevel(key: str, level):
    lib.setLogLevel(cstr(key), level)

def progress(value: float):
    lib.progress(value)