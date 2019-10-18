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

import experimaestro.api as api
from .api import register, DirectLauncher, LocalConnector, Workspace, Typename, Launcher, logger

# --- Annotations to define tasks and types

class Type:
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
        if not issubclass(t, api.PyObject):
            __bases__ = (api.PyObject, ) + t.__bases__
            t = type(t.__name__, __bases__, dict(t.__dict__))

        # Find first registered ancestor
        parentinfo = None
        for subtype in t.__mro__[1:]:
            if issubclass(subtype, api.PyObject) and subtype != api.PyObject:
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


# --- Type proxies are useful shortcuts in python


class Array(api.TypeProxy):
    """Array of object"""
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, register):
        type = register.getType(self.cls)
        return api.ArrayType(type)

class Choice(api.TypeProxy):
    def __init__(self, *args):
        self.choices = args

    def __call__(self, register):
        return api.StringType



class Task(Type):
    """Register a task"""

    def __init__(self, qname, scriptpath=None, pythonpath=None, prefix_args=[], description=None, associate=None):
        super().__init__(qname, description=description, associate=associate)
        self.pythonpath = sys.executable if pythonpath is None else pythonpath
        self.scriptpath = scriptpath
        self.prefix_args = prefix_args

    def __call__(self, t):
        # Register the type
        t = super().__call__(t)
        
        if not issubclass(t, api.PyObject):
            raise Exception("Only experimaestro objects (annotated with Type or AssociateType) can be tasks")

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
        task = api.Task(pyType)
        t.__xpmtask__ = task
        register.addTask(task)

        # Construct command       
        command = api.Command()
        command.add(api.CommandPath(self.pythonpath))
        command.add(api.CommandPath(op.realpath(self.scriptpath)))
        for arg in self.prefix_args:
            command.add(api.CommandString(arg))
        command.add(api.CommandString("run"))
        command.add(api.CommandString("--json-file"))
        command.add(api.CommandParameters())
        command.add(api.CommandString(task.name()))
        commandLine = api.CommandLine()
        commandLine.add(command)
        task.commandline(commandLine)

        return t



class Argument(api.Argument):
    """Defines an argument for an experimaestro type"""
    def __init__(self, name, type=None, default=None, required=None,
                 help=None, ignored=False):
        xpmtype = register.getType(type)
        logger.debug("Registering type argument %s [%s -> %s]", name, type,
                      xpmtype)
        api.Argument.__init__(self, name, xpmtype, help=help)
        if default is not None and required is not None and required:
            raise Exception("Argument is required but default value is given")

        api.lib.argument_setignored(self.ptr, ignored)
        
        required = (default is None) if required is None else required
        api.lib.argument_setrequired(self.ptr, required)
        if default is not None:
            value = api.Value.frompython(default)
            api.lib.argument_setdefault(self.ptr, api.Value._ptr(value))


class PathArgument(api.Argument):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""
    def __init__(self, name, path, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path
        """
        Argument.__init__(self, name, PathType, help=help)
        generator = PathGenerator(path)
        api.lib.argument_setgenerator(self.ptr, Generator._ptr(generator))

class ConstantArgument(api.Argument):
    """
    An constant argument (useful for versionning tasks)
    """
    def __init__(self, name: str, value, help=""):
        value = Value.frompython(value)
        xpmtype = register.getType(value)
        super().__init__(name, xpmtype, help=help)
        api.lib.argument_setconstant(self.ptr, Value._ptr(value))


# --- Export some useful functions



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
    Launcher.DEFAULT = launcher

launcher = DirectLauncher(LocalConnector())
if os.getenv("PYTHONPATH"):
    launcher.setenv("PYTHONPATH", os.getenv("PYTHONPATH"))

set_launcher(launcher)

def tag(name: str, x, object:api.PyObject=None, context=None):
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

def tagspath(value: api.PyObject):
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
        api.lib.workspace_waitUntilTaskCompleted()
    api.lib.stopping()


LogLevel_TRACE = api.lib.LogLevel_TRACE
LogLevel_DEBUG = api.lib.LogLevel_DEBUG
LogLevel_INFO = api.lib.LogLevel_INFO
LogLevel_WARN = api.lib.LogLevel_WARN
LogLevel_ERROR = api.lib.LogLevel_ERROR

def setLogLevel(key: str, level):
    api.lib.setLogLevel(cstr(key), level)

def progress(value: float):
    api.lib.progress(value)


# --- Deprecated

def deprecateClass(klass):
    import inspect
    def __init__(self, *args, **kwargs):
        frameinfo = inspect.stack()[1]
        logger.warning("Class %s is deprecated: use %s in %s:%s (%s)", 
            klass.__name__, klass.__bases__[0].__name__, 
            frameinfo.filename, frameinfo.lineno,
            frameinfo.code_context
        )
        super(klass, self).__init__(*args, **kwargs)
        
    klass.__init__ = __init__
    return klass

@deprecateClass
class RegisterType(Type): pass

@deprecateClass
class RegisterTask(Task): pass

@deprecateClass
class TypeArgument(Argument): pass
