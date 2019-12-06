# Import Python modules

import json
import sys
import inspect
import os.path as op
import os
import logging
import pathlib
from pathlib import Path, PosixPath
from typing import Union, Dict

import experimaestro.api as api
from .api import PyObject, DirectLauncher, LocalConnector, Workspace, Typename, Launcher, logger, register

# --- Annotations to define tasks and types
        
class Type(api.Type):
    REGISTERED:Dict[Typename, PyObject] = {}

    """Annotations for experimaestro types"""
    def __init__(self, typename=None, description=None):
        super().__init__(typename, description)

    def __call__(self, t):
        # Check if conditions are fullfilled
        if self.typename is None:
            self.typename = Typename("%s.%s" % (t.__module__, t.__name__))

        if self.typename in Type.REGISTERED:
            raise Exception("Experimaestro type %s is already declared" % self.typename)
        Type.REGISTERED[self.typename] = t
        
        # --- Add PyObject as an ancestor of t if needed
        if not issubclass(t, api.PyObject):
            __bases__ = (api.PyObject, )
            if t.__bases__ != (object, ):
                __bases__ += t.__bases__
            __dict__ = {key: value for key, value in t.__dict__.items() if key not in ["__dict__"]}
            t = type(t.__name__, __bases__, __dict__)

        t.__xpm__ = self

        self.objecttype = t
        return t

    def validate(self, value):
        if not isinstance(value, PyObject):
            raise ValueError("%s is not an experimaestro type or task", value)
        if not isinstance(value, self.objecttype):
            raise ValueError("%s is not a subtype of %s")
        return value

# --- Type proxies are useful shortcuts in python


class Array(api.TypeProxy):
    """Array of object"""
    def __init__(self, type):
        self.type = api.Type.fromType(type)

    def __call__(self):
        return api.ArrayType(self.type)

class Choice(api.TypeProxy):
    def __init__(self, *args):
        self.choices = args

    def __call__(self):
        return api.StringType


class Task(Type):
    """Register a task"""

    def __init__(self, typename, scriptpath=None, pythonpath=None, prefix_args=[], description=None, associate=None):
        super().__init__(typename, description=description)
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

        self.scriptpath = Path(self.scriptpath).absolute()

        logger.debug("Task %s command: %s %s", t, self.pythonpath,
                     self.scriptpath)
 
        # Construct command       
        command = api.Command()
        command.add(api.CommandPath(self.pythonpath))
        command.add(api.CommandPath(op.realpath(self.scriptpath)))
        for arg in self.prefix_args:
            command.add(api.CommandString(arg))
        command.add(api.CommandString("run"))
        command.add(api.CommandString("--json-file"))
        command.add(api.CommandParameters())
        command.add(api.CommandString(self.typename))
        commandLine = api.CommandLine()
        commandLine.add(command)
        self.commandline = commandLine

        return t


class Argument(api.BaseArgument):
    """Defines an argument for an experimaestro type"""
    def __init__(self, name, type=None, default=None, required=None,
                 ignored=False, help=None):

        super().__init__(name, type, help=help)

        required = (default is None) if required is None else required
        if default is not None and required is not None and required:
            raise Exception("Argument is required but default value is given")

        self.ignored = ignored
        self.defaultvalue = default
        self.required = required

class PathArgument(api.BaseArgument):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""
    def __init__(self, name, path, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path
        """
        super().__init__(name, type=Path, help=help)
        from .generators import PathGenerator
        self.generator = PathGenerator(path)

class ConstantArgument(api.BaseArgument):
    """
    An constant argument (useful for versionning tasks)
    """
    def __init__(self, name: str, value, help=""):
        value = Value.frompython(value)
        xpmtype = register.getType(value)
        super().__init__(name, xpmtype, help=help)
        self.constant = True

class TaggedValue:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value

def tag(name: str, value):
    """Tag a value"""
    return TaggedValue(name, value)

def tags(value):
    """Return the tags associated with a value"""
    if isinstance(value, Value):
        return value.tags()
    return value.__xpm__.sv.tags()

def tagspath(value: api.PyObject):
    """Return the tags associated with a value"""
    p = Path()
    for key, value in value.__xpm__.sv.tags().items():
        p /= "%s=%s" % (key.replace("/","-"), value)
    return p



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
