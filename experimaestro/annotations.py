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
from .api import PyObject, Typename
from .utils import logger
from .workspace import Workspace
import experimaestro.commandline as commandline

# --- Annotations to define tasks and types
        
class Type:

    """Annotations for experimaestro types"""
    def __init__(self, typename=None, description=None, parents=[]):
        super().__init__()
        self.typename = typename
        self.description = description
        self.parents = parents

    def __call__(self, objecttype):
        # Check if conditions are fullfilled
        if self.typename is None:
            self.typename = Typename("%s.%s" % (objecttype.__module__.lower(), objecttype.__name__.lower()))

        originaltype = objecttype
        
        # --- If this is a method, encapsulate
        if inspect.isfunction(objecttype):
            objecttype = api.getfunctionpyobject(objecttype, self.parents)
        
        # --- Add PyObject as an ancestor of t if needed
        elif inspect.isclass(objecttype):
            assert not self.parents, "parents can be used only for functions"
            if not issubclass(objecttype, api.PyObject):
                __bases__ = (api.PyObject, )
                if objecttype.__bases__ != (object, ):
                    __bases__ += objecttype.__bases__
                __dict__ = {key: value for key, value in objecttype.__dict__.items() if key not in ["__dict__"]}
                objecttype = type(objecttype.__name__, __bases__, __dict__)


        else:
            raise ValueError("Cannot use type %s as a type/task" % objecttype)

        logging.debug("Registering %s", self.typename)
        
        xpmtype = api.ObjectType(objecttype, self.typename, self.description)
        objecttype.__xpm__ = xpmtype
        xpmtype.originaltype = originaltype
        return objecttype


class Array(api.TypeProxy):
    """Array of object"""
    def __init__(self, type):
        self.type = api.Type.fromType(type)

    def __call__(self):
        return api.ArrayType(self.type)

class Choice(api.TypeProxy):
    """A string with a choice among several alternative"""
    def __init__(self, *args):
        self.choices = args

    def __call__(self):
        return api.StringType


class Task(Type):
    """Register a task"""
    def __init__(self, typename=None, scriptpath=None, pythonpath=None, description=None, associate=None):
        super().__init__(typename, description)
        self.pythonpath = sys.executable if pythonpath is None else pythonpath
        self.scriptpath = scriptpath

    def __call__(self, objecttype):
        # Register the type
        objecttype = super().__call__(objecttype) 

        # Construct command  
        _type = objecttype.__xpm__.originaltype
        command = commandline.Command()
        command.add(commandline.CommandPath(self.pythonpath))
        command.add(commandline.CommandString("-m"))
        command.add(commandline.CommandString("experimaestro"))
        command.add(commandline.CommandString("run"))

        if _type.__module__ and _type.__module__ != "__main__":
            logger.debug("Task: using module %s [%s]", _type.__module__, _type)
            command.add(commandline.CommandString(_type.__module__))
        else:
            filepath = Path(inspect.getfile(_type)).absolute()
            logger.debug("Task: using file %s [%s]", filepath, _type)
            command.add(commandline.CommandString("--file"))
            command.add(commandline.CommandPath(filepath))

        command.add(commandline.CommandString(str(self.typename)))
        command.add(commandline.CommandParameters())
        commandLine = commandline.CommandLine()
        commandLine.add(command)

        objecttype.__xpm__.task = commandline.CommandLineTask(commandLine)
        return objecttype


# --- Argument related annotations

class Argument():
    """Defines an argument for an experimaestro type"""
    def __init__(self, name, type=None, default=None, required=None,
                 ignored=False, help=None):
        # Determine if required
        self.name = name                
        self.type = api.Type.fromType(type) if type else None
        self.help = help
        self.ignored = ignored
        self.default = default
        self.required = required
        self.generator = None

    def __call__(self, objecttype):
        if self.type is None:
            if default: 
                self.type = api.Type.fromType(type(default))
            else:
                raise ValueError("Type is not defined for argument %s", self.name)
        argument = api.Argument(self.name, self.type, help=self.help, required=self.required, ignored=self.ignored, generator=self.generator, default=self.default)
        objecttype.__xpm__.addArgument(argument)
        return objecttype

class PathArgument(Argument):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""
    def __init__(self, name, path, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path
        """
        super().__init__(name, type=Path, help=help)
        self.generator = lambda jobcontext: jobcontext.jobpath / path


class ConstantArgument(Argument):
    """
    An constant argument (useful for versionning tasks)
    """
    def __init__(self, name: str, value, xpmtype=None, help=""):
        super().__init__(name, type=xpmtype or api.Type.fromType(type(value)), help=help)
        self.generator = lambda jobcontext: api.clone(value)

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
