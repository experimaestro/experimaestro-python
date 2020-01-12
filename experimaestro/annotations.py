# Import Python modules

import json
import sys
import inspect
import os.path as op
import os
import logging
import pathlib
from pathlib import Path, PosixPath
from typing import Union, Dict, Optional

import experimaestro.api as api
from .api import Config, Identifier
from .utils import logger
from .workspace import Workspace

# --- Annotations to define tasks and types
        
class config:

    """Annotations for experimaestro types"""
    def __init__(self, identifier=None, description=None, register=True, parents=[]):
        """[summary]
        
        Keyword Arguments:
            identifier {Identifier, str} -- Unique identifier of the type (default: {None} will use the module/class name)
            description {str} -- Description of the config/task (default: {None})
            parents {list} -- Parent classes if annotating a method (default: {[]})
            register {bool} -- False if the type should not be registered (debug only)
        """
        super().__init__()
        self.identifier = identifier
        if isinstance(self.identifier, str):
            self.identifier = Identifier(self.identifier)
            
        self.description = description
        self.parents = parents
        self.register = register

    def __call__(self, tp, originaltype=None, basetype=api.Config):
        """[summary]
        
        Arguments:
            tp {[type]} -- Can be a method or a class
        
        Keyword Arguments:
            basetype {[type]} -- [description] The base type of the class
        
        Raises:
            ValueError: [description]
        
        Returns:
            [type] -- [description]
        """

        # Check if conditions are fullfilled
        originaltype = originaltype or tp
        if self.identifier is None:
            self.identifier = Identifier("%s.%s" % (originaltype.__module__.lower(), originaltype.__name__.lower()))

        # --- Add Config as an ancestor of t if needed
        if inspect.isclass(tp):
            if not issubclass(tp, api.Config):
                __bases__ = (basetype, )
                if tp.__bases__ != (object, ):
                    __bases__ += tp.__bases__
                __dict__ = {key: value for key, value in tp.__dict__.items() if key not in ["__dict__"]}
                tp = type(tp.__name__, __bases__, __dict__)
        else:
            raise ValueError("Cannot use type %s as a type/task" % tp)

        logging.debug("Registering %s", self.identifier)
        
        objecttype = api.ObjectType.create(tp, self.identifier, self.description, register=self.register)
        tp.__xpm__ = objecttype
        objecttype.originaltype = originaltype
        
        return tp


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


class task(config):
    """Register a task"""
    def __init__(self, identifier=None, parents=None, pythonpath=None, description=None):
        super().__init__(identifier, description)
        self.parents = parents or []
        if self.parents and not isinstance(self.parents, list):
            self.parents = [self.parents]

        self.pythonpath = sys.executable if pythonpath is None else pythonpath

    def __call__(self, tp):
        import experimaestro.commandline as commandline

        originaltype = tp
        if inspect.isfunction(tp):
            tp = api.gettaskclass(tp, self.parents)
        else:
            assert not self.parents, "parents can only be used for functions"

        # Register the type
        tp = super().__call__(tp, originaltype=originaltype, basetype=api.Task) 

        # Construct command  
        _type = tp.__xpm__.originaltype
        command = commandline.Command()
        command.add(commandline.CommandPath(self.pythonpath))
        command.add(commandline.CommandString("-m"))
        command.add(commandline.CommandString("experimaestro"))
        command.add(commandline.CommandString("run"))

        if _type.__module__ and _type.__module__ != "__main__":
            logger.debug("task: using module %s [%s]", _type.__module__, _type)
            command.add(commandline.CommandString(_type.__module__))
        else:
            filepath = Path(inspect.getfile(_type)).absolute()
            logger.debug("task: using file %s [%s]", filepath, _type)
            command.add(commandline.CommandString("--file"))
            command.add(commandline.CommandPath(filepath))

        command.add(commandline.CommandString(str(self.identifier)))
        command.add(commandline.CommandParameters())
        commandLine = commandline.CommandLine()
        commandLine.add(command)

        tp.__xpm__.task = commandline.CommandLineTask(commandLine)
        return tp


# --- argument related annotations

class argument():
    """Defines an argument for an experimaestro type"""
    def __init__(self, name, type=None, default=None, required:bool=None,
                 ignored:Optional[bool]=None, help:Optional[str]=None):
        # Determine if required
        self.name = name                
        self.type = api.Type.fromType(type) if type else None
        self.help = help
        self.ignored = ignored
        self.default = default
        self.required = required
        self.generator = None

    def __call__(self, tp):
        # Get type from default if needed
        if self.type is None:
            if self.default is not None: 
                self.type = api.Type.fromType(type(self.default))

        # Type = any if no type
        if self.type is None:
            self.type = api.Any

        argument = api.Argument(self.name, self.type, help=self.help, required=self.required, ignored=self.ignored, generator=self.generator, default=self.default)
        tp.__xpm__.addArgument(argument)
        return tp

class pathargument(argument):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""
    def __init__(self, name, path, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path
        """
        super().__init__(name, type=Path, help=help)
        self.generator = lambda jobcontext: jobcontext.jobpath / path


class ConstantArgument(argument):
    """
    An constant argument (useful for versionning tasks)
    """
    def __init__(self, name: str, value, xpmtype=None, help=""):
        super().__init__(name, type=xpmtype or api.Type.fromType(type(value)), help=help)
        self.generator = lambda jobcontext: api.clone(value)


# --- Tags


def tag(value):
    """Tag a value"""
    return api.TaggedValue(value)

def tags(value):
    """Return the tags associated with a value"""
    if isinstance(value, Value):
        return value.tags()
    return value.__xpm__.sv.tags()

def tagspath(value: api.Config):
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
