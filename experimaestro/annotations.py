# Import Python modules

import sys
import inspect
import logging
from pathlib import Path
from typing import Callable, Optional, TypeVar

import experimaestro.core.objects as objects
import experimaestro.core.types as types
from experimaestro.generators import PathGenerator
from typing_extensions import Annotated, get_type_hints
import typing_extensions

from .core.arguments import Argument as CoreArgument, TypeAnnotation
from .core.objects import Config
from .core.types import Identifier, TypeProxy, Type, ObjectType
from .utils import logger
from .checkers import Checker

# --- Annotations to define tasks and types


def configmethod(method):
    """(deprecated) Annotate a method that should be kept in the configuration object"""
    return method


class config:
    KEPT_METHODS = set(["config", "__validate__"])
    FORBIDDEN_ATTRIBUTES = set(["__module__", "__slots__", "__dict__"])

    """Annotations for experimaestro types"""

    def __init__(self, identifier=None, description=None, register=True):
        """[summary]

        Keyword Arguments:
            identifier {Identifier, str} -- Unique identifier of the type (default: {None})
            description {str} -- Description of the config/task (default: {None})
            register {bool} -- False if the type should not be registered (debug only)


        The identifier, if not specified, will be set to `X.CLASSNAME`(by order of priority),
        where X is:
            - the parent identifier
            - the module qualified name
        """
        super().__init__()
        self.identifier = identifier
        if isinstance(self.identifier, str):
            self.identifier = Identifier(self.identifier)

        self.description = description
        self.register = register

    def __call__(self, tp, basetype=Config):
        """Annotate the class

        Depending on whether we are running or configuring,
        the behavior is different:

        - when configuring, we return a proxy class
        - when running, we return the same class

        Arguments:
            tp {type} -- The type

        Keyword Arguments:
            basetype {type} -- The base type of the class

        Raises:
            ValueError: [description]

        Returns:
            [type] -- [description]
        """
        assert inspect.isclass(tp), f"{tp} is not a class"

        # --- If in task mode, returns now
        if Config.TASKMODE:
            return tp

        # --- Add Config as an ancestor of t if needed

        # if not in task mode,
        # manipulate the class path so that basetype is a parent
        __bases__ = tp.__bases__
        if not issubclass(tp, basetype):
            __bases__ = (basetype,)
            if tp.__bases__ != (object,):
                __bases__ += tp.__bases__

        # Remove all methods but those marked by @configmethod
        __dict__ = {key: value for key, value in tp.__dict__.items()}

        configtype = type(tp.__name__, __bases__, __dict__)
        configtype.__module__ = tp.__module__

        # Adds to xpminfo for on demand creation of information
        configtype.__xpm__ = ObjectType.create(
            configtype,
            identifier=self.identifier,
            register=self.register,
        )

        return configtype


class Array(TypeProxy):
    """Array of object"""

    def __init__(self, type):
        self.type = Type.fromType(type) if not Config.TASKMODE else None

    def __call__(self):
        return types.ArrayType(self.type)


class Choice(TypeProxy):
    """A string with a choice among several alternative"""

    def __init__(self, *args):
        self.choices = args

    def __call__(self):
        return types.StringType


class task(config):
    """Register a task"""

    def __init__(self, identifier=None, pythonpath=None, description=None):
        super().__init__(identifier, description)
        self.pythonpath = sys.executable if pythonpath is None else pythonpath

    def __call__(self, tp):
        # Register the type
        tp = super().__call__(tp, basetype=objects.Task)
        if Config.TASKMODE:
            return tp

        tp.__xpm__.addAnnotation(self)
        return tp

    def process(self, xpmtype):
        # Construct command
        import experimaestro.commandline as commandline

        command = commandline.Command()
        command.add(commandline.CommandPath(self.pythonpath))
        command.add(commandline.CommandString("-m"))
        command.add(commandline.CommandString("experimaestro"))
        command.add(commandline.CommandString("run"))
        command.add(commandline.CommandParameters())
        commandLine = commandline.CommandLine()
        commandLine.add(command)

        assert xpmtype.task is None
        xpmtype.task = commandline.CommandLineTask(commandLine)


# --- argument related annotations


class param:
    """Defines an argument for an experimaestro type"""

    def __init__(
        self,
        name,
        type=None,
        default=None,
        required: bool = None,
        ignored: Optional[bool] = None,
        help: Optional[str] = None,
        checker: Optional[Checker] = None,
    ):
        # Determine if required
        self.name = name
        self.type = Type.fromType(type) if type and not Config.TASKMODE else None
        self.help = help
        self.ignored = ignored
        self.default = default
        self.required = required
        self.generator = None
        self.checker = checker
        self.subparam = False

    def __call__(self, tp):
        # Don't annotate in task mode
        if Config.TASKMODE:
            return tp

        tp.__xpm__.addAnnotation(self)
        return tp

    def process(self, xpmtype):
        # Get type from default if needed
        if self.type is None:
            if self.default is not None:
                self.type = Type.fromType(type(self.default))

        # Type = any if no type
        if self.type is None:
            self.type = types.Any

        argument = CoreArgument(
            self.name,
            self.type,
            help=self.help,
            required=self.required,
            ignored=self.ignored,
            generator=self.generator,
            default=self.default,
            checker=self.checker,
            subparam=self.subparam,
        )
        xpmtype.addArgument(argument)


class subparam(param):
    """Defines an argument for an experimaestro type"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subparam = True


# Just a rebind (back-compatibility)
argument = param


class option(param):
    """An argument which is ignored

    See argument
    """

    def __init__(self, *args, **kwargs):
        kwargs["ignored"] = True
        super().__init__(*args, **kwargs)


class pathoption(param):
    """Defines a an argument that will be a relative path (automatically
    set by experimaestro)"""

    def __init__(self, name: str, path=None, help=""):
        """
        :param name: The name of argument (in python)
        :param path: The relative path or a function
        """
        super().__init__(name, type=Path, help=help)

        if path is None:
            path = name

        self.generator = PathGenerator(path)


STDERR = lambda jobcontext: "%s.err" % jobcontext.name
STDOUT = lambda jobcontext: "%s.out" % jobcontext.name


class ConstantParam(param):
    """
    An constant argument (useful for versionning tasks)
    """

    def __init__(self, name: str, value, xpmtype=None, help=""):
        super().__init__(name, type=xpmtype or Type.fromType(type(value)), help=help)
        self.generator = lambda jobcontext: objects.clone(value)


# --- Cache


def cache(name: str):
    """Use a cache path for a given config"""

    def annotate(method):

        return objects.cache(method, name)

    return annotate


# --- Tags


def tag(value):
    """Tag a value"""
    return objects.TaggedValue(value)


def tags(value):
    """Return the tags associated with a value"""
    return value.__xpm__.tags()


def tagspath(value: Config):
    """Return a unique path made of tags and their values"""
    sortedtags = sorted(value.__xpm__.tags().items(), key=lambda x: x[0])
    return "_".join(f"""{key.replace("/", "-")}={value}""" for key, value in sortedtags)


# --- Deprecated


def deprecateClass(klass):
    import inspect

    def __init__(self, *args, **kwargs):
        frameinfo = inspect.stack()[1]
        logger.warning(
            "Class %s is deprecated: use %s in %s:%s (%s)",
            klass.__name__,
            klass.__bases__[0].__name__,
            frameinfo.filename,
            frameinfo.lineno,
            frameinfo.code_context,
        )
        super(klass, self).__init__(*args, **kwargs)

    klass.__init__ = __init__
    return klass
