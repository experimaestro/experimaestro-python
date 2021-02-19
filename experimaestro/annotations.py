# Import Python modules

import sys
import inspect
import logging
from pathlib import Path
from typing import Type as TypingType, Optional, TypeVar

import experimaestro.core.objects as objects
import experimaestro.core.types as types
from experimaestro.generators import PathGenerator

from .core.arguments import Argument as CoreArgument
from .core.objects import Config
from .core.types import Identifier, TypeProxy, Type, ObjectType
from .utils import logger
from .checkers import Checker

# --- Annotations to define tasks and types

T = TypeVar("T")


def configmethod(method):
    """(deprecated) Annotate a method that should be kept in the configuration object"""
    return method


class config:
    """Annotations for experimaestro types"""

    def __init__(self, identifier=None, description=None):
        """[summary]

        Keyword Arguments:
            identifier {Identifier, str} -- Unique identifier of the type, generate by default (None)
            description {str} -- (deprecated, use comments) Description of the config/task, and use comments with (default) None
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

    def __call__(self, tp: T) -> T:
        """Annotate the class

        Depending on whether we are running or configuring,
        the behavior is different:

        - when configuring, we return a proxy class
        - when running, we return the same class

        Arguments:
            tp {type} -- The type

        Keyword Arguments:
            basetype {type} -- The base type of the class (Task or Config)
        """
        assert inspect.isclass(tp), f"{tp} is not a class"

        # Adds to xpminfo for on demand creation of information
        return ObjectType(tp, identifier=self.identifier).basetype


class Array(TypeProxy):
    """Array of object"""

    def __init__(self, type):
        self.type = Type.fromType(type)

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
        self.pythonpath = pythonpath

    def __call__(self, tp) -> type:
        # Register the type
        tp = super().__call__(tp)

        def factory(xpmtype):
            return xpmtype.getpythontaskcommand(self.pythonpath)

        tp.__getxpmtype__().taskcommandfactory = factory
        return tp


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
        constant: bool = False,
    ):
        # Determine if required
        self.name = name
        self.type = Type.fromType(type) if type else None
        self.help = help
        self.ignored = ignored
        self.default = default
        self.required = required
        self.generator = None
        self.checker = checker
        self.subparam = False
        self.constant = constant

    def __call__(self, tp):
        # Don't annotate in task mode
        tp.__getxpmtype__().addAnnotation(self)
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
            constant=self.constant,
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


class constant(param):
    """
    An constant argument (useful for versionning tasks)
    """

    def __init__(self, name: str, value, type=None, help=""):
        super().__init__(name, default=value, constant=True, type=type, help=help)


ConstantParam = constant

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
