# Import Python modules

import sys
import inspect
import logging
from pathlib import Path
from typing import Callable, Optional

import experimaestro.core.objects as objects
import experimaestro.core.types as types
from experimaestro.generators import PathGenerator

from .core.arguments import TypeHint, Argument as CoreArgument, TypedArgument
from .core.objects import Config
from .core.types import Identifier, TypeProxy, Type, ObjectType
from .utils import logger
from .typingutils import get_optional
from .checkers import Checker

# --- Annotations to define tasks and types


def configmethod(method):
    """Annotate a method that should be kept in the configuration object"""
    method.__xpmconfig__ = True
    return method


class config:
    KEPT_METHODS = set(["config", "__validate__"])
    FORBIDDEN_ATTRIBUTES = set(["__module__", "__slots__", "__dict__"])

    """Annotations for experimaestro types"""

    def __init__(self, identifier=None, description=None, register=True, parents=[]):
        """[summary]

        Keyword Arguments:
            identifier {Identifier, str} -- Unique identifier of the type (default: {None})
            description {str} -- Description of the config/task (default: {None})
            parents {list} -- Parent classes if annotating a method (default: {[]})
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
        self.parents = parents
        self.register = register

    def __call__(self, tp, originaltype=None, basetype=Config):
        """Annotate the class

        Depending on whether we are running or configuring,
        the behavior is different:

        - when configuring, we return a proxy class
        - when running, we return the same class

        Arguments:
            tp {[type]} -- Can be a method or a class

        Keyword Arguments:
            originaltype {[type]} -- The original type
            basetype {[type]} -- [description] The base type of the class

        Raises:
            ValueError: [description]

        Returns:
            [type] -- [description]
        """

        # --- If in task mode, just return the object
        if Config.TASKMODE:
            return tp

        # The type to annotate
        originaltype = originaltype or tp

        # --- Add Config as an ancestor of t if needed
        if inspect.isclass(tp):
            # if not in task mode,
            # manipulate the class path so that basetype is a parent
            __bases__ = tp.__bases__
            if not issubclass(tp, basetype):
                __bases__ = (basetype,)
                if tp.__bases__ != (object,):
                    __bases__ += tp.__bases__

            keep = getattr(tp, "__xpm_default_keep__", False)

            # Remove all methods but those marked by @configmethod
            __dict__ = {
                key: value
                for key, value in tp.__dict__.items()
                # Not forbidden, and not function (unless we keep everything)
                if (
                    (keep or not inspect.isfunction(value))
                    and key not in config.FORBIDDEN_ATTRIBUTES
                )
                or (key in config.KEPT_METHODS)
                or hasattr(value, "__xpmconfig__")
            }

            tp = type(tp.__name__, __bases__, __dict__)
            tp.__module__ = originaltype.__module__

            # Modify original type __bases__ to get the real hierarchy
            # when creating instances
            if inspect.isclass(originaltype):
                originaltype.__bases__ = tuple(
                    t.__xpm__.originaltype
                    if issubclass(t, Config) and t is not Config
                    else t
                    for t in originaltype.__bases__
                )
        else:
            raise ValueError("Cannot use type %s as a type/task" % tp)

        # Determine the identifier
        if self.identifier is None:
            package = originaltype.__module__.lower()

            # Use the parent identifier
            # if basetype.__xpm__.

            self.identifier = Identifier(
                "%s.%s" % (package, originaltype.__name__.lower())
            )

        logging.debug("Registering %s", self.identifier)

        objecttype = ObjectType.create(
            tp, self.identifier, self.description, register=self.register
        )
        tp.__xpm__ = objecttype
        objecttype.originaltype = originaltype

        module = inspect.getmodule(originaltype)
        objecttype._file = Path(inspect.getfile(originaltype)).absolute()
        objecttype._module = module.__name__
        objecttype._package = module.__package__

        # Add description
        if objecttype.description is None:
            objecttype.description = originaltype.__doc__

        # Adding type-hinted arguments
        if hasattr(originaltype, "__annotations__"):
            for key, value in originaltype.__annotations__.items():
                if isinstance(value, TypedArgument):
                    valuetype = value.type
                    required = None

                    optionaltype = get_optional(valuetype)
                    if optionaltype:
                        valuetype = optionaltype
                        required = False

                    argument = CoreArgument(
                        key,
                        Type.fromType(valuetype),
                        default=getattr(originaltype, key, None),
                        required=required,
                        help=value.help,
                    )
                    objecttype.addArgument(argument)

        return tp


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

    def __init__(
        self, identifier=None, parents=None, pythonpath=None, description=None
    ):
        super().__init__(identifier, description)
        self.parents = parents or []
        if self.parents and not isinstance(self.parents, list):
            self.parents = [self.parents]

        self.pythonpath = sys.executable if pythonpath is None else pythonpath

    def __call__(self, tp):
        import experimaestro.commandline as commandline

        originaltype = tp
        if inspect.isfunction(tp):
            tp = objects.gettaskclass(tp, self.parents)
        else:
            assert not self.parents, "parents can only be used for functions"

        # Register the type
        tp = super().__call__(tp, originaltype=originaltype, basetype=objects.Task)
        if Config.TASKMODE:
            return tp

        # Construct command
        _type = tp.__xpm__.originaltype
        command = commandline.Command()
        command.add(commandline.CommandPath(self.pythonpath))
        command.add(commandline.CommandString("-m"))
        command.add(commandline.CommandString("experimaestro"))
        command.add(commandline.CommandString("run"))
        command.add(commandline.CommandParameters())
        commandLine = commandline.CommandLine()
        commandLine.add(command)

        tp.__xpm__.task = commandline.CommandLineTask(commandLine)
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
        tp.__xpm__.addArgument(argument)
        return tp


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


class ConstantParam(argument):
    """
    An constant argument (useful for versionning tasks)
    """

    def __init__(self, name: str, value, xpmtype=None, help=""):
        super().__init__(name, type=xpmtype or Type.fromType(type(value)), help=help)
        self.generator = lambda jobcontext: objects.clone(value)


Param = TypeHint()


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
