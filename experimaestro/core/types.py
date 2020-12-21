import inspect
from typing import Union, Dict, Iterator, List, Type as TypingType
from collections import ChainMap
from pathlib import Path
import experimaestro.typingutils as typingutils
from experimaestro.utils import logger
from .objects import Config, Task, BaseTaskFunction
from .arguments import Argument


class Identifier:
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


class Type:
    """Any experimaestro type is a child class"""

    DEFINED: Dict[type, "Type"] = {}

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

        t = typingutils.get_list(key)
        if t:
            return ArrayType(Type.fromType(t))

        raise Exception("No type found for %s", key)


class ObjectType(Type):
    """The type of a Config or Task"""

    REGISTERED: Dict[str, TypingType["Config"]] = {}

    def __init__(self, objecttype: TypingType["Config"], identifier, description):
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
            if issubclass(tp, Config) and tp not in [Config, Task, BaseTaskFunction]:
                yield tp.__xpm__

    @staticmethod
    def create(
        configclass: TypingType["Config"], identifier, description, register=True
    ):
        if register and str(identifier) in ObjectType.REGISTERED:
            _objecttype = ObjectType.REGISTERED[str(identifier)]
            if _objecttype.__xpm__.originaltype != configclass:
                # raise Exception("Experimaestro type %s is already declared" % identifier)
                pass

            logger.error("Experimaestro type %s is already declared" % identifier)
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
                try:
                    return classtype(**value)
                except:
                    logger.exception("Could not build object of class %s" % (classtype))
                    raise
            if not Config.TASKMODE:
                raise ValueError("Could not find type %s", valuetype)

            logger.debug("Using argument type (not real type)")
            return self.objecttype(**value)

        if not isinstance(value, Config):
            raise ValueError(f"{value} is not an experimaestro type or task")

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

    def name(self):
        return f"Array[{self.type.name()}]"

    def validate(self, value):
        if not isinstance(value, List):
            raise ValueError("value is not a list")

        return [self.type.validate(x) for x in value]
