"""test_annotations.py

Test annotation handling for configurations and tasks
"""

# Annotation specific tests

import sys
from pathlib import Path
from typing import Dict, Optional, List
from experimaestro.core.context import SerializationContext
from experimaestro.core.types import DictType, IntType, StrType
from enum import Enum
import pytest
from experimaestro import (
    Option,
    Constant,
    Param,
    Task,
    default,
    Meta,
    Config,
    pathgenerator,
    PathGenerator,
    field,
    Annotated,
)
import experimaestro.core.types as types
from experimaestro.xpmutils import DirectoryContext

# --- Test manual name for configuration


class B(Config):
    __xpmid__ = "annotations.b"
    pass


def test_fullname():
    assert str(B.__getxpmtype__().identifier) == "annotations.b"


# --- Automatic name for configuration


class A(Config):
    pass


def test_noname():
    assert str(A.__getxpmtype__().identifier) == "experimaestro.tests.test_param.a"


def serializeCycle(config: Config):
    """Serialize and deserialize a configuration"""
    from experimaestro.core.objects import ConfigInformation
    import experimaestro.taskglobals as taskglobals

    taskglobals.Env.instance().wspath = Path("/tmp-xpm1234")

    objects = config.__xpm__.__get_objects__([], SerializationContext())

    return ConfigInformation.fromParameters(objects)


def ArgumentValue(default=None, *, help=""):
    return default


def test_type_hinting():
    """Test for type hinting"""

    class MyConfig(Config):
        """A configuration

        Attributes:
            w: An integer
        """

        __xpmid__ = "annotations.class_variable.config"

        x: Param[int]
        y: Param[float] = 2.3
        y2: Annotated[float, default(2.3)]
        z: Param[Optional[float]]
        t: Param[List[float]]
        w: Param[int]
        opt: Param[Optional[int]]
        path: Annotated[Path, pathgenerator("world")]
        option: Option[str]

    ot = MyConfig.__getxpmtype__()

    # Check required parameter
    arg_x = ot.getArgument("x")
    assert arg_x.name == "x"
    assert isinstance(arg_x.type, types.IntType)
    assert arg_x.required

    arg_y = ot.getArgument("y")
    assert arg_y.name == "y"
    assert isinstance(arg_y.type, types.FloatType)
    assert arg_y.default == 2.3
    assert not arg_y.required
    arg_y2 = ot.getArgument("y2")
    assert arg_y2.name == "y2"
    assert isinstance(arg_y2.type, types.FloatType)
    assert arg_y2.default == 2.3
    assert not arg_y2.required

    arg_z = ot.getArgument("z")
    assert arg_z.name == "z"
    assert isinstance(arg_y.type, types.FloatType)
    assert not arg_z.required

    arg_t = ot.getArgument("t")
    assert arg_t.name == "t"
    assert isinstance(arg_t.type, types.ArrayType)
    assert isinstance(arg_t.type.type, types.FloatType)
    assert arg_t.required

    arg_w = ot.getArgument("w")
    assert arg_w.help == "An integer"

    arg_opt = ot.getArgument("opt")
    assert not arg_opt.required

    arg_path = ot.getArgument("path")
    assert arg_path.generator is not None
    assert arg_path.generator(DirectoryContext(Path("hello")), None) == Path(
        "hello/world"
    )

    arg_option = ot.getArgument("option")
    assert arg_option.name == "option"
    assert isinstance(arg_option.type, types.StrType)
    assert arg_option.ignored


def test_generatedpath():
    class A(Config):
        path: Meta[Path] = field(default_factory=PathGenerator("test.txt"))

    class B(Config):
        a: Param[A]

    class C(Task):
        b: Param[B]

    basepath = Path("/tmp/testconflict")
    c = C(b=B(a=A())).instance(DirectoryContext(basepath))
    assert c.b.a.path.relative_to(basepath) == Path("out/b/a/test.txt")


def test_config_class():
    """Test configuration declared as a class"""

    class A(Config):
        x: Param[int]

    a = A(x=1)
    assert a.x == 1

    class B(A):
        y: Param[int]

    b = B(x=1, y=2)
    assert b.x == 1
    assert b.y == 2

    class D(Config):
        x: Param[int]

    class C(Config):
        d: Param[D]

    c = C(d=D(x=1))
    assert c.d.x == 1


def test_constant():
    class A(Config):
        x: Constant[int] = 2

    a = A()
    assert a.x == 2, "Constant value not set"

    # We should not be able to change the value
    with pytest.raises(AttributeError):
        a.x = 3


class EnumParam(Enum):
    NONE = 0
    OTHER = 1


class EnumConfig(Config):
    x: Param[EnumParam]


def test_param_enum():
    """Test for enum values"""

    a = EnumConfig(x=EnumParam.OTHER)
    _a = serializeCycle(a)

    assert isinstance(_a, EnumConfig)
    assert _a.x == EnumParam.OTHER


def test_inheritance():
    class A(Config):
        x: Param[int]

    class B(A):
        y: Param[int] = 3

    b = B()
    b.x = 2
    assert b.__xpm__.values["y"] == 3
    assert b.__xpm__.values["x"] == 2


def test_redefined_param():
    class A(Config):
        x: Param[int]

    class B(Config):
        x: Param[int] = 3

    atx = A.C.__getxpmtype__().getArgument("x")
    btx = B.C.__getxpmtype__().getArgument("x")

    assert atx.required

    assert not btx.required


def test_param_dict():
    class A(Config):
        x: Param[Dict[str, int]]

    xpmA = A.__getxpmtype__()
    xarg = xpmA.getArgument("x").type
    assert isinstance(xarg, DictType)
    assert isinstance(xarg.keytype, StrType)
    assert isinstance(xarg.valuetype, IntType)

    A(x={"OK": 1})

    with pytest.raises(TypeError):
        A(x={"wrong": "string"})
    with pytest.raises(TypeError):
        A(x={"wrong": 1.2})


# --- Default


class ConfigWithDefault(Config):
    x: Param[int] = field(default=1)


def test_param_default():
    assert ConfigWithDefault().x == 1


class ConfigWithDefaultFactory(Config):
    x: Param[int] = field(default_factory=lambda: 1)


def test_param_default_factory():
    value = ConfigWithDefaultFactory()
    context = DirectoryContext(Path("/__fakepath__"))
    value.__xpm__.seal(context)
    assert value.x == 1


# --- Task annotations


def test_default_mismatch():
    """Test mismatch between default and type"""

    class A(Config):
        x: Param[int] = 0.2

    with pytest.raises(TypeError):
        A.__getxpmtype__().getArgument("x")


# --- Handling help annotations


def test_help():
    class A(Config):
        """Description of A

        Long description of A.

        Arguments:

            y: Parameter y
        """

        x: Param[int]
        """Parameter x"""

        y: Param[int]

    xpmtype = A.__getxpmtype__()
    xpmtype.__initialize__()

    assert xpmtype.title == "Description of A"
    assert xpmtype.description.strip() == "Long description of A."
    assert xpmtype.arguments["y"].help == "Parameter y"

    # Only python >= 3.9
    if sys.version_info.major == 3 and sys.version_info.minor > 8:
        assert xpmtype.arguments["x"].help == "Parameter x"
