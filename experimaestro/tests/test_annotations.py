"""test_annotations.py

Test all the annotations for configurations and tasks
"""

# Annotation specific tests

from pathlib import Path
from typing import Optional, List
from experimaestro.core.arguments import Option, pathgenerator
import pytest
from experimaestro import config, Config, Param, task
import experimaestro.core.types as types
from experimaestro.xpmutils import DirectoryContext
from typing_extensions import Annotated

# --- Test manual name for configuration


@config("annotations.b")
class B:
    pass


def test_fullname():
    assert str(B.__xpmtype__.identifier) == "annotations.b"


# --- Automatic name for configuration


@config()
class A:
    pass


def test_noname():
    assert str(A.__xpmtype__.identifier) == "experimaestro.tests.test_annotations.a"


# --- Use type annotations


def ArgumentValue(default=None, *, help=""):
    return default


def test_type_hinting():
    """Test for type hinting"""

    @config()
    class MyConfig:
        """A configuration

        Attributes:
            w: An integer
        """

        __xpmid__ = "annotations.class_variable.config"

        x: Param[int]
        y: Param[float] = 2.3
        z: Param[Optional[float]]
        t: Param[List[float]]
        w: Param[int]
        opt: Param[Optional[int]]
        path: Annotated[Path, pathgenerator("world")]
        option: Option[str]

    ot = MyConfig.__xpmtype__

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
    assert arg_path.generator(DirectoryContext(Path("hello"))) == Path("hello/world")

    arg_option = ot.getArgument("option")
    assert arg_option.name == "option"
    assert isinstance(arg_option.type, types.StrType)
    assert arg_option.ignored


def test_inheritance():
    @config()
    class A:
        x: Param[int]

    @config()
    class B(A):
        y: Param[int] = 3

    b = B()
    b.x = 2
    assert b.__xpm__.values["y"] == 3
    assert b.__xpm__.values["x"] == 2


def test_redefined_param():
    @config()
    class A:
        x: Param[int]

    @config()
    class B:
        x: Param[int] = 3

    atx = A._.__xpmtype__.getArgument("x")
    btx = B._.__xpmtype__.getArgument("x")

    assert atx.required

    assert not btx.required


# --- Task annotations


def test_task_config():
    @config()
    class Output:
        pass

    @task()
    class Task:
        def config(self) -> Output:
            return {}

    output = Task().submit(dryrun=True)
    assert type(output) == Output.__xpmtype__.configtype


def test_default_mismatch():
    """Test mismatch between default and type"""

    @config()
    class A:
        x: Param[int] = 0.2

    with pytest.raises(TypeError):
        A.__xpmtype__.getArgument("x")
