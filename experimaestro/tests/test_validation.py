"""Tests for type validation"""

import pytest
from pathlib import Path
from experimaestro import (
    config,
    task,
    Identifier,
    argument,
    pathoption,
    ConstantParam,
    Param,
    Config,
)
from enum import Enum
import experimaestro.core.types as types
from experimaestro.scheduler import Job, JobContext
from .utils import TemporaryExperiment
from experimaestro.xpmutils import EmptyContext
import logging

valns = Identifier("validation")


def expect_validate(value):
    value.__xpm__.validate()


def expect_notvalidate(value):
    with pytest.raises((ValueError, KeyError)):
        value.__xpm__.validate()


@argument("value", type=int)
@config()
class A:
    pass


@argument("a", type=A)
@config()
class B:
    pass


@pathoption("path", "outdir")
@config()
class C:
    pass


def test_simple():
    expect_validate(A(value=1))


def test_missing():
    expect_notvalidate(A())


def test_simple_nested():
    b = B()
    b.a = A(value=1)
    expect_validate(b)


def test_missing_nested():
    b = B()
    b.a = A()
    expect_notvalidate(b)


def test_type():
    @config(valns.type.a)
    class A:
        pass

    @config(valns.type.b)
    class B:
        pass

    @argument("a", A)
    @config(valns.type.c)
    class C:
        pass

    with pytest.raises(ValueError):
        C(a=B())

    with pytest.raises(ValueError):
        c = C()
        c.a = B()


def test_subtype():
    @config(valns.subtype.a)
    class A:
        pass

    @config(valns.subtype.a1)
    class A1(A):
        pass

    @argument("a", A)
    @config(valns.subtype.b)
    class B:
        pass

    expect_validate(B(a=A1()))


def test_path():
    """Test of @pathoption"""

    @pathoption("value", "file.txt")
    @config(valns.path.a)
    class A:
        pass

    a = A()
    a.__xpm__.validate()
    with TemporaryExperiment("constant") as xp:
        jobcontext = Job(a)
        a.__xpm__.seal(jobcontext)
        assert isinstance(a.value, Path)
        parents = list(a.value.parents)
        assert a.value.name == "file.txt"
        assert a.value.parents[0].name == a.__xpm__.identifier.hex()
        assert a.value.parents[1].name == str(a.__xpmtype__.identifier)
        assert a.value.parents[2].name == "jobs"
        assert a.value.parents[3] == xp.workspace.path


def test_constant():
    """Test of @ConstantParam"""

    @ConstantParam("value", 1)
    @config(valns.constant.a)
    class A:
        pass

    a = A()
    a.__xpm__.validate()
    with TemporaryExperiment("constant") as ws:
        joba = Job(a)
        a.__xpm__.seal(JobContext(joba))
        assert a.value == 1


@argument("x", type=int)
@config()
class Parent:
    pass


@config()
class Child(Parent):
    pass


def test_child():
    expect_validate(Child(x=1))


# --- Path argument checks


@pathoption("x", "x")
@config()
class PathParent:
    pass


def test_path():
    c = PathParent()
    expect_validate(c)


# --- Default value


@pytest.mark.parametrize(
    "value,apitype",
    [(1.5, types.FloatType), (1, types.IntType), (False, types.BoolType)],
)
def test_default(value, apitype):
    @argument("default", default=value)
    @config(valns.default[str(type(value))])
    class Default:
        pass

    value = Default()
    expect_validate(value)
    assert Default.__xpmtype__.arguments["default"].type.__class__ == apitype


def test_seal():
    """Test value sealing"""

    @argument("a", int)
    @config()
    class A:
        pass

    a = A(a=2)
    a.__xpm__.seal(EmptyContext())

    with pytest.raises(AttributeError):
        a.a = 1


def test_validation_enum():
    """Path arguments should be ignored"""

    class EnumParam(Enum):
        FIRST = 0
        SECOND = 1

    class EnumConfig(Config):
        a: Param[EnumParam]

    expect_validate(EnumConfig(a=EnumParam.FIRST))

    try:
        EnumConfig(a=1)
        assert False, "Enum value should be rejected"
    except AssertionError as e:
        pass


# --- Task as argument


@config()
class TaskParentConfig:
    pass


@task()
class taskconfig(TaskParentConfig):
    pass


@argument("x", type=TaskParentConfig)
@config()
class TaskConfigConsumer:
    pass


def test_taskargument():
    x = taskconfig()
    with TemporaryExperiment("fake"):
        x.submit(dryrun=True)
        expect_validate(TaskConfigConsumer(x=x))
