"""Tests for type validation"""

import pytest
from pathlib import Path
from experimaestro import Task, field, Identifier, Constant, Param, Config, Meta
from enum import Enum
from experimaestro.generators import PathGenerator
from experimaestro.scheduler import Job, JobContext
from experimaestro.scheduler.workspace import RunMode
from .utils import TemporaryExperiment
from experimaestro.xpmutils import EmptyContext

valns = Identifier("validation")


def expect_validate(value):
    value.__xpm__.validate()


def expect_notvalidate(value):
    with pytest.raises((ValueError, KeyError)):
        value.__xpm__.validate()


class A(Config):
    value: Param[int]


class B(Config):
    a: Param[A]


class C(Config):
    path: Meta[Path] = field(default_factory=PathGenerator("outdir"))
    pass


def test_validation_simple():
    expect_validate(A(value=1))


def test_validation_missing():
    expect_notvalidate(A())


def test_validation_simple_nested():
    b = B()
    b.a = A(value=1)
    expect_validate(b)


def test_validation_missing_nested():
    b = B()
    b.a = A()
    expect_notvalidate(b)


def test_validation_type():
    class A(Config):
        __xpmid__ = valns.type.a
        pass

    class B(Config):
        __xpmid__ = valns.type.b

    class C(Config):
        a: Param[A]
        __xpmid__ = valns.type.c

    with pytest.raises(ValueError):
        C(a=B())

    with pytest.raises(ValueError):
        c = C()
        c.a = B()


def test_validation_subtype():
    class A(Config):
        __xpmid__ = valns.subtype.a

    class A1(A):
        __xpmid__ = valns.subtype.a1

    class B(Config):
        __xpmid__ = valns.subtype.b
        a: Param[A]

    expect_validate(B(a=A1()))


def test_validation_path_generator():
    """Test of path generator"""

    class A(Config):
        __xpmid__ = valns.path.a
        value: Meta[Path] = field(default_factory=PathGenerator("file.txt"))

    a = A()
    a.__xpm__.validate()
    with TemporaryExperiment("constant") as xp:
        jobcontext = Job(a)
        a.__xpm__.seal(JobContext(jobcontext))
        assert isinstance(a.value, Path)
        assert a.value.name == "file.txt"
        assert a.value.parents[0].name == a.__xpm__.identifier.all.hex()
        assert a.value.parents[1].name == str(a.__xpmtype__.identifier)
        assert a.value.parents[2].name == "jobs"
        assert a.value.parents[3] == xp.workspace.path


def test_validation_constant():
    """Test of constant"""

    class A(Config):
        __xpmid__ = valns.constant.a
        value: Constant[int] = 1

    a = A()
    a.__xpm__.validate()
    with TemporaryExperiment("constant"):
        joba = Job(a)
        a.__xpm__.seal(JobContext(joba))
        assert a.value == 1


class Parent(Config):
    x: Param[int]


class Child(Parent):
    pass


def test_validation_child():
    expect_validate(Child(x=1))


# --- Path argument checks


class PathParent(Config):
    x: Meta[Path] = field(default_factory=PathGenerator("x"))


def test_validation_path_option():
    c = PathParent()
    expect_validate(c)


# --- Default value


def test_validation_seal():
    """Test value sealing"""

    class A(Config):
        a: Param[int]

    a = A(a=2)
    a.__xpm__.seal(EmptyContext())

    with pytest.raises(AttributeError):
        a.a = 1


def test_validation_validation_enum():
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
    except AssertionError:
        pass


# --- Task as argument


class TaskParentConfig(Config):
    pass


class taskconfig(TaskParentConfig, Task):
    pass


class TaskConfigConsumer(Config):
    x: Param[TaskParentConfig]


def test_validation_taskargument():
    x = taskconfig()
    with TemporaryExperiment("fake"):
        x.submit(run_mode=RunMode.DRY_RUN)
        expect_validate(TaskConfigConsumer(x=x))
