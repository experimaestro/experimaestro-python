# Tests for identifier computation

from pathlib import Path
import unittest
import logging
from experimaestro import config, Param, param, task, option
from experimaestro.core.arguments import Option, pathgenerator
from typing_extensions import Annotated


@config()
class A:
    a: Param[int]
    pass


@config()
class B:
    a: Param[int]
    pass


@config()
class C:
    a: Param[int] = 1
    b: Param[int]


@config()
class D:
    a: Param[A]


@config()
class Float:
    value: Param[float]


@config()
class Values:
    value1: Param[float]
    value2: Param[float]


def assert_equal(a, b):
    assert a.__xpm__.identifier.all == b.__xpm__.identifier.all


def assert_notequal(a, b):
    assert a.__xpm__.identifier.all != b.__xpm__.identifier.all


def test_int():
    assert_equal(A._(a=1), A._(a=1))


def test_different_type():
    assert_notequal(A._(a=1), B._(a=1))


def test_order():
    assert_equal(Values._(value1=1, value2=2), Values._(value2=2, value1=1))


def test_default():
    assert_equal(C._(a=1, b=2), C._(b=2))


def test_inner_eq():
    assert_equal(D._(a=A._(a=1)), D._(a=A._(a=1)))


def test_float():
    assert_equal(Float._(value=1), Float._(value=1))


def test_float2():
    assert_equal(Float._(value=1.0), Float._(value=1))


# --- Argument name


def test_name():
    """The identifier fully determines the hash code"""

    @config("test.identifier.argumentname")
    class Config0:
        a: Param[int]

    @config("test.identifier.argumentname")
    class Config1:
        b: Param[int]

    @config("test.identifier.argumentname")
    class Config3:
        a: Param[int]

    assert_notequal(Config0._(a=2), Config1._(b=2))
    assert_equal(Config0._(a=2), Config3._(a=2))


# --- Test option


def test_option():
    @config("test.identifier.option")
    class OptionConfig:
        a: Param[int]
        b: Option[int] = 1

    assert_notequal(OptionConfig._(a=2), OptionConfig._(a=1))
    assert_equal(OptionConfig._(a=1, b=2), OptionConfig._(a=1))
    assert_equal(OptionConfig._(a=1, b=2), OptionConfig._(a=1, b=2))


# --- Ignore paths


@config()
class TypeWithPath:
    a: Param[int]
    path: Param[Path]


def test_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath._(a=1, path="/a/b"), TypeWithPath._(a=1, path="/c/d"))
    assert_notequal(TypeWithPath._(a=2, path="/a/b"), TypeWithPath._(a=1, path="/c/d"))


# --- Test with added arguments


def test_pathoption():
    """Path arguments should be ignored"""

    @config("pathoption_test")
    class A_with_path:
        a: Param[int]
        path: Annotated[Path, pathgenerator("path")]

    @config("pathoption_test")
    class A_without_path:
        a: Param[int]

    assert_equal(A_with_path._(a=1), A_without_path._(a=1))


def test_defaultnew():
    """Path arguments should be ignored"""

    @param("b", type=int, default=1)
    @param(name="a", type=int)
    @config("defaultnew")
    class A_with_b:
        pass

    @param(name="a", type=int)
    @config("defaultnew")
    class A:
        pass

    assert_equal(A_with_b._(a=1, b=1), A._(a=1))
    assert_equal(A_with_b._(a=1), A._(a=1))


def test_taskconfigidentifier():
    """Test whether the embedded task arguments make the configuration different"""

    @param("a", type=int)
    @config()
    class Config:
        pass

    @param("x", type=int)
    @task()
    class Task:
        def config(self) -> Config:
            return Config._(a=1)

    assert_equal(Task._(x=1).submit(dryrun=True), Task._(x=1).submit(dryrun=True))
    assert_notequal(Task._(x=2).submit(dryrun=True), Task._(x=1).submit(dryrun=True))
