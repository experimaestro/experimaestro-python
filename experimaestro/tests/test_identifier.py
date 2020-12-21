# Tests for identifier computation

from pathlib import Path
import unittest
import logging
from experimaestro import config, pathoption, param, task, option


@param(name="a", type=int)
@config()
class A:
    pass


@param(name="a", type=int)
@config()
class B:
    pass


@param(name="a", type=int, default=1)
@param(name="b", type=int)
@config()
class C:
    pass


@param("a", type=A)
@config()
class D:
    pass


@param("value", type=float)
@config()
class Float:
    pass


@param("value1", type=float)
@param("value2", type=float)
@config()
class Values:
    pass


def assert_equal(a, b):
    assert a.__xpm__.identifier.all == b.__xpm__.identifier.all


def assert_notequal(a, b):
    assert a.__xpm__.identifier.all != b.__xpm__.identifier.all


def test_int():
    assert_equal(A(a=1), A(a=1))


def test_different_type():
    assert_notequal(A(a=1), B(a=1))


def test_order():
    assert_equal(Values(value1=1, value2=2), Values(value2=2, value1=1))


def test_default():
    assert_equal(C(a=1, b=2), C(b=2))


def test_inner_eq():
    assert_equal(D(a=A(a=1)), D(a=A(a=1)))


def test_float():
    assert_equal(Float(value=1), Float(value=1))


def test_float2():
    assert_equal(Float(value=1.0), Float(value=1))


# --- Argument name


def test_name():
    """Path should be ignored"""

    @param("a", int)
    @config("test.identifier.argumentname", register=False)
    class Config0:
        pass

    @param("b", int)
    @config("test.identifier.argumentname", register=False)
    class Config1:
        pass

    @param("a", int)
    @config("test.identifier.argumentname", register=False)
    class Config3:
        pass

    assert_notequal(Config0(a=2), Config1(b=2))
    assert_equal(Config0(a=2), Config3(a=2))


# --- Test option


def test_option():
    @param("a", int)
    @option("b", type=int, default=1)
    @config("test.identifier.option", register=False)
    class OptionConfig:
        pass

    assert_notequal(OptionConfig(a=2), OptionConfig(a=1))
    assert_equal(OptionConfig(a=1, b=2), OptionConfig(a=1))
    assert_equal(OptionConfig(a=1, b=2), OptionConfig(a=1, b=2))


# --- Ignore paths


@param("a", int)
@param("path", Path)
@config()
class TypeWithPath:
    pass


def test_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath(a=1, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
    assert_notequal(TypeWithPath(a=2, path="/a/b"), TypeWithPath(a=1, path="/c/d"))


# --- Test with added arguments


def test_pathoption():
    """Path arguments should be ignored"""

    @pathoption("path", "path")
    @param(name="a", type=int)
    @config("pathoption_test", register=False)
    class A_with_path:
        pass

    @param(name="a", type=int)
    @config("pathoption_test", register=False)
    class A_without_path:
        pass

    assert_equal(A_with_path(a=1), A_without_path(a=1))


def test_defaultnew():
    """Path arguments should be ignored"""

    @param("b", type=int, default=1)
    @param(name="a", type=int)
    @config("defaultnew", register=False)
    class A_with_b:
        pass

    @param(name="a", type=int)
    @config("defaultnew", register=False)
    class A:
        pass

    assert_equal(A_with_b(a=1, b=1), A(a=1))
    assert_equal(A_with_b(a=1), A(a=1))


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
            return Config(a=1)

    assert_equal(Task(x=1).submit(dryrun=True), Task(x=1).submit(dryrun=True))
    assert_notequal(Task(x=2).submit(dryrun=True), Task(x=1).submit(dryrun=True))
