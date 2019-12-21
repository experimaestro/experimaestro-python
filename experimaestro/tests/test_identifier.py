# Tests for identifier computation

from pathlib import Path
import unittest
import logging
from experimaestro import config, argument

@argument(name="a", type=int)
@config()
class A: pass

@argument(name="a", type=int)
@config()
class B: pass

@argument(name="a", type=int, default=1)
@argument(name="b", type=int)
@config()
class C: pass

@argument("a", type=A)
@config()
class D: pass

@argument("value", type=float)
@config()
class Float: pass


@argument("value1", type=float)
@argument("value2", type=float)
@config()
class Values: pass

def assert_equal(a, b):
    assert a.__xpm__.identifier == b.__xpm__.identifier

def assert_notequal(a, b):
    assert a.__xpm__.identifier != b.__xpm__.identifier


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
    assert_equal(Float(value=1.), Float(value=1))

# --- Argument name

def test_name():
    """Path should be ignored"""
    @argument("a", int)
    @config("test.identifier.argumentname", register=False)
    class Config0: pass

    @argument("b", int)
    @config("test.identifier.argumentname", register=False)
    class Config1: pass

    @argument("a", int)
    @config("test.identifier.argumentname", register=False)
    class Config3: pass


    assert_notequal(Config0(a=2), Config1(b=2))
    assert_equal(Config0(a=2), Config3(a=2))


# --- Ignore paths

@argument("a", int)
@argument("path", Path)
@config()
class TypeWithPath: pass

def test_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath(a=1, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
    assert_notequal(TypeWithPath(a=2, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
