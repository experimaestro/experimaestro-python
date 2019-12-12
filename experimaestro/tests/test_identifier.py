# Tests for identifier computation

from pathlib import Path
import unittest
import logging
from experimaestro import Type, Argument

@Argument(name="a", type=int)
@Type()
class A: pass

@Argument(name="a", type=int)
@Type()
class B: pass

@Argument(name="a", type=int, default=1)
@Argument(name="b", type=int)
@Type()
class C: pass

@Argument("a", type=A)
@Type()
class D: pass

@Argument("value", type=float)
@Type()
class Float: pass


@Argument("value1", type=float)
@Argument("value2", type=float)
@Type()
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


# --- Ignore paths

@Argument("a", int)
@Argument("path", Path)
@Type()
class TypeWithPath: pass

def test_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath(a=1, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
    assert_notequal(TypeWithPath(a=2, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
