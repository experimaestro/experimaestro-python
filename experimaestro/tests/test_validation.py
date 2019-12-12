"""Tests for type validation"""

import pytest
from pathlib import Path
from experimaestro import Type, Typename, Argument, PathArgument, ConstantArgument
import experimaestro.api as api
from experimaestro.scheduler import Job
from .utils import TemporaryExperiment
import logging

valns = Typename("validation")


def expect_validate(value):
    value.__xpm__.validate()

def expect_notvalidate(value):
    with pytest.raises(ValueError):
        value.__xpm__.validate()



@Argument("value", type=int)
@Type()
class A: pass

@Argument("a", type=A)
@Type()
class B: pass

@PathArgument("path", "outdir")
@Type()
class C: pass


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
    @Type(valns.type.a)
    class A(): pass

    @Type(valns.type.b)
    class B: pass

    @Argument("a", A)
    @Type(valns.type.c)
    class C: pass

    with pytest.raises(ValueError):
        C(a=B())

    with pytest.raises(ValueError):
        c = C()
        c.a = B()

def test_subtype():
    @Type(valns.subtype.a)
    class A(): pass

    @Type(valns.subtype.a1)
    class A1(A): pass

    @Argument("a", A)
    @Type(valns.subtype.b)
    class B: pass

    expect_validate(B(a=A1()))


def test_path():
    """Test of @PathArgument"""
    @PathArgument("value", "file.txt")
    @Type(valns.path.a)
    class A: pass

    a = A()
    a.__xpm__.validate()
    with TemporaryExperiment("constant") as ws:
        jobcontext = Job(a)
        a.__xpm__.seal(jobcontext)
        assert isinstance(a.value, Path)
        parents = list(a.value.parents)
        assert a.value.name == "file.txt"
        assert a.value.parents[0].name == a.__xpm__.identifier.hex()
        assert a.value.parents[1].name == str(a.__class__.__xpm__.typename)
        assert a.value.parents[2].name == "jobs"
        assert a.value.parents[3] == ws.path

def test_constant():
    """Test of @ConstantArgument"""
    @ConstantArgument("value", 1)
    @Type(valns.constant.a)
    class A: pass

    a = A()
    a.__xpm__.validate()
    with TemporaryExperiment("constant") as ws:
        jobcontext = Job(a)
        a.__xpm__.seal(jobcontext)
        assert a.value == 1


@Argument("a", int)
@Type()
def notset(a, b): pass


def test_notset():
    expect_notvalidate(notset(a=1))

@Argument("a", int)
@Type()
def notdeclared(): pass

def test_notdeclared():
    expect_notvalidate(notdeclared(a=1))

# --- Default value


@pytest.mark.parametrize('value,apitype', [(1.5, api.FloatType), (1, api.IntType), (False, api.BoolType)])
def test_default(value, apitype):
    @Argument("default", default=value)
    @Type(valns.default[str(type(value))])
    class Default: pass

    value = Default()
    expect_validate(value)
    assert Default.__xpm__.arguments["default"].type.__class__ == apitype
