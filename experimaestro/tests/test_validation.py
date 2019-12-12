"""Tests for type validation"""

import pytest
from pathlib import Path
from experimaestro import Type, Typename, Argument, PathArgument, ConstantArgument
from experimaestro.scheduler import Job
from .utils import TemporaryExperiment

valns = Typename("validation")

@Argument("value", type=int)
@Type()
class A: pass

@Argument("a", type=A)
@Type()
class B: pass

@PathArgument("path", "outdir")
@Type()
class C: pass


def expect_validate(method):
    def test():
        a = method()
        a.__xpm__.validate()
    return test

def expect_notvalidate(method):
    def test():
        with pytest.raises(ValueError):
            a = method()
            a.__xpm__.validate()
            assert False, "Value validated, but should not have"
    return test




@expect_validate
def test_simple():
    return A(value=1)

@expect_notvalidate
def test_missing():
    return A()

@expect_validate
def test_simple_nested():
    b = B()
    b.a = A(value=1)
    return b

@expect_notvalidate
def test_missing_nested():
    b = B()
    b.a = A()
    return b
    
@expect_notvalidate
def test_type():
    @Type(valns.type.a)
    class A(): pass

    @Type(valns.type.b)
    class B: pass

    @Argument("a", A)
    @Type(valns.type.c)
    class C: pass

    return C(a=B())


@expect_validate
def test_subtype():
    @Type(valns.subtype.a)
    class A(): pass

    @Type(valns.subtype.a1)
    class A1(A): pass

    @Argument("a", A)
    @Type(valns.subtype.b)
    class B: pass

    return B(a=A1())


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

@expect_notvalidate
def test_notset():
    return notset(a=1)

@Argument("a", int)
@Type()
def notdeclared(): pass

@expect_notvalidate
def test_notdeclared():
    return notdeclared(a=1)