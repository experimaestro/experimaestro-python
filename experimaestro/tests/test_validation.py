"""Tests for type validation"""

import unittest
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
    def test(self):
        a = method(self)
        a.__xpm__.validate()
    return test

def expect_notvalidate(method):
    def test(self):
        try:
            a = method(self)
            a.__xpm__.validate()
            self.fail("Value validated, but should not have")
        except ValueError:
            pass
    return test




class MainTest(unittest.TestCase):
    @expect_validate
    def test_simple(self):
        return A(value=1)

    @expect_notvalidate
    def test_missing(self):
        return A()

    @expect_validate
    def test_simple_nested(self):
        b = B()
        b.a = A(value=1)
        return b

    @expect_notvalidate
    def test_missing_nested(self):
        b = B()
        b.a = A()
        return b

    def test_path(self):
        """Test of @PathArgument"""
        @PathArgument("value", "file.txt")
        @Type(valns.path.a)
        class A: pass

        a = A()
        a.__xpm__.validate()
        with TemporaryExperiment("constant") as ws:
            jobcontext = Job(a)
            a.__xpm__.seal(jobcontext)
            self.assertTrue(isinstance(a.value, Path))
            parents = list(a.value.parents)
            self.assertEqual(a.value.name, "file.txt")
            self.assertEqual(a.value.parents[0].name, a.__xpm__.identifier.hex())
            self.assertEqual(a.value.parents[1].name, str(a.__class__.__xpm__.typename))
            self.assertEqual(a.value.parents[2].name, "jobs")
            self.assertEqual(a.value.parents[3], ws.path)

    def test_constant(self):
        """Test of @ConstantArgument"""
        @ConstantArgument("value", 1)
        @Type(valns.constant.a)
        class A: pass

        a = A()
        a.__xpm__.validate()
        with TemporaryExperiment("constant") as ws:
            jobcontext = Job(a)
            a.__xpm__.seal(jobcontext)
            self.assertEqual(a.value, 1)

        
