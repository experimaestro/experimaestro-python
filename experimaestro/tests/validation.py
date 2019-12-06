"""Tests for type validation"""

import unittest
from experimaestro import Type, Argument, PathArgument

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
        c = C()
        c.__xpm__.validate()
        
