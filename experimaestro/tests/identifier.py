# Tests for identifier computation

import unittest
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

def expect_equal(method):
    def test(self):
        a, b = method(self)
        self.assertEqual(a.__xpm__.identifier, b.__xpm__.identifier)
    return test
def expect_notequal(method):
    def test(self):
        a, b = method(self)
        self.assertNotEqual(a.__xpm__.identifier, b.__xpm__.identifier)
    return test

class MainTest(unittest.TestCase):
    @expect_equal
    def test_int(self):
        return A(a=1), A(a=1)

    @expect_notequal
    def test_different_type(self):
        return A(a=1), B(a=1)

    @expect_equal
    def test_order(self):
        return Values(value1=1, value2=2), Values(value2=2, value1=1)

    @expect_equal
    def test_default(self):
        return C(a=1, b=2), C(b=2)

    @expect_equal
    def test_inner_eq(self):
        return D(a=A(a=1)), D(a=A(a=1))

    @expect_equal
    def test_float(self):
        return Float(value=1), Float(value=1)

    @expect_equal
    def test_float2(self):
        return Float(value=1.), Float(value=1)
