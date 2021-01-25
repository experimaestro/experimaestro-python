from experimaestro import config
from experimaestro.core.arguments import Param


@config()
class A:
    x: Param[int] = 3


def test_object_default():
    """Test plain default value"""
    a = A()
    assert a.x == 3


@config()
class B:
    a: Param[A] = A(x=3)


@config()
class C(B):
    pass


def test_object_config_default():
    """Test default configurations as default values"""
    b = B()
    assert b.a.x == 3

    c = C()
    assert c.a.x == 3
