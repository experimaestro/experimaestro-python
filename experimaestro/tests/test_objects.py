from experimaestro import config
from experimaestro.core.arguments import Param
from experimaestro.core.objects import Config


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


def test_hierarchy():
    """Test if the object hierarchy is OK"""
    A()
    B()
    C()

    OA = A.xpmtype().objecttype
    OB = B.xpmtype().objecttype
    OC = C.xpmtype().objecttype

    assert issubclass(A, Config)
    assert issubclass(B, Config)
    assert issubclass(C, Config)

    assert not issubclass(OA, Config)
    assert not issubclass(OB, Config)
    assert not issubclass(OC, Config)

    assert issubclass(C, B)

    assert OA.__bases__ == (object,)
    assert OB.__bases__ == (object,)
    assert OC.__bases__ == (OB,)

    assert OA.__mro__ == (OA, object)
    assert OB.__mro__ == (OB, object)
    assert OC.__mro__ == (OC, OB, object)
