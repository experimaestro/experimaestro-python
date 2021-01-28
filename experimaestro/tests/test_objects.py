from experimaestro import config
from experimaestro.core.arguments import Param
from experimaestro.core.objects import Config, TypeConfig


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
    OA = A.__xpmtype__.objecttype
    OB = B.__xpmtype__.objecttype
    OC = C.__xpmtype__.objecttype

    assert issubclass(A, Config)
    assert issubclass(B, Config)
    assert issubclass(C, Config)

    assert not issubclass(OA, TypeConfig)
    assert not issubclass(OB, TypeConfig)
    assert not issubclass(OC, TypeConfig)

    assert issubclass(C, B)

    assert OA.__bases__ == (Config,)
    assert OB.__bases__ == (Config,)
    assert OC.__bases__ == (B,)
