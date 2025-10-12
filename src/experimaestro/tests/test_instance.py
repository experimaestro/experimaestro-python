from typing import Optional
from experimaestro import Param, Config
from experimaestro.core.objects import ConfigMixin
from experimaestro.core.serializers import SerializationLWTask


class A(Config):
    x: Param[int] = 1


class A1(A):
    pass


class B(Config):
    a: Param[A]


def test_simple_instance():
    a = A1(x=1)
    b = B(a=a)
    b = b.instance()

    assert not isinstance(b, ConfigMixin)
    assert isinstance(b, B.__xpmtype__.objecttype)

    assert not isinstance(b.a, ConfigMixin)
    assert isinstance(b.a, A1.__xpmtype__.objecttype)
    assert isinstance(b.a, A.__xpmtype__.basetype)


# --- Test pre tasks


class Model(Config):
    def __post_init__(self):
        self.initialized = False


class Evaluator(Config):
    model: Param[Model]


class LoadModel(SerializationLWTask):
    def execute(self):
        self.value.initialized = True


class ConfigWithOptional(Config):
    x: Param[int] = 1
    y: Param[Optional[int]]


def test_instance_optional():
    """Test that optional parameters are set to None when calling instance"""
    c = ConfigWithOptional().instance()
    assert c.x == 1
    assert c.y is None
