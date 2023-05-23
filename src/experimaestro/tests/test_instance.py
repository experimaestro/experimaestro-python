from typing import Optional
from experimaestro import config, Param, Config
from experimaestro.core.objects import TypeConfig
from experimaestro.core.serializers import SerializedConfig


@config()
class A:
    x: Param[int] = 1


@config()
class A1(A):
    pass


@config()
class B:
    a: Param[A]


def test_simple_instance():
    a = A1(x=1)
    b = B(a=a)
    b = b.instance()

    assert not isinstance(b, TypeConfig)
    assert isinstance(b, B.__xpmtype__.objecttype)

    assert not isinstance(b.a, TypeConfig)
    assert isinstance(b.a, A1.__xpmtype__.objecttype)
    assert isinstance(b.a, A.__xpmtype__.basetype)


class Model(Config):
    def __post_init__(self):
        self.initialized = False


class Trainer(Config):
    model: Param[Model]


class SerializedModel(SerializedConfig):
    def initialize(self):
        self.config.initialized = True


def test_instance_serialized():
    model = SerializedModel(config=Model())
    trainer = Trainer(model=model)
    instance = trainer.instance()

    assert isinstance(
        instance.model, Model
    ), f"The model is not a Model but a {type(instance.model).__qualname__}"
    assert instance.model.initialized, "The model was not initialized"


class ConfigWithOptional(Config):
    x: Param[int] = 1
    y: Param[Optional[int]]


def test_instance_optional():
    """Test that optional parameters are set to None when calling instance"""
    c = ConfigWithOptional().instance()
    assert c.x == 1
    assert c.y is None
