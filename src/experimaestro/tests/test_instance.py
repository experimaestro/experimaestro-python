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
    a = A1.C(x=1)
    b = B.C(a=a)
    b = b.instance()

    assert not isinstance(b, ConfigMixin)
    assert isinstance(b, B.__xpmtype__.value_type)

    assert not isinstance(b.a, ConfigMixin)
    assert isinstance(b.a, A1.__xpmtype__.value_type)
    assert isinstance(b.a, A.__xpmtype__.value_type)


# --- Test pre tasks


class Model(Config):
    def __post_init__(self):
        self.initialized = False


class Evaluator(Config):
    model: Param[Model]


class LoadModel(SerializationLWTask):
    def execute(self):
        self.value.initialized = True


def test_instance_serialized():
    model = Model.C()
    model.add_pretasks(LoadModel.C(value=model))
    trainer = Evaluator.C(model=model)
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
    c = ConfigWithOptional.C().instance()
    assert c.x == 1
    assert c.y is None


def test_instance_keep_config():
    evaluator = Evaluator.C(model=Model.C())
    instance = evaluator.instance(keep=True)

    assert instance.__config__ is evaluator
