from typing import Optional
from experimaestro import Param, Config
from experimaestro.core.objects import TypeConfig
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

    assert not isinstance(b, TypeConfig)
    assert isinstance(b, B.__xpmtype__.objecttype)

    assert not isinstance(b.a, TypeConfig)
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


def test_instance_serialized():
    model = Model()
    model.add_pretasks(LoadModel(value=model))
    trainer = Evaluator(model=model)
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


def test_instance_external():
    class ExtValueConfig(Config):
        # FIXME: move elsewhere
        @classmethod
        def value(cls):
            def wrapped(value_class):
                # Check that we have not already changed the value class
                # TODO: here

                # Check that we are the value class is a subclass
                if not issubclass(value_class, cls):
                    raise RuntimeError(f"{value_class} is not a subclass of {cls}")

                # Check that the value class if a subclass of the parent
                # configurations value class
                # TODO: here

                # Set the attribute
                setattr(cls, "XPMValue", value_class)
                return value_class

            return wrapped

    @ExtValueConfig.value()
    class ValueConfig(ExtValueConfig):
        pass

    class ExtSubValueConfig(ExtValueConfig):
        pass

    @ExtSubValueConfig.value()
    class ExtSubValueConfig(ExtSubValueConfig, ValueConfig):
        pass

    result = ExtValueConfig().instance()

    assert result.__class__ is ValueConfig
