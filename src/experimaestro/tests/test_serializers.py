from typing import Optional
from experimaestro import (
    Config,
    Param,
    state_dict,
    from_state_dict,
)
from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import ConfigMixin


class Object1(Config):
    pass


class Object2(Config):
    object: Param[Object1]


def test_serializers_serialization():
    context = SerializationContext(save_directory=None)

    obj1 = Object1.C()
    obj2 = Object2.C(object=obj1)

    data = state_dict(context, [obj1, obj2])

    [obj1, obj2] = from_state_dict(data)
    assert isinstance(obj1, Object1) and isinstance(obj1, ConfigMixin)
    assert isinstance(obj2, Object2)
    assert obj2.object is obj1

    [obj1, obj2] = from_state_dict(data, as_instance=True)
    assert isinstance(obj1, Object1) and not isinstance(obj1, ConfigMixin)
    assert isinstance(obj2, Object2)
    assert obj2.object is obj1


class SubConfig(Config):
    pass


class MultiParamObject(Config):
    opt_a: Param[Optional[int]]

    x: Param[dict[str, Optional[SubConfig]]]


def test_serializers_types():
    context = SerializationContext(save_directory=None)

    config = MultiParamObject.C(x={"a": None})
    config.__xpm__.seal(context)
    state_dict(context, config)
