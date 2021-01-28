from experimaestro import config, Param, Config
from experimaestro.core.objects import TypeConfig


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


@config()
class SerializedConfig:
    x: Param[int] = 1
    pass


class TestSerialization:
    """Test that a config can be serialized during execution"""

    def test_instance(self):
        import pickle

        a = SerializedConfig(x=2).instance()
        assert not isinstance(a, TypeConfig)
        assert isinstance(a, SerializedConfig)

        s_a = pickle.dumps(a)

        deserialized = pickle.loads(s_a)
        assert not isinstance(deserialized, TypeConfig)
        assert deserialized.x == 2
