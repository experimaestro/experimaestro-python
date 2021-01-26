from experimaestro import config, Param, Config


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
    assert not isinstance(b, Config)
    assert isinstance(b, B.__xpmtype__.objecttype)

    assert not isinstance(b.a, Config)
    assert isinstance(b.a, A.__xpmtype__.objecttype)
    assert isinstance(b.a, A1.__xpmtype__.objecttype)
