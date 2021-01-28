from experimaestro import tag, config, argument


@argument("x", type=int)
@config()
class Config1:
    pass


@argument("x", type=int)
@argument("c", type=Config1)
@config()
class Config2:
    pass


def test_tag():
    c = Config1(x=5)
    c.tag("x", 5)
    assert c.tags() == {"x": 5}


def test_taggedvalue():
    c = Config1(x=tag(5))
    assert c.tags() == {"x": 5}


def test_tagcontain():
    """Test that tags are not propagated to the upper configurations"""
    c1 = Config1(x=5)
    c2 = Config2(c=c1, x=tag(3)).tag("out", 1)
    assert c1.tags() == {}
    assert c2.tags() == {"x": 3, "out": 1}
