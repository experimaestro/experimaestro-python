from experimaestro import tag, config, argument

@argument("x", type=int)
@config()
class Config1: pass

def test_tag():
    c = Config1(x=5)
    c.tag("x", 5)
    assert c.tags() == {"x": 5}


def test_taggedvalue():
    c = Config1(x=tag(5))
    assert c.tags() == {"x": 5}
