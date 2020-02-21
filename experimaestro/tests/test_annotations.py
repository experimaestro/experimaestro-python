# Annotation specific tests

from experimaestro import config

@config()
class A: pass

def test_noname():
    assert str(A.__xpm__.identifier) == "experimaestro.tests.test_annotations.a"


@config("annotations.b")
class B: pass

def test_fullname():
    assert str(B.__xpm__.identifier) == "annotations.b"