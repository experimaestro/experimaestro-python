from pathlib import Path
from experimaestro import Config, config, tag, Annotated
from experimaestro.core.arguments import Param
from experimaestro.core.objects import TypeConfig
from experimaestro.generators import pathgenerator
from experimaestro.xpmutils import DirectoryContext


@config()
class A:
    x: Param[int] = 3


def test_object_default():
    """Test plain default value"""
    a = A()
    assert a.x == 3


@config()
class B:
    a: Param[A] = A(x=3)


@config()
class C(B):
    pass


def test_object_config_default():
    """Test default configurations as default values"""
    b = B()
    assert b.a.x == 3

    c = C()
    assert c.a.x == 3


def test_hierarchy():
    """Test if the object hierarchy is OK"""
    OA = A.__xpmtype__.objecttype
    OB = B.__xpmtype__.objecttype
    OC = C.__xpmtype__.objecttype

    assert issubclass(A, Config)
    assert issubclass(B, Config)
    assert issubclass(C, Config)

    assert not issubclass(OA, TypeConfig)
    assert not issubclass(OB, TypeConfig)
    assert not issubclass(OC, TypeConfig)

    assert issubclass(C, B)

    assert OA.__bases__ == (Config,)
    assert OB.__bases__ == (Config,)
    assert OC.__bases__ == (B,)


# Test tags


def test_objects_tags():
    """Test tags"""

    class A(Config):
        x: Param[int]

    context = DirectoryContext(Path("/__fakepath__"))
    a = A(x=tag(1))
    a.__xpm__.seal(context)
    assert a.__xpm__.tags() == {"x": 1}


class TaskDirectoryContext(DirectoryContext):
    def __init__(self, task, path):
        super().__init__(path)
        self._task = task

    @property
    def task(self):
        return self._task


def test_objects_nested_tags():
    """Tags should be propagated to nested output configurations"""

    class B(Config):
        p: Annotated[Path, pathgenerator("p.txt")]

    class A(Config):
        x: Param[int]
        b: Param[B]

    a = A(x=tag(1), b=B())
    context = TaskDirectoryContext(a, Path("/__fakepath__"))
    a.__xpm__.seal(context)

    # Tags of main object
    assert a.__xpm__.tags() == {"x": 1}
    # should be propagated to output configurations
    assert a.b.__xpm__.tags() == {"x": 1}
