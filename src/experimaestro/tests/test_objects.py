from pathlib import Path

import pytest
from experimaestro import Config, Task, Annotated, copyconfig, default
from experimaestro.core.arguments import Param
from experimaestro.core.objects import TypeConfig
from experimaestro.core.types import XPMValue
from experimaestro.generators import pathgenerator
from experimaestro.scheduler.workspace import RunMode
from experimaestro.tests.utils import TemporaryExperiment


@pytest.fixture()
def xp():
    with TemporaryExperiment("deprecated", maxwait=0, run_mode=RunMode.DRY_RUN) as xp:
        yield xp


class A(Config):
    x: Param[int] = 3


def test_object_default():
    """Test plain default value"""
    a = A()
    assert a.x == 3


class B(Config):
    a: Param[A] = A(x=3)


class C(B):
    pass


class D(B, A):
    pass


class DefaultAnnotationConfig(Config):
    a: Annotated[A, default(A(x=3))]


def test_object_config_default():
    """Test default configurations as default values"""
    b = B()
    assert b.a.x == 3

    c = C()
    assert c.a.x == 3

    annotationConfig = DefaultAnnotationConfig()
    assert annotationConfig.a.x == 3


def test_hierarchy():
    """Test if the object hierarchy is OK"""
    OA = A.__getxpmtype__().objecttype
    OB = B.__getxpmtype__().objecttype
    OC = C.__getxpmtype__().objecttype
    OD = D.__getxpmtype__().objecttype

    assert issubclass(A, Config)
    assert issubclass(B, Config)
    assert issubclass(C, Config)

    assert not issubclass(OA, TypeConfig)
    assert not issubclass(OB, TypeConfig)
    assert not issubclass(OC, TypeConfig)

    assert issubclass(C, B)

    assert OA.__bases__ == (A, XPMValue)
    assert OB.__bases__ == (B, XPMValue)
    assert OC.__bases__ == (C, B.XPMValue)
    assert OD.__bases__ == (D, B.XPMValue, A.XPMValue)


class CopyConfig(Task):
    path: Annotated[Path, pathgenerator("hello.txt")]
    x: Param[int]


def test_copyconfig(xp):
    b = CopyConfig(x=2)

    b.submit()

    copy_b = copyconfig(b)

    assert copy_b.x == b.x
    assert "path" not in copy_b.__xpm__.values
