from pathlib import Path

import pytest
from experimaestro import Config, Task, Annotated, copyconfig, default
from experimaestro.core.arguments import Param
from experimaestro.core.objects import ConfigMixin
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
    a: Param[A] = A.C(x=3)


class C(B):
    pass


class D(B, A):
    pass


class DefaultAnnotationConfig(Config):
    a: Annotated[A, default(A.C(x=3))]


def test_object_config_default():
    """Test default configurations as default values"""
    b = B.C()
    assert b.a.x == 3

    c = C.C()
    assert c.a.x == 3

    annotationConfig = DefaultAnnotationConfig.C()
    assert annotationConfig.a.x == 3


def test_hierarchy():
    """Test if the object hierarchy is OK"""
    OA = A.__getxpmtype__().value_type
    OB = B.__getxpmtype__().value_type
    OC = C.__getxpmtype__().value_type

    assert issubclass(A, Config)
    assert issubclass(B, Config)
    assert issubclass(C, Config)

    assert not issubclass(OA, ConfigMixin)
    assert not issubclass(OB, ConfigMixin)
    assert not issubclass(OC, ConfigMixin)

    assert issubclass(C, B)


class CopyConfig(Task):
    path: Annotated[Path, pathgenerator("hello.txt")]
    x: Param[int]


def test_copyconfig(xp):
    b = CopyConfig.C(x=2)

    b.submit()

    copy_b = copyconfig(b)

    assert copy_b.x == b.x
    assert "path" not in copy_b.__xpm__.values
