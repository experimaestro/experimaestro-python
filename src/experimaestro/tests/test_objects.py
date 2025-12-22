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


# --- Composition operator tests (GH #33) ---


class CompositionA(Config):
    x: Param[int]


class CompositionSubA(CompositionA):
    """Subclass of CompositionA"""

    y: Param[int] = 0


class CompositionB(Config):
    a: Param[CompositionA]


class CompositionC(Config):
    """Config with two parameters of same type - should be ambiguous"""

    a1: Param[CompositionA]
    a2: Param[CompositionA]


class CompositionD(Config):
    """Config with no matching parameter"""

    x: Param[int]


class CompositionE(Config):
    """Config with two parameters, one subclass of the other"""

    base: Param[CompositionA]
    sub: Param[CompositionSubA]


def test_composition_operator():
    """Test that B() @ A(x=1) is equivalent to B(a=A(x=1))"""
    a = CompositionA.C(x=42)
    b = CompositionB.C() @ a

    assert b.a is a
    assert b.a.x == 42


def test_composition_operator_chained():
    """Test chaining composition operators

    Chaining A @ B @ C adds both B and C to A (same outer config).
    For nested structures, use parentheses: A @ (B @ C)
    """

    class MultiParam(Config):
        a: Param[CompositionA]
        b: Param[CompositionB]

    # Chaining adds multiple configs to same outer config
    result = MultiParam.C() @ CompositionA.C(x=10) @ CompositionB.C()

    assert result.a.x == 10
    assert result.b is not None


def test_composition_operator_nested():
    """Test nested composition with parentheses"""

    class Outer(Config):
        b: Param[CompositionB]

    # For nested structures, use parentheses
    result = Outer.C() @ (CompositionB.C() @ CompositionA.C(x=10))

    assert result.b.a.x == 10


def test_composition_operator_ambiguous():
    """Test that ambiguous composition raises ValueError"""
    a = CompositionA.C(x=1)

    with pytest.raises(ValueError, match="Ambiguous"):
        CompositionC.C() @ a


def test_composition_operator_no_match():
    """Test that composition with no matching param raises ValueError"""
    a = CompositionA.C(x=1)

    with pytest.raises(ValueError, match="No parameter"):
        CompositionD.C() @ a


def test_composition_operator_subclass():
    """Test composition works with subclasses"""
    sub_a = CompositionSubA.C(x=5, y=10)
    b = CompositionB.C() @ sub_a

    assert b.a is sub_a
    assert b.a.x == 5


def test_composition_operator_subclass_hierarchy():
    """Test composition when two params have subclass relationship

    When CompositionSubA is passed, both 'base' (CompositionA) and 'sub'
    (CompositionSubA) match. This should be ambiguous since both accept it.
    """
    sub_a = CompositionSubA.C(x=1, y=2)

    # SubA matches both base (CompositionA) and sub (CompositionSubA)
    with pytest.raises(ValueError, match="Ambiguous"):
        CompositionE.C() @ sub_a


def test_composition_operator_exact_match():
    """Test composition when base class instance matches only base param"""
    # CompositionA matches only 'base', not 'sub' (which requires SubA)
    a = CompositionA.C(x=1)
    e = CompositionE.C() @ a

    assert e.base is a
    assert e.base.x == 1
