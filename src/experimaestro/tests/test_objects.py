from pathlib import Path

import pytest
from experimaestro import Config, Task, Annotated, copyconfig, field
from experimaestro.core.arguments import Param
from experimaestro.core.objects import ConfigMixin
from experimaestro.generators import pathgenerator
from experimaestro.scheduler.workspace import RunMode
from experimaestro.tests.utils import TemporaryExperiment


@pytest.fixture()
def xp():
    with TemporaryExperiment("deprecated", run_mode=RunMode.DRY_RUN) as xp:
        yield xp


class A(Config):
    x: Param[int] = field(ignore_default=3)


def test_object_default():
    """Test plain default value"""
    a = A.C()
    assert a.x == 3


class B(Config):
    a: Param[A] = field(ignore_default=A.C(x=3))


class C(B):
    pass


class D(B, A):
    pass


class DefaultAnnotationConfig(Config):
    a: Param[A] = field(default=A.C(x=3))


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

    y: Param[int] = field(ignore_default=0)


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


# --- Value class decorator tests (GH #99) ---

# Test 1: Basic value class registration


class ValueBasicModel(Config):
    x: Param[int] = field(ignore_default=1)


@ValueBasicModel.value_class()
class ValueBasicModelImpl(ValueBasicModel):
    def compute(self):
        return self.x * 2


# Test 2: Subclass without explicit value class


class ValueInheritBase(Config):
    x: Param[int] = field(ignore_default=1)


@ValueInheritBase.value_class()
class ValueInheritBaseImpl(ValueInheritBase):
    pass


class ValueInheritSubNoExplicit(ValueInheritBase):
    """Subclass without explicit value class"""

    y: Param[int] = field(ignore_default=2)


# Test 3: Value class with proper inheritance


class ValueInheritParent(Config):
    x: Param[int] = field(ignore_default=1)


@ValueInheritParent.value_class()
class ValueInheritParentImpl(ValueInheritParent):
    def compute(self):
        return self.x * 2


class ValueInheritChild(ValueInheritParent):
    y: Param[int] = field(ignore_default=2)


@ValueInheritChild.value_class()
class ValueInheritChildImpl(ValueInheritChild, ValueInheritParentImpl):
    def compute_both(self):
        return self.x + self.y


# Test 4: Skip intermediate class (A -> B -> C, only A and C have value classes)


class ValueSkipBase(Config):
    x: Param[int] = field(ignore_default=1)


@ValueSkipBase.value_class()
class ValueSkipBaseImpl(ValueSkipBase):
    def compute(self):
        return self.x * 2


class ValueSkipIntermediate(ValueSkipBase):
    """Intermediate class without explicit value class"""

    y: Param[int] = field(ignore_default=2)


class ValueSkipDeep(ValueSkipIntermediate):
    """Deep subclass with value class"""

    z: Param[int] = field(ignore_default=3)


@ValueSkipDeep.value_class()
class ValueSkipDeepImpl(ValueSkipDeep, ValueSkipBaseImpl):
    def compute_all(self):
        return self.x + self.y + self.z


# --- Value class tests ---


def test_value_decorator_basic():
    """Test basic value class registration"""
    # XPMValue should return the registered value class
    assert ValueBasicModel.XPMValue is ValueBasicModelImpl

    # Creating an instance should use the value class
    config = ValueBasicModel.C(x=5)
    instance = config.instance()

    assert isinstance(instance, ValueBasicModelImpl)
    assert instance.x == 5
    assert instance.compute() == 10


def test_value_decorator_inheritance_no_explicit():
    """Test that subclass without value class uses config class as value"""
    # SubModel has no explicit value class, XPMValue returns the config class
    assert ValueInheritSubNoExplicit.XPMValue is ValueInheritSubNoExplicit

    config = ValueInheritSubNoExplicit.C(x=3, y=4)
    instance = config.instance()

    # Instance is created from the config class (no explicit value type)
    assert isinstance(instance, ValueInheritSubNoExplicit)
    assert instance.x == 3
    assert instance.y == 4


def test_value_decorator_inheritance_with_explicit():
    """Test value class with proper inheritance from parent value class"""
    assert ValueInheritChild.XPMValue is ValueInheritChildImpl

    config = ValueInheritChild.C(x=3, y=4)
    instance = config.instance()

    assert isinstance(instance, ValueInheritChildImpl)
    assert isinstance(instance, ValueInheritParentImpl)
    assert instance.x == 3
    assert instance.y == 4
    assert instance.compute() == 6  # From parent value class
    assert instance.compute_both() == 7  # From this value class


def test_value_decorator_must_be_subclass():
    """Test that value class must be subclass of config"""

    class LocalModel(Config):
        x: Param[int]

    class OtherConfig(Config):
        z: Param[int]

    with pytest.raises(TypeError, match="must be a subclass of"):

        @LocalModel.value_class()
        class InvalidValue(OtherConfig):  # Not a subclass of LocalModel
            pass


def test_value_decorator_must_inherit_parent_value():
    """Test that value class must inherit from parent value class"""

    class LocalBase(Config):
        x: Param[int] = field(ignore_default=1)

    @LocalBase.value_class()
    class LocalBaseImpl(LocalBase):
        pass

    class LocalChild(LocalBase):
        y: Param[int] = field(ignore_default=2)

    with pytest.raises(TypeError, match="must be a subclass of.*parent value class"):

        @LocalChild.value_class()
        class InvalidChildValue(LocalChild):  # Missing LocalBaseImpl inheritance
            pass


def test_value_decorator_skip_intermediate():
    """Test value class when intermediate class has no value class"""
    # ValueSkipBase has impl, ValueSkipIntermediate has none, ValueSkipDeep has impl
    assert ValueSkipDeep.XPMValue is ValueSkipDeepImpl

    config = ValueSkipDeep.C(x=1, y=2, z=3)
    instance = config.instance()

    assert isinstance(instance, ValueSkipDeepImpl)
    assert isinstance(instance, ValueSkipBaseImpl)
    assert instance.compute() == 2  # From ValueSkipBaseImpl
    assert instance.compute_all() == 6  # From ValueSkipDeepImpl
