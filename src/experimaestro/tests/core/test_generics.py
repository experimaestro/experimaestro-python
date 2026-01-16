"""Tests for the use of generics in configurations"""

from typing import Generic, Optional, TypeVar

import pytest
from experimaestro import field, Config, Param
from experimaestro.core.arguments import Argument
from experimaestro.core.types import TypeVarType

# Mark all tests in this module as type system tests
pytestmark = pytest.mark.types

T = TypeVar("T")


class SimpleConfig(Config):
    pass


class SimpleConfigChild(SimpleConfig):
    pass


class SimpleGenericConfig(Config, Generic[T]):
    x: Param[T]


class SimpleGenericConfigChild(SimpleGenericConfig, Generic[T]):
    """A child class of SimpleGenericConfig that also uses generics"""

    pass


def test_core_generics_typevar():
    a = SimpleGenericConfig.C(x=1)

    x_arg = a.__xpmtype__.arguments["x"]

    # Check correct interpretation of typevar
    assert type(x_arg) is Argument
    assert isinstance(x_arg.type, TypeVarType)
    assert x_arg.type.typevar == T

    assert isinstance(a.x, int)


def test_core_generics_simple():
    a = SimpleGenericConfig.C(x=2)

    # OK
    a.x = 3

    # Fails: changing generics is not allowed
    with pytest.raises(TypeError):
        a.x = "a string"

    # typevar bindings are local to the instance,
    # so we can create a new instance with a different type
    SimpleGenericConfig.C(x="a string")


class DoubleGenericConfig(Config, Generic[T]):
    x: Param[T]
    y: Param[T]


def test_core_generics_double():
    # OK
    DoubleGenericConfig.C(x=1, y=1)

    # Fails
    with pytest.raises(TypeError):
        DoubleGenericConfig.C(x=1, y="a")

    a = DoubleGenericConfig.C(x=1, y=1)
    a.y = 2
    with pytest.raises(TypeError):
        a.x = "b"


def test_core_generics_double_rebind():
    a = DoubleGenericConfig.C(x=1, y=1)
    # Rebinding to a different type should not work
    with pytest.raises(TypeError):
        a.x, a.y = "some", "string"


def test_core_generics_double_plus():
    # Testing with inheritance
    # We allow subclasses of the typevar binding
    # We also allow generalizing up the typevar binding
    # This means that we can use a super class of the typevar binding

    # Works
    a = DoubleGenericConfig.C(x=SimpleConfigChild.C())
    a.y = SimpleConfig.C()

    # Works also
    b = DoubleGenericConfig.C(x=SimpleConfig.C())
    b.y = SimpleConfigChild.C()

    a.x = SimpleConfigChild.C()

    with pytest.raises(TypeError):
        a.x = "a string"


def test_core_generics_double_type_escalation():
    a = DoubleGenericConfig.C(x=SimpleConfigChild.C())
    a.y = SimpleConfigChild.C()
    # T is now bound to SimpleConfigChild

    a.y = SimpleConfig.C()
    # T is now bound to SimpleConfig

    a.y = object()
    # T is now bound to object, which is a super class of SimpleConfigChild

    # This is allowed, since we are not changing the typevar binding
    a.x = "a string"

    a.y = dict()
    # This is allowed, since we are not changing the typevar binding


def test_core_generics_double_deep_bind():
    # Since we are deep binding the typevar T to a specific type,
    # we should not be able to have coherent *local-only* type bindings
    # The type bindings are transient

    with pytest.raises(TypeError):
        DoubleGenericConfig.C(
            x=DoubleGenericConfig.C(x=1, y=2), y=DoubleGenericConfig.C(x=3, y=4)
        )


class NestedConfig(Config, Generic[T]):
    x: Param[DoubleGenericConfig[T]]
    y: Param[SimpleGenericConfig[T]]


def test_core_generics_nested():
    # OK
    NestedConfig.C(x=DoubleGenericConfig.C(x=1, y=1), y=SimpleGenericConfig.C(x=2))

    # Not OK
    with pytest.raises(TypeError):
        NestedConfig.C(
            x=DoubleGenericConfig.C(x=1, y=1), y=SimpleGenericConfig.C(x="b")
        )

    with pytest.raises(TypeError):
        a = NestedConfig.C(
            x=DoubleGenericConfig.C(x=1, y=1), y=SimpleGenericConfig.C(x=1)
        )
        a.x.x = "a string"


class TreeGenericConfig(Config, Generic[T]):
    x: Param[T]
    left: Optional["TreeGenericConfig[T]"] = None
    right: Optional["TreeGenericConfig[T]"] = None


class TagTreeGenericConfig(TreeGenericConfig[T], Generic[T]):
    """A tagged version of TreeGenericConfig to test recursive generics"""

    tag: Param[str] = field(ignore_default="default")


def test_core_generics_recursive():
    a = TreeGenericConfig.C(x=1)
    a.left = TreeGenericConfig.C(x=2)
    a.right = TreeGenericConfig.C(x=3)

    with pytest.raises(TypeError):
        a.left.x = "a string"

    # OK to use a child class
    a.left = TagTreeGenericConfig.C(x=4, tag="left")

    with pytest.raises(TypeError):
        a.left.x = "a string"


def test_core_generics_recursive_child():
    # Testing with a child class on the generic value
    a = TreeGenericConfig.C(x=SimpleConfig.C())
    a.left = TreeGenericConfig.C(x=SimpleConfig.C())
    a.right = TreeGenericConfig.C(x=SimpleConfig.C())

    a.left.x = SimpleConfigChild.C()

    with pytest.raises(TypeError):
        a.left.x = "a string"


U = TypeVar("U", bound=SimpleConfigChild)


class BoundGenericConfig(Config, Generic[U]):
    x: Param[U]


def test_core_generics_bound_typevar():
    a = BoundGenericConfig.C(x=SimpleConfigChild.C())
    assert isinstance(a.x, SimpleConfigChild)
    with pytest.raises(TypeError):
        a.x = SimpleConfig.C()


# =============================================================================
# Tests for Generic[T] where T is a Config type
# =============================================================================


class ConfigA(Config):
    """Base config for generic type parameter tests"""

    pass


class ConfigB(ConfigA):
    """Subclass of ConfigA for covariance tests"""

    pass


class GenericConfigHolder(Config, Generic[T]):
    """A generic config that holds another config of type T"""

    held: Param[T]


def test_core_generics_config_type_param():
    """Test basic generic with Config as type parameter"""
    a = GenericConfigHolder.C(held=ConfigA.C())
    assert isinstance(a.held, ConfigA)

    # Can assign same type
    a.held = ConfigA.C()

    # Can assign subtype (covariance)
    a.held = ConfigB.C()


def test_core_generics_config_type_param_subtype_binding():
    """Test that binding to a subtype allows supertype assignment"""
    a = GenericConfigHolder.C(held=ConfigB.C())

    # T is bound to ConfigB, but we allow generalization to ConfigA
    a.held = ConfigA.C()


def test_core_generics_config_type_param_type_mismatch():
    """Test that incompatible config types are rejected"""
    a = GenericConfigHolder.C(held=ConfigA.C())

    # Cannot assign unrelated type
    with pytest.raises(TypeError):
        a.held = "not a config"

    with pytest.raises(TypeError):
        a.held = 123


class WrapperConfig(Config, Generic[T]):
    """Wrapper that holds a GenericConfigHolder[T]"""

    inner: Param[GenericConfigHolder[T]]


def test_core_generics_nested_config_type_param():
    """Test nested generics with Config type parameter"""
    inner = GenericConfigHolder.C(held=ConfigA.C())
    wrapper = WrapperConfig.C(inner=inner)

    assert isinstance(wrapper.inner.held, ConfigA)

    # Nested assignment respects type binding
    wrapper.inner.held = ConfigB.C()


def test_core_generics_nested_config_type_param_mismatch():
    """Test that nested generics enforce type consistency"""
    # Create with ConfigA
    inner_a = GenericConfigHolder.C(held=ConfigA.C())
    wrapper = WrapperConfig.C(inner=inner_a)

    # Cannot assign string to nested held
    with pytest.raises(TypeError):
        wrapper.inner.held = "invalid"


class ParentWithGeneric(Config):
    """Parent class with a generic parameter"""

    holder: Param[GenericConfigHolder[ConfigA]]


class ChildWithGeneric(ParentWithGeneric):
    """Child class that overrides with more specific generic"""

    holder: Param[GenericConfigHolder[ConfigB]]


def test_core_generics_inheritance_override():
    """Test that child can override with more specific generic type"""
    # Child requires GenericConfigHolder[ConfigB]
    child = ChildWithGeneric.C(holder=GenericConfigHolder.C(held=ConfigB.C()))
    assert isinstance(child.holder.held, ConfigB)


def test_core_generics_override_subtype_config():
    """Test that overriding Generic[ConfigA] with Generic[ConfigB] is allowed
    when ConfigB is a subtype of ConfigA"""

    class Parent(Config):
        holder: Param[GenericConfigHolder[ConfigA]]

    # Should succeed - ConfigB is subtype of ConfigA
    class Child(Parent):
        holder: Param[GenericConfigHolder[ConfigB]] = field(overrides=True)

    Child.__getxpmtype__().arguments


def test_core_generics_override_incompatible_config():
    """Test that overriding Generic[ConfigA] with Generic[ConfigC] raises error
    when ConfigC is not a subtype of ConfigA"""

    class ConfigC(Config):
        """Unrelated config type"""

        pass

    class Parent(Config):
        holder: Param[GenericConfigHolder[ConfigA]]

    # Should fail - ConfigC is not a subtype of ConfigA
    with pytest.raises(TypeError, match="is not a subtype"):

        class Child(Parent):
            holder: Param[GenericConfigHolder[ConfigC]] = field(overrides=True)

        Child.__getxpmtype__().arguments


def test_core_generics_override_same_generic_type():
    """Test that overriding with the same generic type is allowed"""

    class Parent(Config):
        holder: Param[GenericConfigHolder[ConfigA]]

    # Should succeed - same type
    class Child(Parent):
        holder: Param[GenericConfigHolder[ConfigA]] = field(overrides=True)

    Child.__getxpmtype__().arguments


def test_core_generics_override_nested_subtype():
    """Test that nested generics can be overridden with subtypes"""

    class Parent(Config):
        wrapper: Param[WrapperConfig[ConfigA]]

    # Should succeed - ConfigB is subtype of ConfigA
    class Child(Parent):
        wrapper: Param[WrapperConfig[ConfigB]] = field(overrides=True)

    Child.__getxpmtype__().arguments
