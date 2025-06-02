"""Tests for the use of generics in configurations"""

from typing import Generic, Optional, TypeVar

import pytest
from experimaestro import Config, Param
from experimaestro.core.arguments import Argument
from experimaestro.core.types import TypeVarType

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

    tag: Param[str] = "default"


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
