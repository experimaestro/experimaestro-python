"""Tests for the use of generics in configurations"""

from typing import Generic, TypeVar

import pytest
from experimaestro import Config, Param
from experimaestro.core.arguments import Argument
from experimaestro.core.types import TypeVarType

T = TypeVar("T")


class SimpleConfig(Config):
    pass


class SimpleConfigChild(Config):
    pass


class SimpleGenericConfig(Config, Generic[T]):
    x: Param[T]


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
        a.x = "arggg"


class DoubleGenericConfig(Config, Generic[T]):
    x: Param[T]
    y: Param[T]


def test_core_generics_double():
    # OK
    DoubleGenericConfig.C(x=1, y=1)

    # Fails
    with pytest.raises(TypeError):
        DoubleGenericConfig.C(x=1, y="a")


def test_core_generics_double_plus():
    # Works
    a = SimpleGenericConfig.C(x=SimpleConfigChild.C())
    a.y = SimpleConfig.C()

    # Works also
    b = SimpleGenericConfig.C(x=SimpleConfig.C())
    b.y = SimpleConfigChild.C()


class NestedConfig(Config, Generic[T]):
    x: DoubleGenericConfig[T]
    y: SimpleGenericConfig[T]


def test_core_generics_nested():
    # OK
    NestedConfig.C(x=DoubleGenericConfig.C(x=1, y=1), y=SimpleGenericConfig.C(x=2))

    # Not OK
    with pytest.raises(TypeError):
        NestedConfig.C(
            x=DoubleGenericConfig.C(x=1, y=1), y=SimpleGenericConfig.C(x="b")
        )


def test_core_generics_nested_more():
    nc = NestedConfig.C(y=SimpleConfig.C())
    nc.x = NestedConfig.C(y=SimpleConfig.C(), x=SimpleConfigChild.C())

    with pytest.raises(TypeError):
        nc = NestedConfig.C(y=2)
        nc.x = NestedConfig.C(y=SimpleConfig.C(), x=SimpleConfigChild.C())
