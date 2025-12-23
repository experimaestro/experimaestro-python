# --- Task and types definitions

import logging
from experimaestro import Config, Param, field
from typing import Union

import pytest
from experimaestro.core.objects import ConfigMixin


def test_multiple_inheritance():
    class A(Config):
        pass

    class B(Config):
        pass

    class B1(B):
        pass

    class C1(B1, A):
        pass

    class C2(A, B1):
        pass

    for C in (C1, C2):
        logging.info("Testing %s", C)
        ctype = C.__getxpmtype__()
        assert issubclass(C, A)
        assert issubclass(C, B)
        assert issubclass(C, B1)

        assert ctype.value_type == C.__getxpmtype__().value_type

        assert issubclass(C.__getxpmtype__().value_type, B1.__getxpmtype__().value_type)
        assert issubclass(C.__getxpmtype__().value_type, B.__getxpmtype__().value_type)
        assert issubclass(C.__getxpmtype__().value_type, A.__getxpmtype__().value_type)
        assert not issubclass(C.__getxpmtype__().value_type, ConfigMixin)


def test_missing_hierarchy():
    class A(Config):
        pass

    class A1(A):
        pass

    class B(A1):
        pass

    B.__getxpmtype__()

    assert issubclass(B, A)
    assert issubclass(B, A1)


def test_types_union():
    class A(Config):
        x: Param[Union[int, str]]

    A.C(x=1)
    A.C(x="hello")
    with pytest.raises(ValueError):
        A.C(x=[])


def test_override_warning_without_flag(caplog):
    """Test that overriding a parameter without overrides=True produces a warning"""

    class Parent(Config):
        value: Param[int]

    with caplog.at_level(logging.WARNING, logger="xpm"):
        # Child overrides value without overrides=True
        class Child(Parent):
            value: Param[int]

        # Force initialization to trigger the warning
        Child.__getxpmtype__().arguments

    assert "overrides parent parameter" in caplog.text
    assert "Child" in caplog.text
    assert "value" in caplog.text


def test_override_no_warning_with_flag(caplog):
    """Test that overriding with overrides=True suppresses the warning"""

    class Parent(Config):
        value: Param[int]

    with caplog.at_level(logging.WARNING, logger="xpm"):
        # Child overrides value with overrides=True
        class Child(Parent):
            value: Param[int] = field(overrides=True)

        # Force initialization
        Child.__getxpmtype__().arguments

    # No warning should be issued
    assert "overrides parent parameter" not in caplog.text


def test_override_type_check_subtype_config():
    """Test that overriding Config type with subtype is allowed"""

    class BaseValue(Config):
        x: Param[int]

    class DerivedValue(BaseValue):
        y: Param[int]

    class Parent(Config):
        value: Param[BaseValue]

    # Should succeed - DerivedValue is subtype of BaseValue
    class Child(Parent):
        value: Param[DerivedValue] = field(overrides=True)

    Child.__getxpmtype__().arguments


def test_override_type_check_incompatible_config():
    """Test that overriding Config type with incompatible type raises error"""

    class ValueA(Config):
        x: Param[int]

    class ValueB(Config):
        y: Param[int]

    class Parent(Config):
        value: Param[ValueA]

    # Should fail - ValueB is not a subtype of ValueA
    with pytest.raises(TypeError, match="is not a subtype"):

        class Child(Parent):
            value: Param[ValueB] = field(overrides=True)

        Child.__getxpmtype__().arguments


def test_override_type_check_primitive_incompatible():
    """Test that overriding primitive type with incompatible type raises error"""

    class Parent(Config):
        value: Param[int]

    # Should fail - str is not a subtype of int
    with pytest.raises(TypeError, match="is not compatible"):

        class Child(Parent):
            value: Param[str] = field(overrides=True)

        Child.__getxpmtype__().arguments


def test_override_type_check_same_type():
    """Test that overriding with the same type is allowed"""

    class Parent(Config):
        value: Param[int]

    # Should succeed - same type
    class Child(Parent):
        value: Param[int] = field(overrides=True)

    Child.__getxpmtype__().arguments


def test_no_override_warning_for_new_param(caplog):
    """Test that defining a new parameter doesn't produce a warning"""

    class Parent(Config):
        x: Param[int]

    with caplog.at_level(logging.WARNING, logger="xpm"):
        # Child defines a new parameter y, doesn't override x
        class Child(Parent):
            y: Param[int]

        Child.__getxpmtype__().arguments

    # No warning should be issued for new parameter
    assert "overrides parent parameter" not in caplog.text
