# Tests for identifier computation

from typing import List
from experimaestro import (
    Param,
    deprecate,
    deprecated,
    Config,
)


def getidentifier(x):
    return x.__xpm__.identifier.all


def assert_equal(a, b, message=""):
    assert getidentifier(a) == getidentifier(b), message


def assert_notequal(a, b, message=""):
    assert getidentifier(a) != getidentifier(b), message


def test_deprecate_class():
    """Test that when submitting the task, the computed identifier is the one of
    the new class"""

    class NewConfig(Config):
        __xpmid__ = "new"

    @deprecate
    class OldConfig(NewConfig):
        __xpmid__ = "old"

    class DerivedConfig(NewConfig):
        __xpmid__ = "derived"

    assert_notequal(
        NewConfig(), DerivedConfig(), "A derived configuration has another ID"
    )
    assert_equal(
        NewConfig(), OldConfig(), "Deprecated and new configuration have the same ID"
    )


def test_deprecated_attribute():
    class Values(Config):
        values: Param[List[int]] = []

        @deprecate
        def value(self, x):
            self.values = [x]

    assert_equal(Values(values=[1]), Values(value=1))


# New configuration has param x instead of y
class ConfigurationA(Config):
    x: Param[int]


# This is the old configuration
@deprecated(ConfigurationA)
class ConfigurationA_v0(Config):
    y: Param[int]

    @classmethod
    def handles(cls, y=None, **kwargs):
        """Returns the new configuration"""
        return y is not None

    def converts(self):
        return ConfigurationA(x=self.y)


class ConfigurationB(Config):
    a: Param[ConfigurationA]

    def __validate__(self):
        assert isinstance(self.a, ConfigurationA), "Validation: a is still an old type"


def test_deprecated_config_identifier():
    cf_v0 = ConfigurationA(y=1)

    # We should not change the identifier
    cf_v0_orig = ConfigurationA_v0(y=1)
    assert_equal(cf_v0, cf_v0_orig)


def test_deprecated_config_use():
    cf_v0 = ConfigurationA(y=1)

    # We should not change the identifier
    cf_v0_orig = ConfigurationA_v0(y=1)
    assert_equal(cf_v0, cf_v0_orig)

    # The new class can be used instead of the old one
    b = ConfigurationB(a=cf_v0)
    b_inst = b.instance()

    assert isinstance(b_inst.a, ConfigurationA), "a is still an old type"
    assert b_inst.a.x == 1
