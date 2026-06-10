from experimaestro import Config, Param
import pytest

class SubModule(Config):
    y: Param[int]

class MainModule(Config):
    sub: Param[SubModule]

def test_xpmconfig_recursive():
    """Test that xpmconfig is available recursively on instantiated objects"""
    config = MainModule.C(sub=SubModule.C(y=20))
    instance = config.instance(keep=True)

    # Check root
    assert instance.xpmconfig is config
    assert isinstance(instance.xpmconfig, MainModule.XPMConfig)

    # Check sub-object (recursive tracking)
    assert instance.sub.xpmconfig is config.sub
    assert isinstance(instance.sub.xpmconfig, SubModule.XPMConfig)
    assert instance.sub.xpmconfig.y == 20

def test_xpmconfig_on_config():
    """Test that xpmconfig returns the object itself when called on a configuration"""
    config = SubModule.C(y=10)
    assert config.xpmconfig is config

def test_xpmconfig_no_keep():
    """Test that xpmconfig raises AttributeError if keep=False was used"""
    config = SubModule.C(y=10)
    instance = config.instance(keep=False)

    with pytest.raises(AttributeError, match="has no configuration tracked"):
        _ = instance.xpmconfig

def test_xpmconfig_default_behavior():
    """Test that xpmconfig is available by default (keep=True is default)"""
    config = SubModule.C(y=10)
    instance = config.instance()
    assert instance.xpmconfig is config
