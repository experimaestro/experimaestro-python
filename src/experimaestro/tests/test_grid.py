from typing import Optional
from experimaestro.experiments.grid import GenericParams, GridSearch, generate_grid, discover_grid_params
from experimaestro.experiments.configuration import ConfigurationBase, configuration

@configuration()
class MySubConfig:
    param: GridSearch[int] = 1

@configuration()
class MyConfig(ConfigurationBase):
    lr: GridSearch[float] = 0.001
    batch_size: GridSearch[int] = 32
    sub: Optional[MySubConfig] = None

def test_generic_params_coercion():
    # Scalar
    gp = GenericParams.from_any(10)
    assert gp.value == 10
    assert not gp.is_grid
    assert gp.as_list() == [10]

    # List
    gp = GenericParams.from_any([1, 2, 3])
    assert gp.values_list == [1, 2, 3]
    assert gp.is_grid
    assert gp.as_list() == [1, 2, 3]

    # Range
    gp = GenericParams.from_any({"values_range": [0, 3]})
    assert gp.is_grid
    assert gp.as_list() == [0, 1, 2]

    # Range with step
    gp = GenericParams.from_any({"values_range": [0, 10, 2]})
    assert gp.is_grid
    assert gp.as_list() == [0, 2, 4, 6, 8]

    # Range alias support
    gp = GenericParams.from_any({"range": [16, 64, 10]})
    assert gp.is_grid
    assert gp.as_list() == [16, 26, 36, 46, 56]

    # Range multiplier support
    gp = GenericParams.from_any({"range_mult": [1e-4, 4, 10]})
    assert gp.is_grid
    assert gp.as_list() == [1e-4, 1e-3, 1e-2, 1e-1]

def test_discover_grid_params():
    cfg = MyConfig(id="test", lr=GenericParams(values_list=[0.1, 0.01]), sub=MySubConfig(param=10))
    grid = discover_grid_params(cfg)
    assert "lr" in grid
    assert grid["lr"].values_list == [0.1, 0.01]
    assert "sub.param" not in grid

    cfg.sub.param = GenericParams(values_list=[1, 2])
    grid = discover_grid_params(cfg)
    assert "sub.param" in grid
    assert grid["sub.param"].values_list == [1, 2]

def test_generate_grid():
    cfg = MyConfig(id="test", lr=[0.1, 0.01], batch_size=32)
    # Manual coercion for now because we didn't use validate_attrs yet
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)

    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert configs[0].lr == 0.1
    assert configs[1].lr == 0.01
    assert configs[0].batch_size == 32
    assert configs[1].batch_size == 32
    assert tags[0]["lr"] == 0.1

    # Mixed with explicit grid_search
    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)
    cfg.grid_search = {"batch_size": [16, 32, 64]}

    configs, tags = generate_grid(cfg)
    assert len(configs) == 3
    assert configs[0].batch_size == 16
    assert configs[1].batch_size == 32
    assert configs[2].batch_size == 64
    assert tags[0]["batch_size"] == 16

def test_nested_generate_grid():
    cfg = MyConfig(
        id="test",
        lr=0.1,
        sub=MySubConfig(param=[1, 2])
    )
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.sub.param = GenericParams.from_any(cfg.sub.param)

    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert configs[0].sub.param == 1
    assert configs[1].sub.param == 2
    assert tags[0]["sub.param"] == 1


def test_unrecognized_key_error():
    import pytest
    with pytest.raises(ValueError, match="Unrecognized keys.*Possible options are"):
        GenericParams.from_any({"unrecognized_key": 123})



def test_unique_value_in_tags():
    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)

    configs, tags = generate_grid(cfg)
    assert len(configs) == 1
    assert tags[0] == {}

    # Test that multi-value params are tagged while single-value params in the same grid are not
    cfg_multi = MyConfig(id="test", lr=[0.1, 0.01], batch_size=32)
    cfg_multi.lr = GenericParams.from_any(cfg_multi.lr)
    cfg_multi.batch_size = GenericParams.from_any(cfg_multi.batch_size)

    configs_multi, tags_multi = generate_grid(cfg_multi)
    assert len(configs_multi) == 2
    assert tags_multi[0] == {"lr": 0.1}
    assert tags_multi[1] == {"lr": 0.01}



