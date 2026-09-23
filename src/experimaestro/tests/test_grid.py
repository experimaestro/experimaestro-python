from typing import Optional
from experimaestro.experiments.grid import (
    GenericParams,
    GridSearch,
    generate_grid,
    discover_grid_params,
)
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
    cfg = MyConfig(
        id="test", lr=GenericParams(values_list=[0.1, 0.01]), sub=MySubConfig(param=10)
    )
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
    cfg = MyConfig(id="test", lr=0.1, sub=MySubConfig(param=[1, 2]))
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


def test_config_dicts_basic():
    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)
    cfg.grid_search = {
        "config_dicts": [
            {"lr": 0.01, "batch_size": 16},
            {"lr": 0.001, "batch_size": 64},
        ]
    }
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert configs[0].lr == 0.01
    assert configs[0].batch_size == 16
    assert configs[1].lr == 0.001
    assert configs[1].batch_size == 64
    assert tags[0] == {"lr": 0.01, "batch_size": 16}
    assert tags[1] == {"lr": 0.001, "batch_size": 64}


def test_config_dicts_key_mismatch_error():
    import pytest

    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)

    # Missing key
    cfg.grid_search = {
        "config_dicts": [
            {"lr": 0.01, "batch_size": 16},
            {"lr": 0.001},
        ]
    }
    with pytest.raises(
        ValueError,
        match=r"All dictionaries in 'config_dicts' must have identical keys.*missing keys: \['batch_size'\]",
    ):
        generate_grid(cfg)

    # Extra key
    cfg.grid_search = {
        "config_dicts": [
            {"lr": 0.01},
            {"lr": 0.001, "batch_size": 16},
        ]
    }
    with pytest.raises(
        ValueError,
        match=r"All dictionaries in 'config_dicts' must have identical keys.*extra keys: \['batch_size'\]",
    ):
        generate_grid(cfg)


def test_config_dicts_invalid_structure_error():
    import pytest

    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)

    # Non-list
    cfg.grid_search = {"config_dicts": "invalid"}
    with pytest.raises(
        ValueError, match=r"'config_dicts' in grid_search must be a list of dictionaries"
    ):
        generate_grid(cfg)

    # Empty list
    cfg.grid_search = {"config_dicts": []}
    with pytest.raises(
        ValueError, match=r"'config_dicts' in grid_search must contain at least one dictionary"
    ):
        generate_grid(cfg)

    # Non-dict element
    cfg.grid_search = {"config_dicts": [123]}
    with pytest.raises(
        ValueError, match=r"All elements in 'config_dicts' must be dictionaries"
    ):
        generate_grid(cfg)

    # Empty dict
    cfg.grid_search = {"config_dicts": [{}]}
    with pytest.raises(
        ValueError, match=r"Dictionary at index 0 in 'config_dicts' is empty"
    ):
        generate_grid(cfg)


def test_config_dicts_collision_error():
    import pytest

    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)
    cfg.grid_search = {
        "lr": [0.1, 0.01],
        "config_dicts": [{"lr": 0.05, "batch_size": 16}],
    }
    with pytest.raises(
        ValueError,
        match=r"Conflicting grid search definition: parameters \['lr'\] are defined both in 'config_dicts' and as independent grid search parameters",
    ):
        generate_grid(cfg)


def test_config_dicts_with_independent_grid():
    cfg = MyConfig(id="test", lr=[0.1, 0.01], batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)
    cfg.grid_search = {
        "config_dicts": [
            {"batch_size": 16},
            {"batch_size": 64},
        ]
    }
    configs, tags = generate_grid(cfg)
    assert len(configs) == 4
    assert configs[0].lr == 0.1 and configs[0].batch_size == 16
    assert configs[1].lr == 0.1 and configs[1].batch_size == 64
    assert configs[2].lr == 0.01 and configs[2].batch_size == 16
    assert configs[3].lr == 0.01 and configs[3].batch_size == 64

    assert tags[0] == {"lr": 0.1, "batch_size": 16}
    assert tags[1] == {"lr": 0.1, "batch_size": 64}
    assert tags[2] == {"lr": 0.01, "batch_size": 16}
    assert tags[3] == {"lr": 0.01, "batch_size": 64}


def test_config_dicts_tags_unique_filter():
    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)

    # batch_size is identical in both dicts, lr varies -> only lr in tags
    cfg.grid_search = {
        "config_dicts": [
            {"lr": 0.01, "batch_size": 32},
            {"lr": 0.001, "batch_size": 32},
        ]
    }
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert tags[0] == {"lr": 0.01}
    assert tags[1] == {"lr": 0.001}

    # Single dict in config_dicts -> tags should be empty
    cfg_single = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg_single.lr = GenericParams.from_any(cfg_single.lr)
    cfg_single.batch_size = GenericParams.from_any(cfg_single.batch_size)
    cfg_single.grid_search = {
        "config_dicts": [{"lr": 0.01, "batch_size": 16}]
    }
    configs_single, tags_single = generate_grid(cfg_single)
    assert len(configs_single) == 1
    assert tags_single[0] == {}


def test_config_dicts_nested_paths():
    cfg = MyConfig(id="test", lr=0.1, sub=MySubConfig(param=1))
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.sub.param = GenericParams.from_any(cfg.sub.param)
    cfg.grid_search = {
        "config_dicts": [
            {"sub.param": 10},
            {"sub.param": 20},
        ]
    }
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert configs[0].sub.param == 10
    assert configs[1].sub.param == 20
    assert tags[0] == {"sub.param": 10}
    assert tags[1] == {"sub.param": 20}


def test_config_dicts_generic_params_wrapper():
    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)
    # config_dicts wrapped in GenericParams (e.g. from Dict[str, GridSearch[Any]])
    cfg.grid_search = {
        "config_dicts": GenericParams.from_any([
            {"lr": 0.01, "batch_size": 16},
            {"lr": None, "batch_size": 32},
        ])
    }
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert configs[0].lr == 0.01
    assert configs[0].batch_size == 16
    assert configs[1].lr is None
    assert configs[1].batch_size == 32
    assert tags[0] == {"lr": 0.01, "batch_size": 16}
    assert tags[1] == {"lr": None, "batch_size": 32}


def test_generic_params_none_scalar():
    gp = GenericParams.from_any(None)
    assert gp.value is None
    assert not gp.is_grid
    assert gp.as_list() == [None]


def test_config_dicts_with_none_scalar_field():
    """Verify that a scalar GenericParams with value=None (e.g. reg_budget: null) does not collapse grid search configs."""
    cfg = MyConfig(id="test", lr=0.1, batch_size=32)
    cfg.lr = GenericParams.from_any(cfg.lr)
    cfg.batch_size = GenericParams.from_any(cfg.batch_size)
    # sub is None and wrapped as a scalar GenericParams(value=None)
    cfg.sub = GenericParams.from_any(None)

    cfg.grid_search = {
        "config_dicts": [
            {"lr": 0.01, "batch_size": 16},
            {"lr": 0.001, "batch_size": 64},
        ]
    }
    configs, tags = generate_grid(cfg)
    assert len(configs) == 2
    assert configs[0].lr == 0.01
    assert configs[0].batch_size == 16
    assert configs[0].sub is None
    assert configs[1].lr == 0.001
    assert configs[1].batch_size == 64
    assert configs[1].sub is None



