from typing import List, Optional
from experimaestro.experiments.grid import GridSearch, GenericParams
from experimaestro.experiments.configuration import ConfigurationBase, configuration
from experimaestro.utils.config import validate_attrs

@configuration()
class SubConfig:
    value: GridSearch[int] = 1

@configuration()
class MainConfig(ConfigurationBase):
    lr: GridSearch[float] = 0.01
    tags: Optional[List[str]] = None
    sub: Optional[SubConfig] = None

def test_validate_attrs_grid_search():
    data = {
        "id": "test",
        "lr": [0.1, 0.01],
        "sub": {"value": {"values_list": [1, 2, 3]}}
    }

    cfg = validate_attrs(MainConfig, data)

    assert isinstance(cfg.lr, GenericParams)
    assert cfg.lr.values_list == [0.1, 0.01]

    assert isinstance(cfg.sub.value, GenericParams)
    assert cfg.sub.value.values_list == [1, 2, 3]

def test_validate_attrs_scalar():
    data = {
        "id": "test",
        "lr": 0.05,
        "sub": {"value": 10}
    }

    cfg = validate_attrs(MainConfig, data)

    assert isinstance(cfg.lr, GenericParams)
    assert cfg.lr.value == 0.05

    assert isinstance(cfg.sub.value, GenericParams)
    assert cfg.sub.value.value == 10

def test_generate_grid_from_cli_config():
    from experimaestro.experiments.grid import generate_grid

    data = {
        "id": "test",
        "lr": 0.01,
        "grid_search": {
            "lr": [0.1, 0.01],
            "sub.value": [1, 2]
        },
        "sub": {"value": 0}
    }

    cfg = validate_attrs(MainConfig, data)
    configs, tags = generate_grid(cfg)

    assert len(configs) == 4

    # Check one permutation
    assert configs[0].lr in [0.1, 0.01]
    assert configs[0].sub.value in [1, 2]


def test_unique_value_in_tags_from_validation():
    from experimaestro.experiments.grid import generate_grid

    data = {
        "id": "test",
        "lr": 0.05,
        "sub": {"value": 10}
    }

    cfg = validate_attrs(MainConfig, data)
    configs, tags = generate_grid(cfg)

    assert len(configs) == 1
    assert tags[0] == {}

    # Test with multi-value param
    data_multi = {
        "id": "test",
        "lr": [0.05, 0.1],
        "sub": {"value": 10}
    }

    cfg_multi = validate_attrs(MainConfig, data_multi)
    configs_multi, tags_multi = generate_grid(cfg_multi)

    assert len(configs_multi) == 2
    assert tags_multi[0] == {"lr": 0.05}
    assert tags_multi[1] == {"lr": 0.1}



def test_unrecognized_key_in_validation():
    import pytest
    from pydantic import ValidationError

    data = {
        "id": "test",
        "lr": {"ranges": [0.1, 0.01]},  # "ranges" is unrecognized
    }

    with pytest.raises(ValidationError) as excinfo:
        validate_attrs(MainConfig, data)

    err_msg = str(excinfo.value)
    assert "Unrecognized keys in GridSearch parameter: ranges" in err_msg
    assert "Possible options are: range, range_mult, value, values_list, values_mult, values_range" in err_msg



