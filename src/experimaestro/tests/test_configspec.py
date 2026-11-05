"""Tests for ConfigSpec (specifications generated from a Config)"""

from enum import Enum
from pathlib import Path

import attr
import pytest

from experimaestro import (
    Config,
    Constant,
    Meta,
    Param,
    PathGenerator,
    Task,
    field,
)
from experimaestro.experiments import (
    ConfigSpec,
    spec_field,
    ExperimentSpecification,
    specification,
)
from experimaestro.experiments.grid import GridSearch, generate_grid
from experimaestro.experiments.specification import MISSING
from experimaestro.utils.config import validate_attrs


class Activation(Enum):
    RELU = "relu"
    GELU = "gelu"


class Encoder(Config):
    size: Param[int]


class Model(Config):
    hidden_dim: Param[int]
    """Size of the hidden layer"""

    lr: Param[float] = field(default=1e-3, ignore_default=True)
    dropout: Param[float] = field(default=0.1)
    activation: Param[Activation] = field(default=Activation.RELU)
    layers: Param[list[int]] = field(default_factory=list)
    data: Param[Path | None] = field(default=None, ignore_default=True)
    encoder: Param[Encoder]
    encoders: Param[list[Encoder]] = field(default_factory=list)
    version: Constant[int] = 2
    batch_size: Meta[int] = field(default=16, ignore_default=True)
    path: Meta[Path] = field(default_factory=PathGenerator("model.pt"))


class ModelSpec(ConfigSpec[Model]):
    pass


def fields(cls):
    return set(attr.fields_dict(cls))


def identifier(config):
    return config.__identifier__().all.hex()


def test_configspec_fields():
    # Configurations, constants, generated values and Meta are not included
    assert fields(ModelSpec) == {
        "hidden_dim",
        "lr",
        "dropout",
        "activation",
        "layers",
        "data",
    }

    spec = ModelSpec()
    assert spec.hidden_dim is MISSING
    assert spec.lr == 1e-3
    assert spec.layers == [] and spec.layers is not ModelSpec().layers
    assert attr.fields_dict(ModelSpec)["hidden_dim"].metadata["help"] == (
        "Size of the hidden layer"
    )


def test_configspec_options():
    class WithMeta(ConfigSpec[Model], meta=True):
        pass

    assert "batch_size" in fields(WithMeta)
    assert "path" not in fields(WithMeta)

    class Only(ConfigSpec[Model], include=["hidden_dim", "lr"]):
        pass

    assert fields(Only) == {"hidden_dim", "lr"}

    class Without(ConfigSpec[Model], exclude=["lr"]):
        pass

    assert "lr" not in fields(Without)

    with pytest.raises(TypeError, match="unknown parameters"):

        class Unknown(ConfigSpec[Model], exclude=["nope"]):
            pass


def test_configspec_declared_fields():
    class Overridden(ConfigSpec[Model]):
        hidden_dim: int = 128
        lr: float = attr.field(default=0.5)
        note: str = ""

    spec = Overridden()
    assert (spec.hidden_dim, spec.lr, spec.note) == (128, 0.5, "")
    # A field that is not a parameter is not given to the configuration
    config = spec.to_config(encoder=Encoder.C(size=2))
    assert config.hidden_dim == 128 and config.lr == 0.5


@pytest.mark.parametrize(
    "values,expected",
    [
        # int given to a float parameter
        ({"hidden_dim": 4, "lr": 1}, {"hidden_dim": 4, "lr": 1.0}),
        # values equal to the defaults
        (
            {"hidden_dim": 4, "lr": 1e-3, "dropout": 0.1},
            {"hidden_dim": 4},
        ),
        (
            {"hidden_dim": "4", "activation": "gelu", "data": "/a/b"},
            {
                "hidden_dim": 4,
                "activation": Activation.GELU,
                "data": Path("/a/b"),
            },
        ),
        ({"hidden_dim": 4, "layers": [1, 2]}, {"hidden_dim": 4, "layers": [1, 2]}),
    ],
)
def test_configspec_identifier(values, expected):
    """The configuration built from YAML has the same identifier as by hand"""
    spec = validate_attrs(ModelSpec, values)
    encoder = Encoder.C(size=2)
    assert identifier(spec.to_config(encoder=encoder)) == identifier(
        Model.C(encoder=encoder, **expected)
    )


def test_configspec_extra():
    spec = validate_attrs(ModelSpec, {"hidden_dim": 4})
    encoder = Encoder.C(size=2)
    config = spec.to_config(encoder=encoder, hidden_dim=8)
    assert config.encoder is encoder
    assert config.hidden_dim == 8

    # Required parameters missing from the spec must be given
    with pytest.raises(Exception):
        ModelSpec().to_config(encoder=encoder).__identifier__()


def test_configspec_in_specification():
    @specification
    class Spec(ExperimentSpecification):
        model: ModelSpec = attr.field(factory=ModelSpec)

    spec = validate_attrs(
        Spec, {"id": "xp", "model": {"hidden_dim": "3", "activation": "gelu"}}
    )
    assert spec.model.hidden_dim == 3
    assert spec.model.activation is Activation.GELU


def test_configspec_grid_search():
    class GridModelSpec(ConfigSpec[Model]):
        hidden_dim: GridSearch[int] = 4

    @specification
    class Spec(ExperimentSpecification):
        model: GridModelSpec = attr.field(factory=GridModelSpec)

    spec = validate_attrs(
        Spec,
        {
            "id": "xp",
            "model": {"hidden_dim": [1, 2]},
            "grid_search": {"model.dropout": ["0.2", 0.3]},
        },
    )
    with pytest.raises(ValueError, match="grid search"):
        spec.model.to_config()

    specs, tags = generate_grid(spec)
    assert [(s.model.hidden_dim, s.model.dropout) for s in specs] == [
        (1, 0.2),
        (1, 0.3),
        (2, 0.2),
        (2, 0.3),
    ]
    encoder = Encoder.C(size=2)
    assert identifier(specs[0].model.to_config(encoder=encoder)) == identifier(
        Model.C(encoder=encoder, hidden_dim=1, dropout=0.2)
    )


def test_configspec_task():
    class Train(Task):
        epochs: Param[int] = field(default=10)
        model: Param[Model]

        def execute(self):
            pass

    class TrainSpec(ConfigSpec[Train]):
        pass

    assert fields(TrainSpec) == {"epochs"}
    model = Model.C(hidden_dim=1, encoder=Encoder.C(size=2))
    assert isinstance(TrainSpec(epochs=3).to_config(model=model), Train)


def test_configspec_requires_config_class():
    with pytest.raises(TypeError, match="ConfigSpec"):

        class NoConfig(ConfigSpec):
            pass


def test_configspec_subclass():
    class Sub(ModelSpec):
        hidden_dim: int = 7

    spec = validate_attrs(Sub, {"lr": "0.5"})
    config = spec.to_config(encoder=Encoder.C(size=1))
    assert (config.hidden_dim, config.lr) == (7, 0.5)


def test_configspec_direct():
    """ConfigSpec[Model] can be used as a field type, with a default"""
    assert ConfigSpec[Model] is ConfigSpec[Model]

    @specification
    class Spec(ExperimentSpecification):
        model: ConfigSpec[Model]
        other: "ConfigSpec[Model]"

    spec = validate_attrs(Spec, {"id": "xp", "model": {"hidden_dim": "3"}})
    assert spec.model.hidden_dim == 3
    assert spec.other.hidden_dim is MISSING

    direct = ConfigSpec[Model](hidden_dim=2)
    assert direct.hidden_dim == 2 and type(direct) is type(spec.model)
    encoder = Encoder.C(size=2)
    assert identifier(direct.to_config(encoder=encoder)) == identifier(
        Model.C(encoder=encoder, hidden_dim=2)
    )


def test_configspec_spec_field():
    @specification
    class Spec(ExperimentSpecification):
        model: ConfigSpec[Model] = spec_field(
            meta=True,
            exclude=["dropout"],
            defaults={"hidden_dim": 8, "layers": [1]},
            grid=["lr"],
        )
        plain: ConfigSpec[Model] = spec_field()

    model_fields = fields(attr.fields_dict(Spec)["model"].type)
    assert "batch_size" in model_fields and "dropout" not in model_fields
    assert "dropout" in fields(attr.fields_dict(Spec)["plain"].type)

    spec = Spec(id="xp")
    assert (spec.model.hidden_dim, spec.model.layers) == (8, [1])
    assert spec.model.layers is not Spec(id="xp").model.layers

    spec = validate_attrs(Spec, {"id": "xp", "model": {"lr": [0.1, 0.2]}})
    specs, _ = generate_grid(spec)
    assert [s.model.lr for s in specs] == [0.1, 0.2]

    with pytest.raises(TypeError, match="cannot be read from YAML"):

        @specification
        class Bad(ExperimentSpecification):
            model: ConfigSpec[Model] = spec_field(defaults={"encoder": None})

    with pytest.raises(TypeError, match="needs a ConfigSpec type"):

        @specification
        class NotSpec(ExperimentSpecification):
            model: int = spec_field()


def test_configspec_subclass_options():
    class Base(ConfigSpec[Model], meta=True):
        hidden_dim: int = 7

    class Child(Base, defaults={"lr": 0.5}):
        pass

    spec = Child()
    # Options and declared fields are inherited
    assert (spec.hidden_dim, spec.lr, spec.batch_size) == (7, 0.5, 16)

    with pytest.raises(TypeError, match="cannot remove"):

        class Removing(Base, exclude=["lr"]):
            pass

    @specification
    class Spec(ExperimentSpecification):
        model: Base = spec_field(defaults={"dropout": 0.3})

    assert (Spec(id="xp").model.hidden_dim, Spec(id="xp").model.dropout) == (7, 0.3)
