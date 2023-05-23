from experimaestro import Config, Task, Param, PathBasedSerializedConfig, copyconfig
from experimaestro.tests.utils import TemporaryExperiment


class SubModel(Config):
    pass


class Model(Config):
    submodel: Param[SubModel]

    def __post_init__(self):
        self.initialized = False
        self.submodel.initialized = False


class SerializedModel(PathBasedSerializedConfig):
    def initialize(self):
        self.config.initialized = True
        self.config.submodel.initialized = True


class Trainer(Task):
    model: Param[Config]

    def taskoutputs(self):
        return SerializedModel(config=copyconfig(self.model))

    def execute(self):
        assert not self.model.initialized, "Model not initialized"


class Evaluate(Task):
    model: Param[Config]
    is_submodel: Param[bool] = False

    def execute(self):
        assert self.model.initialized, "Model not initialized"
        if self.is_submodel:
            assert isinstance(self.model, SubModel)
        else:
            assert isinstance(self.model, Model)


def test_serializers_xp():
    with TemporaryExperiment("serializers", maxwait=10, port=0):
        model = Model(submodel=SubModel())
        trained_model: Model = Trainer(model=model).submit()

        # Use the model itself
        Evaluate(model=trained_model).submit()

        # Use a submodel
        Evaluate(model=trained_model.submodel, is_submodel=True).submit()
