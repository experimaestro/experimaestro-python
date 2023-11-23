from experimaestro import (
    Config,
    Task,
    Param,
    SerializationLWTask,
    copyconfig,
    state_dict,
    from_state_dict,
)
from experimaestro.core.context import SerializationContext
from experimaestro.core.objects import TypeConfig
from experimaestro.tests.utils import TemporaryExperiment


class SubModel(Config):
    pass


class Model(Config):
    submodel: Param[SubModel]

    def __post_init__(self):
        self.initialized = False
        self.submodel.initialized = False


class LoadModel(SerializationLWTask):
    def execute(self):
        self.value.initialized = True
        self.value.submodel.initialized = True


class Trainer(Task):
    model: Param[Config]

    def task_outputs(self, dep):
        model = copyconfig(self.model)
        return model.add_pretasks(dep(LoadModel(value=model)))

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
    with TemporaryExperiment("serializers", maxwait=20, port=0):
        model = Model(submodel=SubModel())
        trained_model: Model = Trainer(model=model).submit()

        # Use the model itself
        Evaluate(model=trained_model).submit()

        # Use a submodel
        Evaluate(model=trained_model.submodel, is_submodel=True).add_pretasks_from(
            trained_model
        ).submit()


class Object1(Config):
    pass


class Object2(Config):
    object: Param[Object1]


def test_serializers_serialization():
    context = SerializationContext(save_directory=None)

    obj1 = Object1()
    obj2 = Object2(object=obj1)

    data = state_dict(context, [obj1, obj2])

    [obj1, obj2] = from_state_dict(data)
    assert isinstance(obj1, Object1) and isinstance(obj1, TypeConfig)
    assert isinstance(obj2, Object2)
    assert obj2.object is obj1

    [obj1, obj2] = from_state_dict(data, as_instance=True)
    assert isinstance(obj1, Object1) and not isinstance(obj1, TypeConfig)
    assert isinstance(obj2, Object2)
    assert obj2.object is obj1
