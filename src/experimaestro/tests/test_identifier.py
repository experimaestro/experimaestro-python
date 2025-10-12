# Tests for identifier computation

import json
from pathlib import Path
from typing import Dict, List, Optional
from experimaestro import (
    Param,
    deprecate,
    Config,
    Constant,
    Meta,
    Option,
    PathGenerator,
    field,
    Task,
    LightweightTask,
)
from experimaestro.core.objects import (
    ConfigInformation,
    ConfigWalkContext,
    setmeta,
)
from experimaestro.scheduler.workspace import RunMode


class A(Config):
    a: Param[int]
    pass


class B(Config):
    a: Param[int]
    pass


class C(Config):
    a: Param[int] = 1
    b: Param[int]


class CField(Config):
    a: Param[int] = field(default_factory=lambda: 1)
    b: Param[int]


class D(Config):
    a: Param[A]


class Float(Config):
    value: Param[float]


class Values(Config):
    value1: Param[float]
    value2: Param[float]


def getidentifier(x):
    return x.__xpm__.identifier.all


def assert_equal(a, b, message=""):
    assert getidentifier(a) == getidentifier(b), message


def assert_notequal(a, b, message=""):
    assert getidentifier(a) != getidentifier(b), message


def test_identifier_int():
    assert_equal(A.C(a=1), A.C(a=1))


def test_identifier_different_type():
    assert_notequal(A.C(a=1), B.C(a=1))


def test_identifier_order():
    assert_equal(Values.C(value1=1, value2=2), Values.C(value2=2, value1=1))


def test_identifier_default():
    assert_equal(C.C(a=1, b=2), C.C(b=2))


def test_identifier_default_field():
    assert_equal(CField.C(a=1, b=2), CField.C(b=2))


def test_identifier_inner_eq():
    assert_equal(D.C(a=A.C(a=1)), D.C(a=A.C(a=1)))


def test_identifier_float():
    assert_equal(Float.C(value=1), Float.C(value=1))


def test_identifier_float2():
    assert_equal(Float.C(value=1.0), Float.C(value=1))


# --- Argument name


def test_identifier_name():
    """The identifier fully determines the hash code"""

    class Config0(Config):
        __xpmid__ = "test.identifier.argumentname"
        a: Param[int]

    class Config1(Config):
        __xpmid__ = "test.identifier.argumentname"
        b: Param[int]

    class Config3(Config):
        __xpmid__ = "test.identifier.argumentname"
        a: Param[int]

    assert_notequal(Config0.C(a=2), Config1.C(b=2))
    assert_equal(Config0.C(a=2), Config3.C(a=2))


# --- Test option


def test_identifier_option():
    class OptionConfig(Config):
        __xpmid__ = "test.identifier.option"
        a: Param[int]
        b: Option[int] = 1

    assert_notequal(OptionConfig.C(a=2), OptionConfig.C(a=1))
    assert_equal(OptionConfig.C(a=1, b=2), OptionConfig.C(a=1))
    assert_equal(OptionConfig.C(a=1, b=2), OptionConfig.C(a=1, b=2))


# --- Dictionnary


def test_identifier_dict():
    """Test identifiers of dictionary structures"""

    class B(Config):
        x: Param[int]

    class A(Config):
        bs: Param[Dict[str, B]]

    assert_equal(A.C(bs={"b1": B.C(x=1)}), A.C(bs={"b1": B.C(x=1)}))
    assert_equal(
        A.C(bs={"b1": B.C(x=1), "b2": B.C(x=2)}),
        A.C(bs={"b2": B.C(x=2), "b1": B.C(x=1)}),
    )

    assert_notequal(A.C(bs={"b1": B.C(x=1)}), A.C(bs={"b1": B.C(x=2)}))
    assert_notequal(A.C(bs={"b1": B.C(x=1)}), A.C(bs={"b2": B.C(x=1)}))


# --- Ignore paths


class TypeWithPath(Config):
    a: Param[int]
    path: Param[Path]


def test_identifier_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath.C(a=1, path="/a/b"), TypeWithPath.C(a=1, path="/c/d"))
    assert_notequal(TypeWithPath.C(a=2, path="/a/b"), TypeWithPath.C(a=1, path="/c/d"))


# --- Test with added arguments


def test_identifier_pathoption():
    """Path arguments should be ignored"""

    class A_with_path(Config):
        __xpmid__ = "pathoption_test"
        a: Param[int]
        path: Meta[Path] = field(default_factory=PathGenerator("path"))

    class A_without_path(Config):
        __xpmid__ = "pathoption_test"
        a: Param[int]

    assert_equal(A_with_path.C(a=1), A_without_path.C(a=1))


def test_identifier_enum():
    """test enum parameters"""
    from enum import Enum

    class EnumParam(Enum):
        FIRST = 0
        SECOND = 1

    class EnumConfig(Config):
        a: Param[EnumParam]

    assert_notequal(EnumConfig.C(a=EnumParam.FIRST), EnumConfig.C(a=EnumParam.SECOND))
    assert_equal(EnumConfig.C(a=EnumParam.FIRST), EnumConfig.C(a=EnumParam.FIRST))


def test_identifier_addnone():
    """Test the case of new parameter (with None default)"""

    class B(Config):
        x: Param[int]

    class A_with_b(Config):
        __xpmid__ = "defaultnone"
        b: Param[Optional[B]] = None

    class A(Config):
        __xpmid__ = "defaultnone"

    assert_equal(A_with_b.C(), A.C())
    assert_notequal(A_with_b.C(b=B.C(x=1)), A.C())


def test_identifier_defaultnew():
    """Path arguments should be ignored"""

    class A_with_b(Config):
        __xpmid__ = "defaultnew"

        a: Param[int]
        b: Param[int] = 1

    class A(Config):
        __xpmid__ = "defaultnew"
        a: Param[int]

    assert_equal(A_with_b.C(a=1, b=1), A.C(a=1))
    assert_equal(A_with_b.C(a=1), A.C(a=1))


def test_identifier_taskconfigidentifier():
    """Test whether the embedded task arguments make the configuration different"""

    class MyConfig(Config):
        a: Param[int]

    class MyTask(Task):
        x: Param[int]

        def task_outputs(self, dep):
            return dep(MyConfig.C(a=1))

    assert_equal(
        MyTask.C(x=1).submit(run_mode=RunMode.DRY_RUN),
        MyTask.C(x=1).submit(run_mode=RunMode.DRY_RUN),
    )
    assert_notequal(
        MyTask.C(x=2).submit(run_mode=RunMode.DRY_RUN),
        MyTask.C(x=1).submit(run_mode=RunMode.DRY_RUN),
    )


def test_identifier_constant():
    """Test if constants are taken into account for signature computation"""

    class A1(Config):
        __xpmid__ = "test.constant"
        version: Constant[int] = 1

    class A1bis(Config):
        __xpmid__ = "test.constant"
        version: Constant[int] = 1

    assert_equal(A1.C(), A1bis.C())

    class A2(Config):
        __xpmid__ = "test.constant"
        version: Constant[int] = 2

    assert_notequal(A1.C(), A2.C())


def test_identifier_deprecated_class():
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
        NewConfig.C(), DerivedConfig.C(), "A derived configuration has another ID"
    )
    assert_equal(
        NewConfig.C(),
        OldConfig.C(),
        "Deprecated and new configuration have the same ID",
    )


def test_identifier_deprecated_attribute():
    class Values(Config):
        values: Param[List[int]] = []

        @deprecate
        def value(self, x):
            self.values = [x]

    assert_equal(Values.C(values=[1]), Values.C(value=1))


class MetaA(Config):
    x: Param[int]


def test_identifier_meta():
    """Test forced meta-parameter"""

    class B(Config):
        a: Param[MetaA]

    class C(Config):
        a: Meta[MetaA]

    class ArrayConfig(Config):
        array: Param[List[MetaA]]

    class DictConfig(Config):
        params: Param[Dict[str, MetaA]]

    # As meta
    assert_notequal(B.C(a=MetaA.C(x=1)), B.C(a=MetaA.C(x=2)))
    assert_equal(B.C(a=setmeta(MetaA.C(x=1), True)), B.C(a=setmeta(MetaA.C(x=2), True)))

    # As parameter
    assert_equal(C.C(a=MetaA.C(x=1)), C.C(a=MetaA.C(x=2)))
    assert_notequal(
        C.C(a=setmeta(MetaA.C(x=1), False)), C.C(a=setmeta(MetaA.C(x=2), False))
    )

    # Array with mixed
    assert_equal(
        ArrayConfig.C(array=[MetaA.C(x=1)]),
        ArrayConfig.C(array=[MetaA.C(x=1), setmeta(MetaA.C(x=2), True)]),
    )

    # Array with empty list
    assert_equal(
        ArrayConfig.C(array=[]), ArrayConfig.C(array=[setmeta(MetaA.C(x=2), True)])
    )

    # Dict with mixed
    assert_equal(
        DictConfig.C(params={"a": MetaA.C(x=1)}),
        DictConfig.C(params={"a": MetaA.C(x=1), "b": setmeta(MetaA.C(x=2), True)}),
    )


def test_identifier_meta_default_dict():
    class DictConfig(Config):
        params: Param[Dict[str, MetaA]] = {}

    assert_equal(
        DictConfig.C(params={}),
        DictConfig.C(params={"b": setmeta(MetaA.C(x=2), True)}),
    )

    # Dict with mixed
    assert_equal(
        DictConfig.C(params={"a": MetaA.C(x=1)}),
        DictConfig.C(params={"a": MetaA.C(x=1), "b": setmeta(MetaA.C(x=2), True)}),
    )


def test_identifier_meta_default_array():
    class ArrayConfigWithDefault(Config):
        array: Param[List[MetaA]] = []

    # Array (with default) with mixed
    assert_equal(
        ArrayConfigWithDefault.C(array=[MetaA.C(x=1)]),
        ArrayConfigWithDefault.C(array=[MetaA.C(x=1), setmeta(MetaA.C(x=2), True)]),
    )
    # Array (with default) with empty list
    assert_equal(
        ArrayConfigWithDefault.C(array=[]),
        ArrayConfigWithDefault.C(array=[setmeta(MetaA.C(x=2), True)]),
    )


def test_identifier_init_task():
    class MyConfig(Config):
        pass

    class IdentifierInitTask(LightweightTask):
        pass

    class IdentifierInitTask2(Task):
        pass

    class IdentifierTask(Task):
        x: Param[MyConfig]

    task = IdentifierTask.C(x=MyConfig.C()).submit(run_mode=RunMode.DRY_RUN)
    task_with_pre = IdentifierTask.C(x=MyConfig.C()).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[IdentifierInitTask.C(), IdentifierInitTask2.C()],
    )
    task_with_pre_2 = IdentifierTask.C(x=MyConfig.C()).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[IdentifierInitTask.C(), IdentifierInitTask2.C()],
    )
    task_with_pre_3 = IdentifierTask.C(x=MyConfig.C()).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[IdentifierInitTask2.C(), IdentifierInitTask.C()],
    )

    assert_notequal(task, task_with_pre, "Should be different with init-task")
    assert_equal(task_with_pre, task_with_pre_2, "Same parameters")
    assert_notequal(task_with_pre, task_with_pre_3, "Other parameters")


def test_identifier_init_task_dep():
    class Loader(LightweightTask):
        param1: Param[float]

        def execute(self):
            pass

    class FirstTask(Task):
        def task_outputs(self, dep):
            return dep(Loader.C(param1=1))

        def execute(self):
            pass

    class SecondTask(Task):
        param3: Param[int]

        def execute(self):
            pass

    # Two identical tasks
    task_a_1 = FirstTask.C()
    task_a_2 = FirstTask.C()
    assert_equal(task_a_1, task_a_2)

    # We process them with two different init tasks
    loader_1 = task_a_1.submit(
        init_tasks=[Loader.C(param1=0.5)], run_mode=RunMode.DRY_RUN
    )
    loader_2 = task_a_2.submit(
        init_tasks=[Loader.C(param1=5)], run_mode=RunMode.DRY_RUN
    )
    assert_notequal(loader_1, loader_2)

    # Now, we process
    c_1 = SecondTask.C(param3=2).submit(init_tasks=[loader_1], run_mode=RunMode.DRY_RUN)

    c_2 = SecondTask.C(param3=2).submit(init_tasks=[loader_2], run_mode=RunMode.DRY_RUN)
    assert_notequal(c_1, c_2)


# --- Check configuration reloads


def check_reload(config):
    """Checks that the serialized configuration, when reloaded,
    gives the same identifier"""
    old_identifier = config.__xpm__.identifier.all

    # Get the data structure
    data = json.loads(config.__xpm__.__json__())

    # Reload the configuration
    new_config = ConfigInformation.fromParameters(
        data, as_instance=False, discard_id=True
    )
    assert new_config.__xpm__._identifier is None
    new_identifier = new_config.__xpm__.identifier.all

    assert new_identifier == old_identifier


class IdentifierReloadConfig(Config):
    id: Param[str]


def test_identifier_reload_config():
    # Creates the configuration
    check_reload(IdentifierReloadConfig.C(id="123"))


class IdentifierReload(Task):
    id: Param[str]

    def task_outputs(self, dep) -> IdentifierReloadConfig.C:
        return IdentifierReloadConfig.C(id=self.id)


class IdentifierReloadDerived(Config):
    task: Param[IdentifierReloadConfig]


def test_identifier_reload_taskoutput():
    """When using a task output, the identifier should not be different"""

    # Creates the configuration
    task = IdentifierReload.C(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadDerived.C(task=task)
    check_reload(config)


class IdentifierReloadTask(Task):
    id: Param[str]


class IdentifierReloadTaskConfig(Config):
    x: Param[int]


class IdentifierReloadTaskDerived(Config):
    task: Param[IdentifierReloadTask]
    other: Param[IdentifierReloadTaskConfig]


def test_identifier_reload_task_direct():
    """When using a direct task output, the identifier should not be different"""

    # Creates the configuration
    task = IdentifierReloadTask.C(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskDerived.C(
        task=task, other=IdentifierReloadTaskConfig.C(x=2)
    )
    check_reload(config)


def test_identifier_reload_meta():
    """Test identifier don't change when using meta"""
    # Creates the configuration
    task = IdentifierReloadTask.C(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskDerived.C(
        task=task, other=setmeta(IdentifierReloadTaskConfig.C(x=2), True)
    )
    check_reload(config)


class LoopA(Config):
    param_b: Param["LoopB"]


class LoopB(Config):
    param_c: Param["LoopC"]


class LoopC(Config):
    param_a: Param["LoopA"]
    param_b: Param["LoopB"]


def test_identifier_loop():
    c = LoopC.C()
    b = LoopB.C(param_c=c)
    a = LoopA.C(param_b=b)
    c.param_a = a
    c.param_b = b

    configs = [a, b, c]
    identifiers = [[] for _ in configs]

    for i in range(len(configs)):
        context = ConfigWalkContext()
        configs[i].__xpm__.seal(context)
        assert all([c.__xpm__._sealed for c in configs])

        for j in range(len(configs)):
            identifiers[j].append(configs[j].__xpm__.identifier)

        configs[i].__xpm__.__unseal__()
        assert all([not c.__xpm__._sealed for c in configs])

    # Check
    for i in range(len(configs)):
        for j in range(1, len(configs)):
            assert identifiers[i][0] == identifiers[i][j]
