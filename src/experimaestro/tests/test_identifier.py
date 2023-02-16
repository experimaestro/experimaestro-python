# Tests for identifier computation

import json
from pathlib import Path
from typing import Dict, List, Optional
from experimaestro import (
    config,
    Param,
    param,
    task,
    deprecate,
    Config,
    Constant,
    Meta,
    Option,
    pathgenerator,
    Annotated,
    Task,
)
from experimaestro.core.objects import ConfigInformation, TaskOutput, setmeta
from experimaestro.scheduler.workspace import RunMode


@config()
class A:
    a: Param[int]
    pass


@config()
class B:
    a: Param[int]
    pass


@config()
class C:
    a: Param[int] = 1
    b: Param[int]


@config()
class D:
    a: Param[A]


@config()
class Float:
    value: Param[float]


@config()
class Values:
    value1: Param[float]
    value2: Param[float]


def getidentifier(x):
    if isinstance(x, TaskOutput):
        return x.__xpm__.identifier.all
    return x.__xpm__.identifier.all


def assert_equal(a, b, message=""):
    assert getidentifier(a) == getidentifier(b), message


def assert_notequal(a, b, message=""):
    assert getidentifier(a) != getidentifier(b), message


def test_int():
    assert_equal(A(a=1), A(a=1))


def test_different_type():
    assert_notequal(A(a=1), B(a=1))


def test_order():
    assert_equal(Values(value1=1, value2=2), Values(value2=2, value1=1))


def test_default():
    assert_equal(C(a=1, b=2), C(b=2))


def test_inner_eq():
    assert_equal(D(a=A(a=1)), D(a=A(a=1)))


def test_float():
    assert_equal(Float(value=1), Float(value=1))


def test_float2():
    assert_equal(Float(value=1.0), Float(value=1))


# --- Argument name


def test_name():
    """The identifier fully determines the hash code"""

    @config("test.identifier.argumentname")
    class Config0:
        a: Param[int]

    @config("test.identifier.argumentname")
    class Config1:
        b: Param[int]

    @config("test.identifier.argumentname")
    class Config3:
        a: Param[int]

    assert_notequal(Config0(a=2), Config1(b=2))
    assert_equal(Config0(a=2), Config3(a=2))


# --- Test option


def test_option():
    @config("test.identifier.option")
    class OptionConfig:
        a: Param[int]
        b: Option[int] = 1

    assert_notequal(OptionConfig(a=2), OptionConfig(a=1))
    assert_equal(OptionConfig(a=1, b=2), OptionConfig(a=1))
    assert_equal(OptionConfig(a=1, b=2), OptionConfig(a=1, b=2))


# --- Dictionnary


def test_identifier_dict():
    """Test identifiers of dictionary structures"""

    class B(Config):
        x: Param[int]

    class A(Config):
        bs: Param[Dict[str, B]]

    assert_equal(A(bs={"b1": B(x=1)}), A(bs={"b1": B(x=1)}))
    assert_equal(A(bs={"b1": B(x=1), "b2": B(x=2)}), A(bs={"b2": B(x=2), "b1": B(x=1)}))

    assert_notequal(A(bs={"b1": B(x=1)}), A(bs={"b1": B(x=2)}))
    assert_notequal(A(bs={"b1": B(x=1)}), A(bs={"b2": B(x=1)}))


# --- Ignore paths


@config()
class TypeWithPath:
    a: Param[int]
    path: Param[Path]


def test_path():
    """Path should be ignored"""
    assert_equal(TypeWithPath(a=1, path="/a/b"), TypeWithPath(a=1, path="/c/d"))
    assert_notequal(TypeWithPath(a=2, path="/a/b"), TypeWithPath(a=1, path="/c/d"))


# --- Test with added arguments


def test_pathoption():
    """Path arguments should be ignored"""

    @config("pathoption_test")
    class A_with_path:
        a: Param[int]
        path: Annotated[Path, pathgenerator("path")]

    @config("pathoption_test")
    class A_without_path:
        a: Param[int]

    assert_equal(A_with_path(a=1), A_without_path(a=1))


def test_identifier_enum():
    """Path arguments should be ignored"""
    from enum import Enum

    class EnumParam(Enum):
        FIRST = 0
        SECOND = 1

    class EnumConfig(Config):
        a: Param[EnumParam]

    assert_notequal(EnumConfig(a=EnumParam.FIRST), EnumConfig(a=EnumParam.SECOND))
    assert_equal(EnumConfig(a=EnumParam.FIRST), EnumConfig(a=EnumParam.FIRST))


def test_identifier_addnone():
    """Test the case of new parameter (with None default)"""

    class B(Config):
        x: Param[int]

    class A_with_b(Config):
        __xpmid__ = "defaultnone"
        b: Param[Optional[B]] = None

    class A(Config):
        __xpmid__ = "defaultnone"

    assert_equal(A_with_b(), A())
    assert_notequal(A_with_b(b=B(x=1)), A())


def test_defaultnew():
    """Path arguments should be ignored"""

    @param("b", type=int, default=1)
    @param(name="a", type=int)
    @config("defaultnew")
    class A_with_b:
        pass

    @param(name="a", type=int)
    @config("defaultnew")
    class A:
        pass

    assert_equal(A_with_b(a=1, b=1), A(a=1))
    assert_equal(A_with_b(a=1), A(a=1))


def test_taskconfigidentifier():
    """Test whether the embedded task arguments make the configuration different"""

    @param("a", type=int)
    @config()
    class Config:
        pass

    @param("x", type=int)
    @task()
    class Task:
        def config(self):
            return Config(a=1)

    assert_equal(
        Task(x=1).submit(run_mode=RunMode.DRY_RUN),
        Task(x=1).submit(run_mode=RunMode.DRY_RUN),
    )
    assert_notequal(
        Task(x=2).submit(run_mode=RunMode.DRY_RUN),
        Task(x=1).submit(run_mode=RunMode.DRY_RUN),
    )


def test_constant():
    """Test if constants are taken into account for signature computation"""

    @config("test.constant")
    class A1:
        version: Constant[int] = 1

    @config("test.constant")
    class A1bis:
        version: Constant[int] = 1

    assert_equal(A1(), A1bis())

    @config("test.constant")
    class A2:
        version: Constant[int] = 2

    assert_notequal(A1(), A2())


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
        NewConfig(), DerivedConfig(), "A derived configuration has another ID"
    )
    assert_equal(
        NewConfig(), OldConfig(), "Deprecated and new configuration have the same ID"
    )


def test_identifier_deprecated_attribute():
    class Values(Config):
        values: Param[List[int]] = []

        @deprecate
        def value(self, x):
            self.values = [x]

    assert_equal(Values(values=[1]), Values(value=1))


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
    assert_notequal(B(a=MetaA(x=1)), B(a=MetaA(x=2)))
    assert_equal(B(a=setmeta(MetaA(x=1), True)), B(a=setmeta(MetaA(x=2), True)))

    # As parameter
    assert_equal(C(a=MetaA(x=1)), C(a=MetaA(x=2)))
    assert_notequal(C(a=setmeta(MetaA(x=1), False)), C(a=setmeta(MetaA(x=2), False)))

    # Array with mixed
    assert_equal(
        ArrayConfig(array=[MetaA(x=1)]),
        ArrayConfig(array=[MetaA(x=1), setmeta(MetaA(x=2), True)]),
    )

    # Array with empty list
    assert_equal(ArrayConfig(array=[]), ArrayConfig(array=[setmeta(MetaA(x=2), True)]))

    # Dict with mixed
    assert_equal(
        DictConfig(params={"a": MetaA(x=1)}),
        DictConfig(params={"a": MetaA(x=1), "b": setmeta(MetaA(x=2), True)}),
    )


def test_identifier_meta_default_dict():
    class DictConfig(Config):
        params: Param[Dict[str, MetaA]] = {}

    assert_equal(
        DictConfig(params={}),
        DictConfig(params={"b": setmeta(MetaA(x=2), True)}),
    )

    # Dict with mixed
    assert_equal(
        DictConfig(params={"a": MetaA(x=1)}),
        DictConfig(params={"a": MetaA(x=1), "b": setmeta(MetaA(x=2), True)}),
    )


def test_identifier_meta_default_array():
    class ArrayConfigWithDefault(Config):
        array: Param[List[MetaA]] = []

    # Array (with default) with mixed
    assert_equal(
        ArrayConfigWithDefault(array=[MetaA(x=1)]),
        ArrayConfigWithDefault(array=[MetaA(x=1), setmeta(MetaA(x=2), True)]),
    )
    # Array (with default) with empty list
    assert_equal(
        ArrayConfigWithDefault(array=[]),
        ArrayConfigWithDefault(array=[setmeta(MetaA(x=2), True)]),
    )


def check_reload(config):
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
    check_reload(IdentifierReloadConfig(id="123"))


class IdentifierReloadTaskOutput(Task):
    id: Param[str]

    def taskoutputs(self):
        return IdentifierReloadConfig(id=self.id)


class IdentifierReloadTaskOutputDerived(Config):
    task: Param[IdentifierReloadConfig]


def test_identifier_reload_taskoutput():
    """When using a task output, the identifier should not be different"""

    # Creates the configuration
    task = IdentifierReloadTaskOutput(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskOutputDerived(task=task)
    check_reload(config)


class IdentifierReloadTask(Task):
    id: Param[str]


class IdentifierReloadTaskConfig(Config):
    x: Param[int]


class IdentifierReloadTaskDerived(Config):
    task: Param[IdentifierReloadTask]
    other: Param[IdentifierReloadTaskConfig]


def test_identifier_reload_task():
    """When using a task output, the identifier should not be different"""

    # Creates the configuration
    task = IdentifierReloadTask(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskDerived(
        task=task, other=IdentifierReloadTaskConfig(x=2)
    )
    check_reload(config)


def test_identifier_reload_meta():
    """Test identifier don't change when using meta"""
    # Creates the configuration
    task = IdentifierReloadTask(id="123").submit(run_mode=RunMode.DRY_RUN)
    config = IdentifierReloadTaskDerived(
        task=task, other=setmeta(IdentifierReloadTaskConfig(x=2), True)
    )
    check_reload(config)
