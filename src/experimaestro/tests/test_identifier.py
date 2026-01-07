# Tests for identifier computation

import json
from pathlib import Path
from typing import Dict, List, Optional
from experimaestro import (
    Param,
    Config,
    InstanceConfig,
    Constant,
    Meta,
    Option,
    PathGenerator,
    field,
    Task,
    LightweightTask,
    partial,
    param_group,
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
    a: Param[int] = field(ignore_default=1)
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
        b: Option[int] = field(ignore_default=1)

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
        b: Param[int] = field(ignore_default=1)

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
        params: Param[Dict[str, MetaA]] = field(ignore_default={})

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
        array: Param[List[MetaA]] = field(ignore_default=[])

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


# --- Test InstanceConfig


class SubModel(InstanceConfig):
    """Test InstanceConfig - instances are distinguished even with same params"""

    pass


class SubModelAsConfig(Config):
    """Same as SubModel but as regular Config for backwards compat testing"""

    __xpmid__ = "test.SubModel"
    pass


class Model(Config):
    """Model that can contain SubModel instances"""

    m1: Param[SubModel]
    m2: Param[SubModel]


class ModelWithRegularConfig(Config):
    """Model using regular Config instead of InstanceConfig"""

    __xpmid__ = "test.Model"
    m1: Param[SubModelAsConfig]
    m2: Param[SubModelAsConfig]


def test_instanceconfig_backwards_compat():
    """Model using single InstanceConfig should have same ID as with regular Config"""
    # Using InstanceConfig (first occurrence only, no instance marker added)
    sm1 = SubModel.C()
    sm1.__xpmtype__.identifier.name = "test.SubModel"  # Match the __xpmid__
    m_instance = Model.C(m1=sm1, m2=sm1)
    m_instance.__xpmtype__.identifier.name = "test.Model"

    # Using regular Config
    sc1 = SubModelAsConfig.C()
    m_regular = ModelWithRegularConfig.C(m1=sc1, m2=sc1)

    # Should have same identifier (backwards compatible)
    assert_equal(
        m_instance, m_regular, "Single InstanceConfig should be backwards compatible"
    )


def test_instanceconfig_same_params_different_instances():
    """Model with separate InstanceConfig instances should differ from shared"""
    sm1 = SubModel.C()
    sm2 = SubModel.C()

    # Using the same instance twice (shared)
    m1 = Model.C(m1=sm1, m2=sm1)

    # Using different instances (separate)
    m2 = Model.C(m1=sm1, m2=sm2)

    # These should be different because sm2 is a second instance with same params
    assert_notequal(m1, m2, "Models with shared vs separate instances should differ")


def test_instanceconfig_reused_instance():
    """Reusing the same InstanceConfig instance should give same ID"""
    sm1 = SubModel.C()

    # Using the same instance object multiple times should be OK
    m1 = Model.C(m1=sm1, m2=sm1)
    m2 = Model.C(m1=sm1, m2=sm1)

    # These should be the same because we're reusing the exact same objects
    assert_equal(m1, m2, "Models with same instance objects should be equal")


def test_instanceconfig_serialization():
    """InstanceConfig identifiers should be stable after serialization"""
    sm1 = SubModel.C()
    sm2 = SubModel.C()

    # Create a model with two different instances
    m1 = Model.C(m1=sm1, m2=sm2)
    original_id = getidentifier(m1)

    # Serialize and reload
    check_reload(m1)

    # The identifier should remain the same
    assert getidentifier(m1) == original_id


# --- Test ignore_default vs default in field() ---


def test_identifier_field_ignore_default():
    """Test that field(ignore_default=X) ignores value in identifier when value == X"""

    class ConfigWithIgnoreDefault(Config):
        __xpmid__ = "test.identifier.field_ignore_default"
        a: Param[int] = field(ignore_default=1)
        b: Param[int]

    # When a=1 (matches ignore_default), should be same as not specifying a
    class ConfigWithoutA(Config):
        __xpmid__ = "test.identifier.field_ignore_default"
        b: Param[int]

    assert_equal(
        ConfigWithIgnoreDefault.C(a=1, b=2),
        ConfigWithIgnoreDefault.C(b=2),
        "field(ignore_default=1) should ignore a=1 in identifier",
    )
    assert_equal(
        ConfigWithIgnoreDefault.C(a=1, b=2),
        ConfigWithoutA.C(b=2),
        "Config with ignore_default should match config without that param",
    )

    # When a=2 (doesn't match ignore_default), should be included
    assert_notequal(
        ConfigWithIgnoreDefault.C(a=2, b=2),
        ConfigWithIgnoreDefault.C(b=2),
        "field(ignore_default=1) should include a=2 in identifier",
    )


def test_identifier_field_default():
    """Test that field(default=X) includes value in identifier even when value == X"""

    class ConfigWithDefault(Config):
        __xpmid__ = "test.identifier.field_default"
        a: Param[int] = field(default=1)
        b: Param[int]

    class ConfigWithoutA(Config):
        __xpmid__ = "test.identifier.field_default"
        b: Param[int]

    # When a=1 (matches default), should still be included in identifier
    # so Config with a=1 should differ from Config without a
    assert_notequal(
        ConfigWithDefault.C(a=1, b=2),
        ConfigWithoutA.C(b=2),
        "field(default=1) should include a=1 in identifier",
    )

    # But two configs with same a=1 should be equal
    assert_equal(
        ConfigWithDefault.C(a=1, b=2),
        ConfigWithDefault.C(a=1, b=2),
        "Same values should have same identifier",
    )


def test_identifier_field_default_vs_ignore_default():
    """Test difference between field(default=X) and field(ignore_default=X)"""

    class ConfigWithDefault(Config):
        __xpmid__ = "test.identifier.field_default_vs_ignore"
        a: Param[int] = field(default=1)
        b: Param[int]

    class ConfigWithIgnoreDefault(Config):
        __xpmid__ = "test.identifier.field_default_vs_ignore"
        a: Param[int] = field(ignore_default=1)
        b: Param[int]

    # Both with a=1, b=2 - should differ because one includes a, other doesn't
    assert_notequal(
        ConfigWithDefault.C(a=1, b=2),
        ConfigWithIgnoreDefault.C(a=1, b=2),
        "field(default=1) vs field(ignore_default=1) should differ when a=1",
    )

    # Both with a=2 (not matching default), should be the same
    assert_equal(
        ConfigWithDefault.C(a=2, b=2),
        ConfigWithIgnoreDefault.C(a=2, b=2),
        "field(default=1) vs field(ignore_default=1) should be same when a!=1",
    )


# --- Test partial identifiers (partial) ---


# Define parameter groups at module level
iter_group = param_group("iter")
model_group = param_group("model")


def get_partial_identifier(config, sp):
    """Helper to get partial identifier for a config and partial"""
    return config.__xpm__.get_partial_identifier(sp).all


def test_partial_identifier_excludes_grouped_params():
    """Test that partial identifier excludes parameters in excluded groups"""

    class ConfigWithGroups(Config):
        checkpoints = partial(exclude_groups=[iter_group])
        max_iter: Param[int] = field(groups=[iter_group])
        learning_rate: Param[float]

    c1 = ConfigWithGroups.C(max_iter=100, learning_rate=0.1)
    c2 = ConfigWithGroups.C(max_iter=200, learning_rate=0.1)

    # Full identifiers should differ (max_iter is different)
    assert_notequal(c1, c2, "Full identifiers should differ when max_iter differs")

    # Partial identifiers should be the same (max_iter is excluded)
    pid1 = get_partial_identifier(c1, ConfigWithGroups.checkpoints)
    pid2 = get_partial_identifier(c2, ConfigWithGroups.checkpoints)
    assert pid1 == pid2, (
        "Partial identifiers should match when only excluded params differ"
    )


def test_partial_identifier_includes_ungrouped_params():
    """Test that partial identifier includes parameters not in excluded groups"""

    class ConfigWithGroups(Config):
        checkpoints = partial(exclude_groups=[iter_group])
        max_iter: Param[int] = field(groups=[iter_group])
        learning_rate: Param[float]

    c1 = ConfigWithGroups.C(max_iter=100, learning_rate=0.1)
    c2 = ConfigWithGroups.C(max_iter=100, learning_rate=0.2)

    # Partial identifiers should differ (learning_rate is not excluded)
    pid1 = get_partial_identifier(c1, ConfigWithGroups.checkpoints)
    pid2 = get_partial_identifier(c2, ConfigWithGroups.checkpoints)
    assert pid1 != pid2, (
        "Partial identifiers should differ when non-excluded params differ"
    )


def test_partial_identifier_matches_config_without_excluded():
    """Test that partial identifier matches config without the excluded fields"""

    class ConfigWithIter(Config):
        __xpmid__ = "test.partial_identifier.config"
        checkpoints = partial(exclude_groups=[iter_group])
        max_iter: Param[int] = field(groups=[iter_group])
        learning_rate: Param[float]

    class ConfigWithoutIter(Config):
        __xpmid__ = "test.partial_identifier.config"
        learning_rate: Param[float]

    c_with = ConfigWithIter.C(max_iter=100, learning_rate=0.1)
    c_without = ConfigWithoutIter.C(learning_rate=0.1)

    # The partial identifier of c_with should match full identifier of c_without
    pid = get_partial_identifier(c_with, ConfigWithIter.checkpoints)
    full_id = getidentifier(c_without)
    assert pid == full_id, (
        "Partial identifier should match config without excluded fields"
    )


def test_partial_identifier_multiple_groups():
    """Test partial identifier with parameter in multiple groups"""

    class ConfigMultiGroup(Config):
        checkpoints = partial(exclude_groups=[iter_group])
        # This parameter is in both groups - should be excluded if any group is excluded
        x: Param[int] = field(groups=[iter_group, model_group])
        y: Param[float]

    c1 = ConfigMultiGroup.C(x=1, y=0.1)
    c2 = ConfigMultiGroup.C(x=2, y=0.1)

    # Partial identifiers should be the same (x is in iter_group which is excluded)
    pid1 = get_partial_identifier(c1, ConfigMultiGroup.checkpoints)
    pid2 = get_partial_identifier(c2, ConfigMultiGroup.checkpoints)
    assert pid1 == pid2, (
        "Partial identifiers should match when param is in any excluded group"
    )


def test_partial_identifier_include_overrides_exclude():
    """Test that include_groups overrides exclude_groups"""

    class ConfigIncludeOverride(Config):
        # iter_group is excluded but also included, so it should NOT be excluded
        partial = partial(
            exclude_groups=[iter_group, model_group], include_groups=[iter_group]
        )
        x: Param[int] = field(groups=[iter_group])
        y: Param[int] = field(groups=[model_group])
        z: Param[float]

    c1 = ConfigIncludeOverride.C(x=1, y=1, z=0.1)
    c2 = ConfigIncludeOverride.C(x=2, y=1, z=0.1)
    c3 = ConfigIncludeOverride.C(x=1, y=2, z=0.1)

    # x is in iter_group which is included (overrides exclusion)
    # so different x should give different partial identifiers
    pid1 = get_partial_identifier(c1, ConfigIncludeOverride.partial)
    pid2 = get_partial_identifier(c2, ConfigIncludeOverride.partial)
    assert pid1 != pid2, "Include should override exclude - x should be included"

    # y is in model_group which is excluded (not included)
    # so different y should give SAME partial identifiers
    pid3 = get_partial_identifier(c3, ConfigIncludeOverride.partial)
    assert pid1 == pid3, "y is excluded - different y should give same partial ID"


def test_partial_identifier_exclude_all():
    """Test exclude_all option"""

    class ConfigExcludeAll(Config):
        # Exclude all, but include model_group
        partial = partial(exclude_all=True, include_groups=[model_group])
        x: Param[int] = field(groups=[iter_group])
        y: Param[int] = field(groups=[model_group])
        z: Param[float]  # No group

    c1 = ConfigExcludeAll.C(x=1, y=1, z=0.1)
    c2 = ConfigExcludeAll.C(x=2, y=1, z=0.1)  # Different x (excluded)
    c3 = ConfigExcludeAll.C(x=1, y=2, z=0.1)  # Different y (included)
    c4 = ConfigExcludeAll.C(x=1, y=1, z=0.2)  # Different z (excluded - no group)

    pid1 = get_partial_identifier(c1, ConfigExcludeAll.partial)
    pid2 = get_partial_identifier(c2, ConfigExcludeAll.partial)
    pid3 = get_partial_identifier(c3, ConfigExcludeAll.partial)
    pid4 = get_partial_identifier(c4, ConfigExcludeAll.partial)

    # x is excluded (in iter_group, not included) - same partial ID
    assert pid1 == pid2, "x is excluded - should have same partial ID"

    # y is included (in model_group) - different partial ID
    assert pid1 != pid3, "y is included - should have different partial ID"

    # z is excluded (no group, exclude_all=True) - same partial ID
    assert pid1 == pid4, (
        "z (no group) is excluded by exclude_all - should have same partial ID"
    )


def test_partial_identifier_exclude_no_group():
    """Test exclude_no_group option"""

    class ConfigExcludeNoGroup(Config):
        partial = partial(exclude_no_group=True)
        x: Param[int] = field(groups=[iter_group])
        y: Param[float]  # No group

    c1 = ConfigExcludeNoGroup.C(x=1, y=0.1)
    c2 = ConfigExcludeNoGroup.C(x=2, y=0.1)  # Different x (has group - not excluded)
    c3 = ConfigExcludeNoGroup.C(x=1, y=0.2)  # Different y (no group - excluded)

    pid1 = get_partial_identifier(c1, ConfigExcludeNoGroup.partial)
    pid2 = get_partial_identifier(c2, ConfigExcludeNoGroup.partial)
    pid3 = get_partial_identifier(c3, ConfigExcludeNoGroup.partial)

    # x has a group, so it's NOT excluded by exclude_no_group
    assert pid1 != pid2, "x has group - should have different partial ID"

    # y has no group, so it IS excluded by exclude_no_group
    assert pid1 == pid3, "y has no group - should have same partial ID"
