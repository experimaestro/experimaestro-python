"""Tests for deprecation mechanism.

Tests cover:
- Legacy @deprecate pattern (deprecated class inherits from new class)
- New @deprecate(Target) pattern with explicit target
- @deprecate(Target, replace=True) for immediate conversion
- Deprecated attributes
- fix_deprecated CLI for migrating job directories
- Task deprecation with fix_deprecated symlink creation
"""

import logging
from typing import List

from experimaestro import field, Config, Param, Task, deprecate
from experimaestro.core.identifier import IdentifierComputer
from experimaestro.scheduler.workspace import RunMode
from experimaestro.tools.jobs import fix_deprecated

from .utils import TemporaryExperiment


def assert_equal(a, b, message=""):
    """Assert two configs have the same identifier."""
    a_id = IdentifierComputer.compute(a)
    b_id = IdentifierComputer.compute(b)
    assert a_id == b_id, f"{message}: {a_id} != {b_id}"


def assert_notequal(a, b, message=""):
    """Assert two configs have different identifiers."""
    a_id = IdentifierComputer.compute(a)
    b_id = IdentifierComputer.compute(b)
    assert a_id != b_id, f"{message}: {a_id} == {b_id}"


# --- Legacy deprecation tests ---


def test_deprecated_class_legacy():
    """Test legacy @deprecate pattern where deprecated class inherits from new class."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecated.legacy.new"

    @deprecate
    class OldConfig(NewConfig):
        __xpmid__ = "test.deprecated.legacy.old"

    class DerivedConfig(NewConfig):
        __xpmid__ = "test.deprecated.legacy.derived"

    assert_notequal(
        NewConfig.C(), DerivedConfig.C(), "A derived configuration has another ID"
    )
    assert_equal(
        NewConfig.C(),
        OldConfig.C(),
        "Deprecated and new configuration have the same ID",
    )


def test_deprecated_attribute():
    """Test deprecating a parameter with a conversion method."""

    class Values(Config):
        __xpmid__ = "test.deprecated.attribute"
        values: Param[List[int]] = field(ignore_default=[])

        @deprecate
        def value(self, x):
            self.values = [x]

    assert_equal(Values.C(values=[1]), Values.C(value=1))


# --- New @deprecate(Target) pattern tests ---


def test_deprecate_with_explicit_target():
    """Test @deprecate(TargetConfig) with explicit target class."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecate.explicit.new"
        value: Param[int]

    @deprecate(NewConfig)
    class OldConfig(Config):
        __xpmid__ = "test.deprecate.explicit.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(value=self.value)

    # Creating OldConfig returns OldConfig (not NewConfig)
    old = OldConfig.C(value=42)
    assert type(old).__name__ == "OldConfig.XPMConfig"

    # But identifiers should match because __convert__ is used for ID computation
    new = NewConfig.C(value=42)
    assert_equal(old, new)


def test_deprecate_with_parameter_transformation():
    """Test __convert__ that transforms parameters."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecate.transform.new"
        values: Param[List[int]]

    @deprecate(NewConfig)
    class OldConfig(Config):
        __xpmid__ = "test.deprecate.transform.old"
        value: Param[int]

        def __convert__(self):
            # Convert single value to list
            return NewConfig.C(values=[self.value])

    # Create old config - stays as OldConfig
    old = OldConfig.C(value=42)
    assert type(old).__name__ == "OldConfig.XPMConfig"
    assert old.value == 42

    # Identifier should match the equivalent NewConfig
    new = NewConfig.C(values=[42])
    assert_equal(old, new)


def test_deprecate_chained_versions():
    """Test chained deprecation: A_v0 -> A_v1 -> A

    Each deprecated version computes its identifier via __convert__ chain.
    """

    class A(Config):
        __xpmid__ = "test.deprecate.chained.a"
        data: Param[List[str]]

    @deprecate(A)
    class A_v1(Config):
        __xpmid__ = "test.deprecate.chained.a_v1"
        items: Param[List[str]]

        def __convert__(self):
            return A.C(data=self.items)

    @deprecate(A_v1)
    class A_v0(Config):
        __xpmid__ = "test.deprecate.chained.a_v0"
        item: Param[str]

        def __convert__(self):
            return A_v1.C(items=[self.item])

    # Create configs - each stays as its own type
    v0 = A_v0.C(item="hello")
    v1 = A_v1.C(items=["hello"])
    current = A.C(data=["hello"])

    assert type(v0).__name__ == "A_v0.XPMConfig"
    assert type(v1).__name__ == "A_v1.XPMConfig"
    assert type(current).__name__ == "A.XPMConfig"

    # All should have the same identifier (computed via conversion chain)
    assert_equal(v0, v1)
    assert_equal(v1, current)
    assert_equal(v0, current)


def test_deprecate_preserves_original_identifier():
    """Test that deprecated identifier is preserved for migration (fix_deprecated)."""

    class NewStyle(Config):
        __xpmid__ = "test.deprecate.preserve.new"
        x: Param[int]

    @deprecate(NewStyle)
    class OldStyle(Config):
        __xpmid__ = "test.deprecate.preserve.old"
        x: Param[int]

        def __convert__(self):
            return NewStyle.C(x=self.x)

    xpmtype = OldStyle.__getxpmtype__()

    # The deprecated identifier should be preserved for migration
    assert xpmtype._deprecated_identifier is not None
    assert xpmtype._deprecated_identifier.name == "test.deprecate.preserve.old"

    # The target identifier should be accessible
    assert xpmtype.identifier.name == "test.deprecate.preserve.new"


# --- @deprecate(Target, replace=True) tests ---


def test_deprecate_replace_basic():
    """Test @deprecate(Target, replace=True) immediately converts to new config."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecate.replace.basic.new"
        values: Param[list[int]]

    @deprecate(NewConfig, replace=True)
    class OldConfig(Config):
        __xpmid__ = "test.deprecate.replace.basic.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(values=[self.value])

    # Creating OldConfig should return a NewConfig instance
    result = OldConfig.C(value=42)

    # The result should be the new config type
    assert type(result).__name__ == "NewConfig.XPMConfig"

    # The values should be converted
    assert result.values == [42]


def test_deprecate_replace_identifier():
    """Test that replaced configs have the same identifier as direct creation."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecate.replace.id.new"
        values: Param[list[int]]

    @deprecate(NewConfig, replace=True)
    class OldConfig(Config):
        __xpmid__ = "test.deprecate.replace.id.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(values=[self.value])

    # Create via old and new ways
    via_old = OldConfig.C(value=42)
    via_new = NewConfig.C(values=[42])

    # Both should be NewConfig instances with same identifier
    assert type(via_old).__name__ == "NewConfig.XPMConfig"
    assert type(via_new).__name__ == "NewConfig.XPMConfig"
    assert_equal(via_old, via_new)


def test_deprecate_replace_preserves_deprecated_id():
    """Test that deprecated identifier is preserved for fix_deprecated even with replace=True."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecate.replace.preserve.new"
        values: Param[list[int]]

    @deprecate(NewConfig, replace=True)
    class OldConfig(Config):
        __xpmid__ = "test.deprecate.replace.preserve.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(values=[self.value])

    xpmtype = OldConfig.__getxpmtype__()

    # The deprecated identifier should be preserved
    assert xpmtype._deprecated_identifier is not None
    assert xpmtype._deprecated_identifier.name == "test.deprecate.replace.preserve.old"

    # The current identifier should point to new config
    assert xpmtype.identifier.name == "test.deprecate.replace.preserve.new"


def test_deprecate_replace_deprecated_from_preserved():
    """Test that converted config preserves reference to original deprecated config."""

    class NewConfig(Config):
        __xpmid__ = "test.deprecate.replace.from.new"
        values: Param[list[int]]

    @deprecate(NewConfig, replace=True)
    class OldConfig(Config):
        __xpmid__ = "test.deprecate.replace.from.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(values=[self.value])

    result = OldConfig.C(value=42)

    # The converted config should have reference to original
    assert result._deprecated_from is not None
    assert type(result._deprecated_from).__name__ == "OldConfig.XPMConfig"
    assert result._deprecated_from.value == 42


# --- fix_deprecated tests ---


def test_deprecation_info_available_for_fix_deprecated():
    """Test that deprecation info is available for fix_deprecated tool.

    The fix_deprecated tool needs to know:
    1. The original (deprecated) identifier
    2. The new (target) identifier
    This test verifies this information is available via _deprecation.
    """

    class NewConfig(Config):
        __xpmid__ = "test.fixdeprecated.info.new"
        values: Param[list[int]]

    @deprecate(NewConfig)
    class OldConfig(Config):
        __xpmid__ = "test.fixdeprecated.info.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(values=[self.value])

    xpmtype = OldConfig.__getxpmtype__()

    # Verify deprecation info is accessible
    assert xpmtype._deprecation is not None
    assert (
        xpmtype._deprecation.original_identifier.name == "test.fixdeprecated.info.old"
    )
    assert xpmtype._deprecation.target == NewConfig
    assert xpmtype._deprecation.replace is False

    # The identifier should point to new config
    assert xpmtype.identifier.name == "test.fixdeprecated.info.new"


def test_deprecation_info_with_replace():
    """Test that deprecation info includes replace flag."""

    class NewConfig(Config):
        __xpmid__ = "test.fixdeprecated.replace.new"
        values: Param[list[int]]

    @deprecate(NewConfig, replace=True)
    class OldConfig(Config):
        __xpmid__ = "test.fixdeprecated.replace.old"
        value: Param[int]

        def __convert__(self):
            return NewConfig.C(values=[self.value])

    xpmtype = OldConfig.__getxpmtype__()

    # Verify replace flag is set
    assert xpmtype._deprecation is not None
    assert xpmtype._deprecation.replace is True
    assert (
        xpmtype._deprecation.original_identifier.name
        == "test.fixdeprecated.replace.old"
    )


# =============================================================================
# Task deprecation tests (with actual experiment context and fix_deprecated)
# =============================================================================


class NewConfigForTask(Config):
    """New configuration used in task deprecation tests."""

    __xpmid__ = "test.deprecated.task.config.new"


@deprecate
class DeprecatedConfigForTask(NewConfigForTask):
    """Deprecated configuration (legacy pattern)."""

    __xpmid__ = "test.deprecated.task.config.deprecated"


class OldConfigForTask(NewConfigForTask):
    """Old configuration without deprecate flag (for comparison)."""

    __xpmid__ = "test.deprecated.task.config.deprecated"


class TaskWithDeprecatedConfig(Task):
    """Task that uses a deprecated config as parameter."""

    __xpmid__ = "test.deprecated.task.with.config"
    p: Param[NewConfigForTask]

    def execute(self):
        pass


def _check_symlink_paths(task_new, task_old_path):
    """Helper to verify symlink was created correctly."""
    task_new_path = task_new.__xpm__.job.path  # type: Path

    assert task_new_path.exists(), f"New path {task_new_path} should exist"
    assert task_new_path.is_symlink(), f"New path {task_new_path} should be a symlink"
    assert task_new_path.resolve() == task_old_path


def test_task_deprecated_config_identifier():
    """Test that tasks using deprecated configs have correct identifiers."""
    with TemporaryExperiment("deprecated_config"):
        # Create tasks with new, old, and deprecated configs
        task_new = TaskWithDeprecatedConfig.C(p=NewConfigForTask.C()).submit(
            run_mode=RunMode.DRY_RUN
        )
        task_old = TaskWithDeprecatedConfig.C(p=OldConfigForTask.C()).submit(
            run_mode=RunMode.DRY_RUN
        )
        task_deprecated = TaskWithDeprecatedConfig.C(
            p=DeprecatedConfigForTask.C()
        ).submit(run_mode=RunMode.DRY_RUN)

        logging.debug("New task ID: %s", task_new.__xpm__.identifier.all.hex())
        logging.debug("Old task ID: %s", task_old.__xpm__.identifier.all.hex())
        logging.debug(
            "Deprecated task ID: %s", task_deprecated.__xpm__.identifier.all.hex()
        )

        # Old (non-deprecated) and new should have different paths
        assert task_new.stdout() != task_old.stdout(), (
            "Old and new path should be different"
        )

        # Deprecated should have same path as new (identifier matches)
        assert task_new.stdout() == task_deprecated.stdout(), (
            "Deprecated path should be the same as non deprecated"
        )


def test_task_deprecated_config_fix_deprecated():
    """Test fix_deprecated creates symlinks for tasks with deprecated configs."""
    with TemporaryExperiment("deprecated_config_fix") as xp:
        task_new = TaskWithDeprecatedConfig.C(p=NewConfigForTask.C()).submit(
            run_mode=RunMode.DRY_RUN
        )

        # Run old task (before deprecation)
        task_old = TaskWithDeprecatedConfig.C(p=OldConfigForTask.C()).submit()
        task_old.wait()
        task_old_path = task_old.stdout().parent

        # Now deprecate the old config
        OldConfigForTask.__xpmtype__.deprecate()

        # Run fix_deprecated
        fix_deprecated(xp.workspace.path, fix=True, cleanup=False)

        # Verify symlink was created
        _check_symlink_paths(task_new, task_old_path)


class NewTask(Task):
    """New task for deprecation tests."""

    __xpmid__ = "test.deprecated.task.new"
    x: Param[int]

    def execute(self):
        pass


class OldTask(NewTask):
    """Old task without deprecate flag (for comparison)."""

    __xpmid__ = "test.deprecated.task.deprecated"


@deprecate
class DeprecatedTask(NewTask):
    """Deprecated task (legacy pattern)."""

    __xpmid__ = "test.deprecated.task.deprecated"


def test_task_deprecated_identifier():
    """Test that deprecated tasks have correct identifiers."""
    with TemporaryExperiment("deprecated_task", timeout_multiplier=9):
        task_new = NewTask.C(x=1).submit(run_mode=RunMode.DRY_RUN)
        task_old = OldTask.C(x=1).submit(run_mode=RunMode.DRY_RUN)
        task_deprecated = DeprecatedTask.C(x=1).submit(run_mode=RunMode.DRY_RUN)

        # Old and new should have different paths
        assert task_new.stdout() != task_old.stdout(), (
            "Old and new path should be different"
        )

        # Deprecated should have same path as new
        assert task_new.stdout() == task_deprecated.stdout(), (
            "Deprecated path should be the same as non deprecated"
        )


def test_task_deprecated_fix_deprecated():
    """Test fix_deprecated creates symlinks for deprecated tasks."""
    with TemporaryExperiment("deprecated_task_fix", timeout_multiplier=9) as xp:
        task_new = NewTask.C(x=1).submit(run_mode=RunMode.DRY_RUN)

        # Run old task (before deprecation)
        task_old = OldTask.C(x=1).submit()
        task_old.wait()
        task_old_path = task_old.stdout().parent

        # Deprecate and fix
        OldTask.__xpmtype__.deprecate()
        fix_deprecated(xp.workspace.path, fix=True, cleanup=False)

        # Verify symlink
        _check_symlink_paths(task_new, task_old_path)


# =============================================================================
# Extended tests for new deprecation mechanism with tasks
# =============================================================================


class NewTaskWithConvert(Task):
    """New task with different parameter structure."""

    __xpmid__ = "test.deprecated.task.convert.new"
    values: Param[List[int]]

    def execute(self):
        pass


@deprecate(NewTaskWithConvert)
class OldTaskWithConvert(Task):
    """Old task with single value, deprecated to new task with list."""

    __xpmid__ = "test.deprecated.task.convert.old"
    value: Param[int]

    def __convert__(self):
        return NewTaskWithConvert.C(values=[self.value])

    def execute(self):
        pass


def test_task_deprecated_with_convert_identifier():
    """Test deprecated task with __convert__ has correct identifier."""
    with TemporaryExperiment("deprecated_task_convert"):
        # Old task should compute identifier via __convert__
        task_old = OldTaskWithConvert.C(value=42).submit(run_mode=RunMode.DRY_RUN)
        task_new = NewTaskWithConvert.C(values=[42]).submit(run_mode=RunMode.DRY_RUN)

        # Identifiers should match (computed via __convert__)
        assert task_old.stdout() == task_new.stdout(), (
            "Deprecated task should have same path as equivalent new task"
        )


@deprecate(NewTaskWithConvert, replace=True)
class ReplacedTaskWithConvert(Task):
    """Old task that gets immediately replaced with new task."""

    __xpmid__ = "test.deprecated.task.replace.old"
    value: Param[int]

    def __convert__(self):
        return NewTaskWithConvert.C(values=[self.value])

    def execute(self):
        pass


def test_task_deprecated_replace_returns_new_type():
    """Test deprecated task with replace=True returns new task type."""
    with TemporaryExperiment("deprecated_task_replace"):
        # Creating old task should return new task
        result = ReplacedTaskWithConvert.C(value=42)

        # Should be NewTaskWithConvert type
        assert type(result).__name__ == "NewTaskWithConvert.XPMConfig"
        assert result.values == [42]

        # Submit should work with the new task
        task = result.submit(run_mode=RunMode.DRY_RUN)

        # Create equivalent new task directly
        task_new = NewTaskWithConvert.C(values=[42]).submit(run_mode=RunMode.DRY_RUN)

        # Paths should match
        assert task.stdout() == task_new.stdout()


# =============================================================================
# Attribute warning tests for replaced configs
# =============================================================================


def test_deprecate_replace_warns_on_old_attribute(caplog):
    """Test that setting attributes that existed on deprecated config warns."""

    class NewConfigForWarn(Config):
        __xpmid__ = "test.deprecate.replace.warn.new"
        values: Param[list[int]]

    @deprecate(NewConfigForWarn, replace=True)
    class OldConfigForWarn(Config):
        __xpmid__ = "test.deprecate.replace.warn.old"
        value: Param[int]
        extra: Param[str] = field(
            ignore_default="default"
        )  # This doesn't exist in NewConfigForWarn

        def __convert__(self):
            return NewConfigForWarn.C(values=[self.value])

    # Creating OldConfig returns NewConfig
    result = OldConfigForWarn.C(value=42)
    assert type(result).__name__ == "NewConfigForWarn.XPMConfig"

    # Setting an attribute that only exists on OldConfigForWarn should warn
    import logging

    with caplog.at_level(logging.WARNING):
        result.extra = "new_value"

    # Check that warning was logged
    assert any("extra" in record.message for record in caplog.records)
    assert any("deprecated" in record.message.lower() for record in caplog.records)

    # The attribute should NOT be set on the new config
    assert not hasattr(result, "extra") or getattr(result, "extra", None) is None
