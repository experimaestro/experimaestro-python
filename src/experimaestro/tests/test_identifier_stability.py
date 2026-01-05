# ruff: noqa: T201 - This module uses print for CLI output when run as script
# Tests for identifier stability across versions

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List

from experimaestro import (
    Param,
    Config,
    field,
    InstanceConfig,
    Task,
    LightweightTask,
    Option,
)
from experimaestro.scheduler.workspace import RunMode


# --- Basic types ---


class ConfigInt(Config):
    """Config with int parameter"""

    __xpmid__ = "test.stability.ConfigInt"
    x: Param[int]


class ConfigStr(Config):
    """Config with str parameter"""

    __xpmid__ = "test.stability.ConfigStr"
    s: Param[str]


class ConfigBool(Config):
    """Config with bool parameter"""

    __xpmid__ = "test.stability.ConfigBool"
    b: Param[bool]


class ConfigFloat(Config):
    """Config with float parameter"""

    __xpmid__ = "test.stability.ConfigFloat"
    f: Param[float]


# --- Enum ---


class MyEnum(Enum):
    """Test enum"""

    VALUE_A = 1
    VALUE_B = 2
    VALUE_C = 3


# Override module name to ensure stable identifiers across different import methods
MyEnum.__module__ = "test_identifier_stability"


class ConfigEnum(Config):
    """Config with enum parameter"""

    __xpmid__ = "test.stability.ConfigEnum"
    e: Param[MyEnum]


# --- Collections ---


class ConfigList(Config):
    """Config with list parameter"""

    __xpmid__ = "test.stability.ConfigList"
    items: Param[List[int]]


class ConfigDict(Config):
    """Config with dict parameter"""

    __xpmid__ = "test.stability.ConfigDict"
    mapping: Param[Dict[str, int]]


class ConfigNestedList(Config):
    """Config with nested list of configs"""

    __xpmid__ = "test.stability.ConfigNestedList"
    configs: Param[List[ConfigInt]]


class ConfigNestedDict(Config):
    """Config with nested dict of configs"""

    __xpmid__ = "test.stability.ConfigNestedDict"
    configs: Param[Dict[str, ConfigInt]]


# --- Nested configs ---


class ConfigNested(Config):
    """Config with nested config parameter"""

    __xpmid__ = "test.stability.ConfigNested"
    inner: Param[ConfigInt]


class ConfigMultiNested(Config):
    """Config with multiple nested configs"""

    __xpmid__ = "test.stability.ConfigMultiNested"
    a: Param[ConfigInt]
    b: Param[ConfigStr]
    c: Param[ConfigNested]


# --- Options and defaults ---


class ConfigWithOption(Config):
    """Config with option parameter"""

    __xpmid__ = "test.stability.ConfigWithOption"
    required: Param[int]
    optional: Option[int] = field(ignore_default=42)


class ConfigWithDefault(Config):
    """Config with default parameter"""

    __xpmid__ = "test.stability.ConfigWithDefault"
    x: Param[int] = field(ignore_default=10)
    y: Param[int]


# --- Tasks ---


class SimpleTask(Task):
    """Simple task"""

    __xpmid__ = "test.stability.SimpleTask"
    x: Param[int]

    def execute(self):
        pass


class TaskWithConfig(Task):
    """Task with config parameter"""

    __xpmid__ = "test.stability.TaskWithConfig"
    config: Param[ConfigInt]

    def execute(self):
        pass


class TaskWithOutput(Task):
    """Task that outputs a config"""

    __xpmid__ = "test.stability.TaskWithOutput"
    value: Param[int]

    def task_outputs(self, dep):
        return ConfigInt.C(x=self.value)

    def execute(self):
        pass


class MyLightweightTask(LightweightTask):
    """Lightweight task for init_tasks"""

    __xpmid__ = "test.stability.MyLightweightTask"
    param: Param[int]

    def execute(self):
        pass


# --- Cycles ---


class CycleA(Config):
    """Config that can reference CycleB"""

    __xpmid__ = "test.stability.CycleA"
    b: Param["CycleB"]


class CycleB(Config):
    """Config that can reference CycleA"""

    __xpmid__ = "test.stability.CycleB"
    a: Param["CycleA"]


# --- InstanceConfig ---


class SubModel(InstanceConfig):
    """InstanceConfig for testing instance identity"""

    __xpmid__ = "test.stability.SubModel"
    value: Param[int] = field(ignore_default=100)


class ModelContainer(Config):
    """Config that contains SubModel instances"""

    __xpmid__ = "test.stability.ModelContainer"
    m1: Param[SubModel]
    m2: Param[SubModel]


def get_configurations():
    """Return all test configurations with their identifiers

    Returns a dict mapping test case names to configuration objects
    """
    configs = {}

    # Basic types
    configs["int_positive"] = ConfigInt.C(x=42)
    configs["int_negative"] = ConfigInt.C(x=-10)
    configs["int_zero"] = ConfigInt.C(x=0)
    configs["str_simple"] = ConfigStr.C(s="hello")
    configs["str_empty"] = ConfigStr.C(s="")
    configs["str_unicode"] = ConfigStr.C(s="héllo wörld 🌍")
    configs["bool_true"] = ConfigBool.C(b=True)
    configs["bool_false"] = ConfigBool.C(b=False)
    configs["float_simple"] = ConfigFloat.C(f=3.14)
    configs["float_negative"] = ConfigFloat.C(f=-2.5)
    configs["float_zero"] = ConfigFloat.C(f=0.0)

    # Enum
    configs["enum_value_a"] = ConfigEnum.C(e=MyEnum.VALUE_A)
    configs["enum_value_b"] = ConfigEnum.C(e=MyEnum.VALUE_B)
    configs["enum_value_c"] = ConfigEnum.C(e=MyEnum.VALUE_C)

    # Lists
    configs["list_empty"] = ConfigList.C(items=[])
    configs["list_single"] = ConfigList.C(items=[1])
    configs["list_multiple"] = ConfigList.C(items=[1, 2, 3, 4, 5])
    configs["list_nested_empty"] = ConfigNestedList.C(configs=[])
    configs["list_nested_single"] = ConfigNestedList.C(configs=[ConfigInt.C(x=1)])
    configs["list_nested_multiple"] = ConfigNestedList.C(
        configs=[ConfigInt.C(x=1), ConfigInt.C(x=2), ConfigInt.C(x=3)]
    )

    # Dicts
    configs["dict_empty"] = ConfigDict.C(mapping={})
    configs["dict_single"] = ConfigDict.C(mapping={"a": 1})
    configs["dict_multiple"] = ConfigDict.C(mapping={"a": 1, "b": 2, "c": 3})
    configs["dict_nested_empty"] = ConfigNestedDict.C(configs={})
    configs["dict_nested_single"] = ConfigNestedDict.C(configs={"x": ConfigInt.C(x=1)})
    configs["dict_nested_multiple"] = ConfigNestedDict.C(
        configs={"a": ConfigInt.C(x=1), "b": ConfigInt.C(x=2), "c": ConfigInt.C(x=3)}
    )

    # Nested configs
    configs["nested_simple"] = ConfigNested.C(inner=ConfigInt.C(x=100))
    configs["nested_multi"] = ConfigMultiNested.C(
        a=ConfigInt.C(x=1),
        b=ConfigStr.C(s="test"),
        c=ConfigNested.C(inner=ConfigInt.C(x=2)),
    )

    # Options and defaults
    configs["option_with_default"] = ConfigWithOption.C(required=5)
    configs["option_override"] = ConfigWithOption.C(required=5, optional=100)
    configs["default_with_default"] = ConfigWithDefault.C(y=20)
    configs["default_override"] = ConfigWithDefault.C(x=99, y=20)

    # Tasks (without submission)
    configs["task_simple"] = SimpleTask.C(x=10)
    configs["task_with_config"] = TaskWithConfig.C(config=ConfigInt.C(x=5))

    # Tasks with submission (creates task outputs)
    configs["task_submitted"] = SimpleTask.C(x=15).submit(run_mode=RunMode.DRY_RUN)
    configs["task_with_output"] = TaskWithOutput.C(value=25).submit(
        run_mode=RunMode.DRY_RUN
    )

    # Task using output from another task
    task_output = TaskWithOutput.C(value=30).submit(run_mode=RunMode.DRY_RUN)
    configs["task_using_output"] = TaskWithConfig.C(config=task_output)

    # Tasks with init_tasks
    configs["task_with_init"] = SimpleTask.C(x=20).submit(
        run_mode=RunMode.DRY_RUN, init_tasks=[MyLightweightTask.C(param=1)]
    )
    configs["task_with_multiple_init"] = SimpleTask.C(x=25).submit(
        run_mode=RunMode.DRY_RUN,
        init_tasks=[MyLightweightTask.C(param=1), MyLightweightTask.C(param=2)],
    )

    # Cycles
    cycle_a = CycleA.C()
    cycle_b = CycleB.C(a=cycle_a)
    cycle_a.b = cycle_b
    configs["cycle_simple"] = cycle_a

    # InstanceConfig - test instance identity
    # Single instance used twice (shared) - backwards compatible with regular Config
    sm_single = SubModel.C(value=100)
    configs["instance_shared"] = ModelContainer.C(m1=sm_single, m2=sm_single)

    # Two separate instances with same parameters - different identifiers
    sm1 = SubModel.C(value=100)
    sm2 = SubModel.C(value=100)
    configs["instance_separate"] = ModelContainer.C(m1=sm1, m2=sm2)

    # InstanceConfig with different parameter values
    sm3 = SubModel.C(value=200)
    sm4 = SubModel.C(value=300)
    configs["instance_different_values"] = ModelContainer.C(m1=sm3, m2=sm4)

    return configs


def save_reference(output_file: Path, overwrite: bool = False):
    """Generate and save reference identifiers

    Args:
        output_file: Path to the JSON file to save
        overwrite: If True, overwrite existing file even if there are changes
    """
    configs = get_configurations()
    reference = {}

    for name, config in configs.items():
        identifier = config.__xpm__.identifier.all.hex()
        reference[name] = identifier

    # Check if file exists and compare
    if output_file.exists() and not overwrite:
        existing = load_reference(output_file)

        changes = []
        for name in sorted(set(existing.keys()) | set(reference.keys())):
            old_id = existing.get(name)
            new_id = reference.get(name)

            if old_id is None:
                changes.append(f"  + {name}: NEW")
            elif new_id is None:
                changes.append(f"  - {name}: REMOVED")
            elif old_id != new_id:
                changes.append(f"  ! {name}: CHANGED")
                changes.append(f"      Old: {old_id}")
                changes.append(f"      New: {new_id}")

        if changes:
            print(
                f"⚠️  WARNING: Reference file has {len([c for c in changes if c.startswith('  ')])} change(s):"
            )
            print("\n".join(changes))
            print("\nTo overwrite, run with --overwrite flag")
            return None

    # Save to JSON file
    with output_file.open("w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)

    print(f"✓ Saved {len(reference)} reference identifiers to {output_file}")
    return reference


def load_reference(reference_file: Path) -> Dict[str, str]:
    """Load reference identifiers from JSON file

    Args:
        reference_file: Path to the JSON file

    Returns:
        Dictionary mapping test case names to identifier hex strings
    """
    with reference_file.open("r") as f:
        return json.load(f)


def test_identifier_stability():
    """Test that identifiers are stable across experimaestro versions"""

    # Get the reference file path (same directory as this test file)
    reference_file = Path(__file__).parent / "identifier_stability.json"

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference file {reference_file} not found. Run 'python {__file__}' to generate it."
        )

    # Load reference identifiers
    reference = load_reference(reference_file)

    # Get current configurations
    configs = get_configurations()

    # Check each configuration
    mismatches = []
    for name, config in configs.items():
        current_id = config.__xpm__.identifier.all.hex()
        expected_id = reference.get(name)

        if expected_id is None:
            mismatches.append(
                f"  - {name}: NEW (not in reference file)\n    Current: {current_id}"
            )
        elif current_id != expected_id:
            mismatches.append(
                f"  - {name}: MISMATCH\n    Expected: {expected_id}\n    Current:  {current_id}"
            )

    # Check for removed configurations
    for name in reference:
        if name not in configs:
            mismatches.append(f"  - {name}: REMOVED (no longer in test suite)")

    # Report results
    if mismatches:
        error_msg = (
            f"Identifier stability test failed! {len(mismatches)} mismatch(es):\n"
            + "\n".join(mismatches)
        )
        raise AssertionError(error_msg)

    print(f"✓ All {len(configs)} identifiers are stable")


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    overwrite = "--overwrite" in sys.argv

    # Generate the reference file
    reference_file = Path(__file__).parent / "identifier_stability.json"
    reference = save_reference(reference_file, overwrite=overwrite)

    if reference is None:
        # Changes detected but not overwriting
        sys.exit(1)

    # Print summary
    print(f"\nGenerated {len(reference)} reference identifiers:")
    for name in sorted(reference.keys()):
        print(f"  - {name}")
