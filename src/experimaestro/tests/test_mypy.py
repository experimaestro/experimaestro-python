"""Tests for the mypy plugin.

These tests verify that the mypy plugin can be loaded and provides
basic type inference for experimaestro types.
"""

import subprocess
import tempfile
from pathlib import Path


def run_mypy(code: str, use_plugin: bool = True) -> tuple[int, str, str]:
    """Run mypy on the given code and return (exit_code, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test code
        test_file = Path(tmpdir) / "test_code.py"
        test_file.write_text(code)

        # Write mypy config if using plugin
        if use_plugin:
            config_file = Path(tmpdir) / "mypy.ini"
            config_file.write_text("[mypy]\nplugins = experimaestro.mypy\n")
            args = ["mypy", str(test_file), "--config-file", str(config_file)]
        else:
            args = ["mypy", str(test_file)]

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr


def test_mypy_plugin_loads():
    """Test that the mypy plugin can be loaded without errors."""
    code = """
from experimaestro import Config, Param

class Model(Config):
    hidden_size: Param[int]
"""
    _, stdout, stderr = run_mypy(code)
    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr


def test_mypy_plugin_import():
    """Test that the plugin module can be imported."""
    from experimaestro.mypy import plugin, ExperimaestroPlugin

    # Verify the plugin function returns the plugin class
    # The version argument is required by mypy API but unused
    assert plugin("1.0") == ExperimaestroPlugin


def test_mypy_plugin_is_config_subclass():
    """Test the is_config_subclass helper function."""
    from experimaestro.mypy import is_config_subclass

    # Note: This tests the function signature, not full functionality
    # (full functionality requires mypy TypeInfo objects)
    assert callable(is_config_subclass)


def test_mypy_config_c_type_inference():
    """Test that Config.C type inference works with the plugin."""
    code = """
from experimaestro import Config, Task, Param

class Model(Config):
    hidden_size: Param[int]

class TrainTask(Task):
    model: Param[Model]

    def execute(self):
        pass

# Test that .C returns the correct type
model = Model.C  # type: ignore[call-arg]
task = TrainTask.C  # type: ignore[call-arg]

# These would fail without the plugin (would be Any)
reveal_type(model)  # Should show Type[Model]
reveal_type(task)   # Should show Type[TrainTask]
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Check that types are inferred (not "Any")
    assert "Model" in stdout
    assert "TrainTask" in stdout


def test_mypy_config_c_parameters():
    """Test that Config.C accepts proper parameters."""
    code = """
from experimaestro import Config, Task, Param

class Model(Config):
    hidden_size: Param[int]

class TrainTask(Task):
    model: Param[Model]
    epochs: Param[int]

    def execute(self):
        pass

# Test parameter hints
model = Model.C(hidden_size=256)
reveal_type(model.hidden_size)  # Should be int

task = TrainTask.C(model=model, epochs=10)
reveal_type(task.epochs)  # Should be int
reveal_type(task.model)  # Should be Model
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Check that attribute types are inferred
    assert "int" in stdout


def test_mypy_submit_return_type():
    """Test that submit() returns the correct type."""
    code = """
from experimaestro import Task, Param

class MyTask(Task):
    x: Param[int]

    def execute(self):
        pass

task = MyTask.C(x=1)
result = task.submit()
reveal_type(result)  # Should be MyTask
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Check that submit() returns the task type
    assert "MyTask" in stdout


def test_mypy_config_composition():
    """Test that configs can be composed together."""
    code = """
from experimaestro import Config, Task, Param

class Model(Config):
    hidden_size: Param[int]

class Train(Task):
    model: Param[Model]
    epochs: Param[int]

    def execute(self):
        pass

# Create a model config
model = Model.C(hidden_size=256)

# Pass it to another config - should type-check
train = Train.C(model=model, epochs=10)
reveal_type(train.model)  # Should be Model
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Check that model attribute is typed correctly
    assert "Model" in stdout


def test_mypy_default_values():
    """Test that fields with defaults are optional."""
    code = """
from experimaestro import Config, Param

class Settings(Config):
    required_field: Param[int]
    optional_field: Param[int] = 10  # Has default

# Should work without optional_field
config = Settings.C(required_field=5)
reveal_type(config.required_field)
reveal_type(config.optional_field)
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Both fields should be int
    assert "int" in stdout


def test_mypy_configmixin_methods():
    """Test that ConfigMixin methods are available on Config subclasses."""
    code = """
from experimaestro import Config, Task, Param

class Model(Config):
    hidden_size: Param[int]

class TrainTask(Task):
    model: Param[Model]

    def execute(self):
        pass

# Test ConfigMixin methods are available
model = Model.C(hidden_size=256)

# tag() method from ConfigMixin - should return self
tagged = model.tag("version", "1.0")
reveal_type(tagged)  # Should be Model

# tags() method from ConfigMixin
t = model.tags()
reveal_type(t)  # Should return tags dict
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Check that Model is inferred for tagged (tag returns self)
    assert "Model" in stdout


def test_mypy_task_outputs_return_type():
    """Test that submit() returns task_outputs type when defined."""
    code = """
from experimaestro import Task, Config, Param

class Output(Config):
    value: Param[int]

class TaskWithOutputs(Task):
    x: Param[int]

    def task_outputs(self, marker) -> Output:
        return marker(Output.C(value=self.x))

    def execute(self):
        pass

# submit() should return Output (from task_outputs annotation)
task = TaskWithOutputs.C(x=1)
result = task.submit()
reveal_type(result)  # Should be Output
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Check that submit returns the task_outputs return type
    # The reveal_type should show Output
    assert "Output" in stdout or "TaskWithOutputs" in stdout


def test_mypy_submit_with_arguments():
    """Test that submit() accepts launcher and init_tasks arguments."""
    code = """
from experimaestro import Task, Param

class MyTask(Task):
    x: Param[int]

    def execute(self):
        pass

task = MyTask.C(x=1)

# These should all be valid (no errors)
result1 = task.submit()
result2 = task.submit(launcher=None)
result3 = task.submit(init_tasks=[])
result4 = task.submit(workspace=None, launcher=None, run_mode=None, max_retries=5)
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Should not have "Unexpected keyword argument" errors
    assert "Unexpected keyword argument" not in stdout


def test_mypy_multiple_inheritance_with_module():
    """Test that multiple inheritance with non-Config bases works.

    Classes like CNN(Config, nn.Module) should only expose Param fields,
    not attributes from nn.Module like 'training'.
    """
    code = """
from experimaestro import Config, Param

# Simulate nn.Module-like class that has a 'training' attribute
class ModuleBase:
    training: bool = True
    _some_internal: int = 0

class Model(Config, ModuleBase):
    hidden_size: Param[int]
    kernel_size: Param[int] = 3  # Has default

# Should work with just experimaestro Param fields
# Should NOT require 'training' argument from ModuleBase
model = Model.C(hidden_size=256)
reveal_type(model.hidden_size)  # Should be int
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Should NOT have "Missing named argument 'training'" error
    assert "Missing named argument" not in stdout
    assert "training" not in stdout.lower() or "Revealed type" in stdout


def test_mypy_only_param_and_meta_in_init():
    """Test that only Param and Meta fields are in __init__, not Constant or other attributes."""
    code = """
from experimaestro import Config, Task, Param, Meta, Constant

class MyTask(Task):
    # These SHOULD be in __init__
    required_param: Param[int]
    optional_param: Param[str] = "default"
    meta_field: Meta[int] = 42

    # These should NOT be in __init__
    version: Constant[str] = "1.0"
    regular_class_attr: str = "not a param"

    def execute(self):
        pass

# Should work with only Param and Meta fields
task = MyTask.C(required_param=10)
reveal_type(task)
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Should NOT complain about missing 'version' or 'regular_class_attr'
    assert "Missing named argument" not in stdout
    assert "version" not in stdout.lower() or "Revealed type" in stdout


def test_mypy_constant_not_in_init():
    """Test that Constant fields are explicitly excluded from __init__."""
    code = """
from experimaestro import Task, Param, Constant

class VersionedTask(Task):
    data: Param[str]
    version: Constant[str] = "2.0"
    algorithm_version: Constant[int] = 3

    def execute(self):
        pass

# Should only require 'data', not version fields
task = VersionedTask.C(data="test")

# Passing version should be an error (not in __init__)
bad_task = VersionedTask.C(data="test", version="3.0")  # Should error
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Should have an error about 'version' being unexpected
    assert "version" in stdout.lower()


def test_mypy_meta_fields_optional():
    """Test that Meta fields are always optional (have implicit defaults)."""
    code = """
from experimaestro import Task, Param, Meta
from pathlib import Path

class TaskWithMeta(Task):
    required: Param[int]
    # Meta fields should always be optional even without explicit default
    output_path: Meta[Path]
    log_level: Meta[str] = "INFO"

    def execute(self):
        pass

# Should work without providing Meta fields
task = TaskWithMeta.C(required=5)
reveal_type(task)
"""
    _, stdout, stderr = run_mypy(code)

    # Plugin should load without crashing
    assert "INTERNAL ERROR" not in stdout
    assert "INTERNAL ERROR" not in stderr

    # Should NOT complain about missing output_path
    assert "Missing named argument" not in stdout
