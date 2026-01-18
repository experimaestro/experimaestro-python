# Running experiments

The main class is {py:class}`~experimaestro.experiment` - context manager for running experiments, handling job submission and monitoring.

When using the command line interface to run experiment, the main object
of interaction is {py:class}`~experimaestro.experiments.cli.ExperimentHelper` - helper class for CLI-based experiment execution.

## Experiment services

- {py:class}`~experimaestro.scheduler.services.Service` - Base class for experiment services.
- {py:class}`~experimaestro.scheduler.services.WebService` - Web-based service with HTTP endpoint.
- {py:class}`~experimaestro.scheduler.services.ServiceListener` - Listener for service state changes.


## Experiment configuration

The module `experimaestro.experiments` contain code factorizing boilerplate for
launching experiments. It allows to setup the experimental environment and
read ``YAML`` configuration files to setup some experimental parameters.

This can be extended to support more specific experiment helpers (see e.g.
experimaestro-ir for an example).

{py:class}`~experimaestro.experiments.ConfigurationBase` should be the parent class of any configuration.

### Example

An `experiment.py` file:

```python
from experimaestro.experiments import ExperimentHelper, configuration, ConfigurationBase

@configuration
class Configuration(ConfigurationBase):
    #: Default learning rate
    learning_rate: float = 1e-3

def run(
    helper: ExperimentHelper, cfg: Configuration
):
    # Experimental code
    ...
```

With `full.yaml` located in the same folder as `experiment.py`

```yaml
    file: experiment
    learning_rate: 1e-4
```

The experiment can be started with

```sh
    experimaestro run-experiment --run-mode NORMAL full.yaml
```

See the [CLI documentation](cli.md#running-experiments) for more details

### Experiment code in a module

The Python path can be set by the configuration file, and module be used instead
of a file:

```yaml
    # Module name containing the "run" function
    module: first_stage.experiment

    # Python paths relative to the directory containing this YAML file
    # By default, the python path is based on the hypothesis that
    # the YAML file is in the same folder as the loaded python module.
    # For instance, for `first_stage.experiment`, the python path
    # would be set automatically to the parent folder `..`. For `first_stage.sub.experiment`,
    # this would be set to `../..`
    pythonpath:
        - ..
```

### YAML Configuration Reference

The YAML configuration file supports the following options from {py:class}`~experimaestro.experiments.ConfigurationBase`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | **required** | Unique identifier for the experiment |
| `file` | string | `"experiment"` | Relative path to the Python file containing the `run` function |
| `module` | string | `None` | Python module containing the `run` function (mutually exclusive with `file`) |
| `pythonpath` | list | `None` | List of paths to add to Python path (relative to YAML file directory) |
| `parent` | string | `None` | Relative path to a parent YAML file to inherit from |
| `pre_experiment` | string | `None` | Python file path or module name to execute before importing the experiment |
| `title` | string | `""` | Short description of the experiment |
| `subtitle` | string | `""` | Additional details about the experiment |
| `description` | string | `""` | Full description of the experiment |
| `paper` | string | `""` | Source paper reference for this experiment |
| `add_timestamp` | bool | `False` | Append timestamp (`YYYYMMDD-HHMM`) to experiment ID |
| `dirty_git` | string | `"warn"` | Action when git has uncommitted changes: `ignore`, `warn`, or `error` |

#### Configuration inheritance

YAML files can inherit from parent configurations using the `parent` option:

```yaml
# base.yaml
id: base-experiment
learning_rate: 1e-3
batch_size: 32
```

```yaml
# experiment.yaml
parent: base.yaml
id: my-experiment
learning_rate: 1e-4  # Override parent value
```

#### Multiple YAML files

You can also merge multiple YAML files using CLI options:

```bash
# Pre-yaml files are loaded first, then main file, then post-yaml files
experimaestro run-experiment --pre-yaml defaults.yaml --post-yaml overrides.yaml main.yaml
```

#### Inline configuration overrides

Override specific values from the command line using `-c` with [OmegaConf dotlist syntax](https://omegaconf.readthedocs.io/):

```bash
# Simple values
experimaestro run-experiment -c learning_rate=1e-5 -c batch_size=64 experiment.yaml

# Nested values using dot notation
experimaestro run-experiment -c model.hidden_size=512 -c model.num_layers=6 experiment.yaml

# List items by index
experimaestro run-experiment -c data.transforms.0.name=resize experiment.yaml
```

#### Previewing the merged configuration

Use `--show` to output the final merged configuration as JSON without running the experiment. This is useful for debugging configuration inheritance and overrides:

```bash
experimaestro run-experiment --show experiment.yaml

# With overrides - see the final result
experimaestro run-experiment --show -c learning_rate=1e-5 --pre-yaml base.yaml experiment.yaml
```

### Pre-experiment Setup

The `pre_experiment` option allows you to run Python code **before** the experiment module is imported. It can be specified as:

- **A file path**: Relative path to a Python file (e.g., `pre_setup.py`)
- **A module name**: Python module to import (e.g., `mypackage.pre_experiment`)

This is useful for:

- Setting environment variables to control library behavior
- Mocking heavy modules to speed up the experiment setup phase (the actual job execution will use real modules)
- Configuring logging or other global state

#### Example: Speed up PyTorch imports

PyTorch's `torch.compile` and module initialization can significantly slow down experiment startup. You can use `pre_experiment` to mock these components:

Create a `pre_setup.py` file:
```python
import os
import sys
from unittest.mock import MagicMock

# Reduce torch compile threads
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# Make torch.compile a no-op decorator
class MockCompile:
    def __call__(self, fn=None, **kwargs):
        if fn is not None:
            return fn
        def decorator(func):
            return func
        return decorator

# Mock torch module with no-op compile
class TorchMock(MagicMock):
    compile = MockCompile()

sys.modules['torch'] = TorchMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
```

Then reference it in your YAML:
```yaml
id: my-experiment
pre_experiment: pre_setup.py
file: experiment
```

#### Example: Set environment variables

For simpler use cases like setting environment variables:

```python
# pre_env.py
import os

# Control threading behavior
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

```yaml
id: my-experiment
pre_experiment: pre_env.py
file: experiment
```

#### Example: Mock heavy modules with mock_modules

For experiments that import heavy libraries like PyTorch or transformers, you can use {py:func}`~experimaestro.experiments.mock_modules` to mock these modules during the experiment setup phase. This significantly speeds up configuration parsing while the actual job execution still uses the real modules.

```python
# pre_experiment.py
from experimaestro.experiments import mock_modules

# Mock PyTorch and related modules
mock_modules(
    # Modules to mock (submodules are automatically included)
    ['torch', 'pytorch_lightning', 'transformers', 'huggingface_hub'],
    # Decorators to make no-ops
    decorators=[
        'torch.compile',
        'torch.jit.script',
        'torch.jit.unused',
        'torch.jit.export',
        'torch.jit.ignore',
        'torch.no_grad',
        'torch.inference_mode',
    ]
)
```

```yaml
id: my-experiment
pre_experiment: pre_experiment.py
file: experiment
```

The `mock_modules` function provides:

- **Module mocking**: Any import of the specified modules returns fake objects that silently accept attribute access, method calls, and instantiation
- **Decorator handling**: Specified decorator paths (like `torch.compile`) work as no-op decorators that return the function unchanged
- **Inheritance support**: Code that inherits from mocked classes (like `torch.nn.Module`) works correctly without metaclass conflicts
- **Generic type support**: Subscript notation like `List[int]` or `Module[str, Tensor]` works correctly

This is particularly useful for large codebases with many PyTorch modules where importing takes significant time during experiment configuration.

#### Example: Using a module name

If you have a package with a pre-experiment module, you can reference it by module name:

```yaml
id: my-experiment
pre_experiment: mypackage.pre_experiment
module: mypackage.experiment
```

This is useful when:
- Your pre-experiment code is part of an installed package
- You want to share pre-experiment setup across multiple experiments
- You're using a library that provides pre-experiment utilities (like `experimaestro.experiments.FakeModuleFinder`)

### Dirty Git Check

Experimaestro can check whether your project's git repository has uncommitted changes when starting an experiment. This helps ensure reproducibility by warning you (or preventing you) from running experiments with uncommitted code changes.

The `dirty_git` option controls the behavior:

| Value | Description |
|-------|-------------|
| `ignore` | Don't check or warn about uncommitted changes |
| `warn` | Log a warning if there are uncommitted changes (default) |
| `error` | Raise a `DirtyGitError` exception and abort the experiment |

#### YAML configuration

```yaml
id: my-experiment
dirty_git: error  # Fail if git is dirty
```

#### Python API

When using the experiment context manager directly, you can pass the `dirty_git` parameter:

```python
from experimaestro import experiment, DirtyGitAction

# Using the enum
with experiment(workdir, "my-experiment", dirty_git=DirtyGitAction.ERROR) as xp:
    ...

# Or using the string value
with experiment(workdir, "my-experiment", dirty_git=DirtyGitAction.WARN) as xp:
    ...
```

#### Handling DirtyGitError

When `dirty_git` is set to `error`, a {py:class}`~experimaestro.DirtyGitError` exception is raised if the repository has uncommitted changes:

```python
from experimaestro import experiment, DirtyGitAction, DirtyGitError

try:
    with experiment(workdir, "my-experiment", dirty_git=DirtyGitAction.ERROR) as xp:
        ...
except DirtyGitError as e:
    print(f"Cannot run experiment: {e}")
```

### Common handling

See {py:class}`~experimaestro.experiments.cli.ExperimentHelper` for the CLI helper class.

## Experiment metadata

### Hostname tracking

Experimaestro automatically records the hostname where each experiment run is launched. This information is useful for identifying which machine was used when running experiments across multiple hosts.

The hostname is:
- Recorded when a new experiment run starts
- Stored in both the workspace database and on disk (in `xp/{experiment_id}/informations.json`)
- Displayed in the experiments list in both CLI and TUI
- Preserved during database resync operations

To view the hostname for experiments:

```bash
# CLI - shows hostname in brackets
experimaestro experiments list --workdir /path/to/workspace
# Output: my-experiment [hostname.local] (5/10 jobs)

# TUI - hostname shown in "Host" column
experimaestro experiments monitor --console --workdir /path/to/workspace
```
