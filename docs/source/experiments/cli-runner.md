# The CLI experiment runner

The `experimaestro.experiments` module factorizes the boilerplate of launching
an experiment: your code provides a `run(helper, spec)` entry point, and the
experimental parameters come from `YAML` files: the experiment specification.

The experiment is then started with

```sh
experimaestro run full.yaml
```

It exits with code 1 if jobs failed (their log files are listed), so that
scripts and CI jobs can tell; a failed job does not stop the others. Within
Python, the {py:class}`~experimaestro.experiment` context does not raise in
that case: check `xp.failed` after `xp.wait()`.

See the [CLI documentation](../cli.md#running-experiments) for all the options,
and {py:class}`~experimaestro.experiments.cli.ExperimentHelper` for the helper
passed to `run`. This mechanism can be extended to support more specific helpers
(see e.g. experimaestro-ir).

## The entry point

The specification class must derive from
{py:class}`~experimaestro.experiments.ExperimentSpecification`:

```python
# experiment.py
from experimaestro.experiments import (
    ExperimentHelper,
    specification,
    ExperimentSpecification,
)


@specification
class Specification(ExperimentSpecification):
    #: Default learning rate
    learning_rate: float = 1e-3


def run(helper: ExperimentHelper, spec: Specification):
    # Experimental code
    ...
```

with `full.yaml` located in the same folder:

```yaml
id: my-experiment
file: experiment
learning_rate: 1e-4
```

### Using a module

The entry point can also be a module, in which case the Python path can be set
by the YAML file:

```yaml
# Module name containing the "run" function
module: first_stage.experiment

# Python paths relative to the directory containing this YAML file.
# By default, the python path assumes that the YAML file is in the same folder
# as the loaded module: for `first_stage.experiment` it is set to `..`, and for
# `first_stage.sub.experiment` to `../..`.
pythonpath:
    - ..
```

## YAML reference

The options below come from
{py:class}`~experimaestro.experiments.ExperimentSpecification`; any other key is
passed to your own specification class.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `id` | string | **required** | Unique identifier for the experiment |
| `file` | string | `"experiment"` | Relative path to the Python file containing the `run` function |
| `module` | string | `None` | Python module containing the `run` function (mutually exclusive with `file`) |
| `pythonpath` | list | `None` | List of paths to add to Python path (relative to YAML file directory) |
| `imports` | list | `None` | List of YAML file paths to import and merge (current file takes priority) |
| `parent` | string | `None` | *(Deprecated, use `imports`)* Relative path to a parent YAML file to inherit from |
| `pre_experiment` | string | `None` | Python file path or module name to execute before importing the experiment |
| `title` | string | `""` | Short description of the experiment |
| `subtitle` | string | `""` | Additional details about the experiment |
| `description` | string | `""` | Full description of the experiment |
| `paper` | string | `""` | Source paper reference for this experiment |
| `add_timestamp` | bool | `False` | Append timestamp (`YYYYMMDD-HHMM`) to experiment ID |
| `dirty_git` | string | `"warn"` | Action when git has uncommitted changes: `ignore`, `warn`, or `error` |
| `tags` | list | `None` | Tags of the experiment (e.g. `[eval, mice]`), to group and search experiments in the [monitoring interfaces](../interfaces.md), and to select them on the command line (`experiments list --group`, [`carbon summary --group`](carbon.md)) |

## Composing specifications

Specifications are merged from several sources. In increasing order of
priority: the files listed in `imports`, the `--pre-yaml` files, the main file,
the `--post-yaml` files, and the `-c` overrides.

```yaml
# experiment.yaml — imported files are merged in order
imports:
  - base.yaml
  - optimizer.yaml
id: my-experiment
learning_rate: 1e-4  # Overrides the value from base.yaml
```

```bash
# Additional files on the command line
experimaestro run --pre-yaml defaults.yaml --post-yaml overrides.yaml main.yaml

# Individual values, using the OmegaConf dotlist syntax: plain values,
# nested values, and list items by index
experimaestro run -c learning_rate=1e-5 -c model.hidden_size=512 \
    -c data.transforms.0.name=resize experiment.yaml
```

Use `--show` to print the merged specification as JSON instead of running the
experiment — the quickest way to debug an inheritance or override problem:

```bash
experimaestro run --show -c learning_rate=1e-5 --pre-yaml base.yaml experiment.yaml
```

## Specifications from configurations

Rather than copying the parameters of a {py:class}`~experimaestro.Config` into
your specification, use {py:class}`~experimaestro.experiments.ConfigSpec`:

```python
from experimaestro.experiments import (
    ConfigSpec,
    ExperimentSpecification,
    spec_field,
    specification,
)

from .model import Encoder, Model


@specification
class Specification(ExperimentSpecification):
    model: ConfigSpec[Model]
    encoder: ConfigSpec[Encoder] = spec_field(defaults={"size": 128}, grid=["size"])


def run(helper, spec: Specification):
    encoder = spec.encoder.to_config()
    model = spec.model.to_config(encoder=encoder)
```

```yaml
id: my-experiment
model:
  hidden_dim: 256
  dropout: 0.2
encoder:
  size: [64, 128]
```

`ConfigSpec[Model]` has one field per parameter of `Model` whose value can be
read from YAML (numbers, strings, booleans, paths, enums, and lists, sets,
tuples and dictionaries of those), with the same type, documentation and
default (required parameters default to `MISSING`). Parameters whose value is
or contains a configuration, constants and generated values (e.g. paths created
by a `PathGenerator`) are left out: give them to `to_config(**extra)`, which
builds `Model.C(...)` from the specification values, `extra` replacing them.
Values go through the configuration's own validation, so the identifier is the
same as when building the configuration by hand. A `ConfigSpec` field defaults
to the specification with all its default values.

The fields can be changed with these options:

| Option | Default | Description |
|--------|---------|-------------|
| `meta` | `False` | Also include the `Meta` parameters |
| `include` | `None` | Only include these parameters |
| `exclude` | `None` | Do not include these parameters |
| `defaults` | `None` | Default values of parameters (`{"lr": 1e-4}`) |
| `grid` | `None` | Parameters accepting a grid search (their type is `GridSearch[T]`) |

They are given either to `spec_field(...)` (other keyword arguments go to
`attr.field`), or as class keyword arguments of a subclass, which can also
declare fields (replacing the generated ones, or new fields that are not given
to the configuration) and methods:

```python
class ModelSpec(ConfigSpec[Model], meta=True, exclude=["dropout"]):
    hidden_dim: GridSearch[int] = 128


@specification
class Specification(ExperimentSpecification):
    model: ModelSpec
    small_model: ModelSpec = spec_field(defaults={"hidden_dim": 16})
```

Options are inherited by subclasses, which can add fields but not remove them
(`exclude` or `include` of an inherited field is an error). Grid searches must
be expanded with `generate_grid` before calling `to_config`.

## Pre-experiment setup

`pre_experiment` runs Python code **before** the experiment module is imported,
either from a file path (`pre_experiment.py`) or from a module name
(`mypackage.pre_experiment`, useful when the setup is shared by several
experiments). Use it to set environment variables, configure logging, or mock
heavy modules.

Mocking is worth a special mention: with
{py:func}`~experimaestro.experiments.mock_modules`, imports of the given
modules return fake objects during the configuration phase, which is much
faster for code bases importing PyTorch and friends. The jobs themselves are
executed in their own process and use the real modules.

```python
# pre_experiment.py
import os
from experimaestro.experiments import mock_modules

os.environ["OMP_NUM_THREADS"] = "4"

# Submodules are automatically included
mock_modules(["torch", "pytorch_lightning", "transformers", "huggingface_hub"])
```

```yaml
id: my-experiment
pre_experiment: pre_experiment.py
file: experiment
```

Mocked objects accept attribute access, calls and instantiation; they work as
decorators (`@torch.compile`, `@torch.no_grad()`), as base classes
(`torch.nn.Module`), and support subscript notation (`Tensor[int]`).

## Dirty git check

Experimaestro can check whether your project's git repository has uncommitted
changes when an experiment starts, so that a result can always be traced back
to a commit:

| Value | Description |
|-------|-------------|
| `ignore` | Don't check or warn about uncommitted changes |
| `warn` | Log a warning if there are uncommitted changes (default) |
| `error` | Raise a {py:class}`~experimaestro.DirtyGitError` and abort the experiment |

```yaml
id: my-experiment
dirty_git: error  # Fail if git is dirty
```

The same check is available from the Python API, and raises
{py:class}`~experimaestro.DirtyGitError` when set to `error`:

```python
from experimaestro import experiment, DirtyGitAction, DirtyGitError

try:
    with experiment(workdir, "my-experiment", dirty_git=DirtyGitAction.ERROR) as xp:
        ...
except DirtyGitError as e:
    print(f"Cannot run experiment: {e}")
```
