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
| `title` | string | `""` | Short description of the experiment |
| `subtitle` | string | `""` | Additional details about the experiment |
| `description` | string | `""` | Full description of the experiment |
| `paper` | string | `""` | Source paper reference for this experiment |
| `add_timestamp` | bool | `False` | Append timestamp (`YYYYMMDD-HHMM`) to experiment ID |

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
