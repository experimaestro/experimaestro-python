---
name: experimaestro
description: Experimaestro experiment manager best practices and conventions. Use when working with experimaestro tasks, configurations, experiments, launchers, or SLURM job scheduling. Helps write correct Config/Task classes, set up experiments, configure launchers, and follow framework patterns.
---

# Experimaestro

Official experimaestro skill for writing experiment code with best practices and correct patterns.

Experimaestro is a Python experiment manager and workflow orchestrator. It manages task automation, dependencies, resource allocation, and reproducibility through a type system based on `Config` and `Task` classes.

**Documentation**: https://experimaestro-python.readthedocs.io

## Python 3.10+ Syntax

Always use modern Python syntax:

- `dict[str, int]`, `list[Path]`, `tuple[int, ...]` instead of `Dict`, `List`, `Tuple`
- `X | Y` instead of `Union[X, Y]`; `X | None` instead of `Optional[X]`

## Defining Configurations

Configurations inherit from `Config` and use `Param[T]` for parameters:

```python
from experimaestro import Config, Param, field

class MyModel(Config):
    """A model configuration.

    Attributes:
        hidden_size: Size of hidden layers
        num_layers: Number of transformer layers
    """
    hidden_size: Param[int]
    num_layers: Param[int] = field(default=6, ignore_default=True)
```

### Supported types

- Primitives: `str`, `int`, `float`, `bool`, `Path`
- Collections: `list[T]`, `set[T]`, `dict[U, V]`
- Enumerations: `enum.Enum` subclasses
- Other `Config` subclasses (nested configurations)

### Parameter defaults with `field()`

Always use `field()` for defaults:

```python
from experimaestro import Config, Param, field

class MyConfig(Config):
    # Default always included in identifier computation
    x: Param[int] = field(default=1)

    # Default ignored in identifier when value == default
    # Use this when adding new parameters to existing configs
    y: Param[int] = field(default=1, ignore_default=True)

    # Factory default
    z: Param[SomeConfig] = field(default_factory=SomeConfig.C)
```

**When to use which:**

| Syntax | Identifier includes default? | Use case |
|--------|------------------------------|----------|
| `field(default=X)` | Yes | Default is part of the task signature |
| `field(default=X, ignore_default=True)` | No (when value==X) | Adding new params without breaking old identifiers |

### Metadata (not part of identifier)

Use `Meta[T]` for parameters ignored during identifier computation:

```python
from experimaestro import Config, Meta

class MyConfig(Config):
    name: Meta[str]  # Not included in identifier
```

### Path generators

For output paths relative to the task directory:

```python
from experimaestro import Task, Meta, PathGenerator, field
from pathlib import Path

class MyTask(Task):
    output: Meta[Path] = field(default_factory=PathGenerator("output.txt"))
```

### Constants

Use `Constant[T]` for values that signal behavior changes:

```python
from experimaestro import Config, Constant

class MyConfig(Config):
    version: Constant[str] = "2.1"
```

### Sets

Sets are supported. Order does not matter for identifiers.

For `Config` objects in sets, elements must be sealed first:

```python
from experimaestro import Config, Param, sealed_set

class Model(Config):
    lr: Param[float]

class Ensemble(Config):
    models: Param[set[Model]]

m1 = Model.C(lr=0.01)
m2 = Model.C(lr=0.02)
ensemble = Ensemble.C(models=sealed_set(m1, m2))
```

### Composition operator (`@`)

The `@` operator composes configurations when the target parameter type is unambiguous:

```python
class Inner(Config):
    x: Param[int]

class Outer(Config):
    inner: Param[Inner]

# These are equivalent:
outer1 = Outer.C(inner=Inner.C(x=42))
outer2 = Outer.C() @ Inner.C(x=42)
```

### Validation

Use `__validate__` for parameter validation before submission:

```python
class ModelLearn(Config):
    batch_size: Param[int] = field(default=100, ignore_default=True)
    micro_batch_size: Param[int] = field(default=100, ignore_default=True)

    def __validate__(self):
        assert self.batch_size % self.micro_batch_size == 0
```

### Object lifecycle

During task execution:
1. `__init__()` is called
2. Attributes are set
3. `__post_init__()` is called (if defined)
4. Pre-tasks run (if any)

Use `@initializer` for one-time deferred initialization.

## Defining Tasks

Tasks inherit from `Task` and implement `execute()`:

```python
from experimaestro import Task, Param, Meta, PathGenerator, field
from pathlib import Path

class ModelLearn(Task):
    """Train a model.

    Attributes:
        epochs: Number of training epochs
        model: The model to train
    """
    epochs: Param[int] = field(default=100, ignore_default=True)
    model: Param[Model]
    parameters: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))

    def execute(self):
        # Task implementation
        # self.__taskdir__ is the task directory
        # self.__tags__ contains tags as a dictionary
        for epoch in range(self.epochs):
            train_one_epoch(self.model)
        save(self.parameters)
```

### Task outputs and dependencies

By default, `task_outputs` returns `dep(self)`. Override for custom outputs:

```python
class RandomFold(Task):
    dataset: Param[Dataset]
    topics: Param[Path] = field(default_factory=PathGenerator("topics.tsv"))

    def task_outputs(self, dep) -> Dataset:
        return dep(Dataset.C(
            topics=dep(Topics.C(path=self.topics)),
            documents=self.dataset.documents,  # No dep() = no dependency
        ))
```

### Task dependencies

Tasks automatically resolve dependencies:

```python
task1 = Task1.C(...).submit()
task2 = Task2.C(...).submit()
task3 = Task3.C(inputs=[task1, task2]).submit()  # Waits for task1 and task2
```

## Resumable Tasks

For long-running tasks that may timeout (e.g., SLURM walltime):

```python
from experimaestro import ResumableTask, GracefulTimeout, Param, Meta, PathGenerator, field
from pathlib import Path

class LongTraining(ResumableTask):
    epochs: Param[int] = field(default=1000, ignore_default=True)
    checkpoint: Meta[Path] = field(default_factory=PathGenerator("checkpoint.pth"))

    def execute(self):
        start_epoch = 0
        if self.checkpoint.exists():
            start_epoch = load_checkpoint(self.checkpoint)

        for epoch in range(start_epoch, self.epochs):
            remaining = self.remaining_time()
            if remaining is not None and remaining < 300:
                save_checkpoint(self.checkpoint, epoch)
                raise GracefulTimeout("Not enough time for another epoch")

            train_one_epoch()
            save_checkpoint(self.checkpoint, epoch)
```

Key points:

- `remaining_time()` returns seconds remaining (SLURM) or `None` (local)
- Raise `GracefulTimeout` to stop cleanly and trigger retry
- Only timeout failures trigger retries (not other errors)
- Configure retries: `task.submit(max_retries=10)` or globally in settings

### Graceful termination (SIGTERM/SIGINT)

Catch `TaskCancelled` for custom cleanup:

```python
from experimaestro import Task, TaskCancelled

class MyTask(Task):
    def execute(self):
        try:
            do_work()
        except TaskCancelled:
            save_partial_state()
            raise  # Always re-raise
```

## Running Experiments

### Python API

```python
from experimaestro import experiment

with experiment(workdir, "my-experiment", port=12345) as xp:
    task = MyTask.C(param1="value", param2=42).submit()
```

### CLI with YAML configuration

```python
# experiment.py
from experimaestro.experiments import ExperimentHelper, configuration, ConfigurationBase

@configuration
class Configuration(ConfigurationBase):
    learning_rate: float = 1e-3

def run(helper: ExperimentHelper, cfg: Configuration):
    with helper.experiment() as xp:
        MyTask.C(lr=cfg.learning_rate).submit()
```

```yaml
# experiment.yaml
id: my-experiment
file: experiment
learning_rate: 1e-4
```

```bash
experimaestro run-experiment --run-mode NORMAL experiment.yaml
```

Override from CLI:

```bash
experimaestro run-experiment -c learning_rate=1e-5 experiment.yaml
```

## Launchers

### Direct (local)

Default launcher, runs tasks as local processes.

### SLURM

```python
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions

launcher = SlurmLauncher(nodes=1)
gpulauncher = launcher.config(gpu_per_node=1)

with experiment(workdir, "my-experiment", launcher=launcher) as xp:
    # Default launcher
    task1.submit()

    # GPU launcher for specific tasks
    task2.submit(launcher=gpulauncher)
```

### Launcher finder (`launchers.py`)

Define how requirements map to launchers in `~/.config/experimaestro/launchers.py`:

```python
from experimaestro.launcherfinder import HostRequirement, HostSpecification, CudaSpecification, CPUSpecification
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.launchers.direct import DirectLauncher
from experimaestro.connectors.local import LocalConnector

def find_launcher(requirements: HostRequirement, tags: set[str] = set()):
    if match := requirements.match(HostSpecification(accelerators=[])):
        return DirectLauncher(connector=LocalConnector.instance())

    if match := requirements.match(
        HostSpecification(
            max_duration=100 * 3600,
            cpu=CPUSpecification(cores=32, memory=129 * (1024**3)),
            accelerators=[CudaSpecification(memory=24 * (1024**3)) for _ in range(8)],
        )
    ):
        return SlurmLauncher(
            connector=LocalConnector.instance(),
            options=SlurmOptions(gpus_per_node=len(match.requirement.accelerators)),
        )

    return None
```

Use in code:

```python
from experimaestro.launcherfinder import find_launcher

launcher = find_launcher("duration=4h & cuda(mem=8GiB) * 2 & cpu(mem=16GiB)")
```

### Dynamic launcher

Select from multiple launchers based on priority:

```python
from experimaestro.launchers import DynamicLauncher
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions

fast = SlurmLauncher(options=SlurmOptions(partition="fast"), priority=10)
slow = SlurmLauncher(options=SlurmOptions(partition="slow"), priority=1)
dynamic = DynamicLauncher([fast, slow])
```

## Dynamic Task Outputs

For tasks producing outputs during execution (e.g., checkpoints):

```python
from experimaestro import ResumableTask, Config, Param, DependentMarker

class Checkpoint(Config):
    step: Param[int]
    model: Param[Model]

class Validation(Config):
    model: Param[Model]

    def checkpoint(self, dep: DependentMarker, *, step: int) -> Checkpoint:
        return dep(Checkpoint.C(model=self.model, step=step))

    def compute(self, step: int):
        self.register_task_output(self.checkpoint, step=step)

class Learn(ResumableTask):
    model: Param[Model]
    validation: Param[Validation]

    def execute(self):
        for step in range(100):
            train_step()
            if step % 10 == 0:
                self.validation.compute(step)

# Watch outputs
def on_checkpoint(checkpoint: Checkpoint):
    Evaluate.C(checkpoint=checkpoint).submit()

learn = Learn.C(model=model, validation=validation)
learn.watch_output(validation.checkpoint, on_checkpoint)
learn.submit()
```

## Deprecation

### Deprecating a configuration

When renaming or restructuring a config, use `@deprecate` to preserve identifier compatibility:

```python
from experimaestro import Config, Param, deprecate

class NewConfig(Config):
    values: Param[list[int]]

@deprecate(NewConfig)
class OldConfig(Config):
    value: Param[int]

    def __convert__(self):
        return NewConfig(values=[self.value])
```

### Deprecating a parameter

```python
class Learning(Config):
    losses: Param[list[Loss]]

    @deprecate
    def loss(self, value):
        assert len(self.losses) == 0
        self.losses.append(value)
```

Fix deprecated identifiers:

```bash
experimaestro deprecated list WORKDIR
```

## Value Classes

Separate configuration from implementation (e.g., avoid importing PyTorch at config time):

```python
from experimaestro import Config, Param

class Model(Config):
    hidden_size: Param[int]

@Model.value_class()
class TorchModel(Model):
    def __post_init__(self):
        import torch.nn as nn
        self.layers = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        return self.layers(x)
```

## Caching with `@cache`

For lightweight computations worth caching without a full task:

```python
from experimaestro import Config, cache
from pathlib import Path
import numpy as np

class Terms(Config):
    @cache("terms.npy")
    def load(self, path: Path):
        if path.is_file():
            return np.load(path)
        weights = self.compute_weights()
        np.save(path, weights)
        return weights
```

## Workspace Settings

Configure in `~/.config/experimaestro/settings.yaml`:

```yaml
workspaces:
  - id: my-workspace
    path: ~/experiments
    env:
      CUDA_VISIBLE_DEVICES: "0"
    triggers:
      - "my-experiment-*"
    max_retries: 3

history:
  max_done: 5
  max_failed: 1
```

## Monitoring

```bash
# TUI (terminal)
experimaestro experiments --workdir /path/to/workdir monitor --console

# Web UI
experimaestro experiments --workdir /path/to/workdir monitor --port 12345
```

## Common Pitfalls

1. **Identifier stability**: Changing parameter names, types, or `__xpmid__` breaks caching. Use `deprecate` for backward compatibility.
2. **Unsealed configs in sets**: Always use `sealed_set()` or `.seal()` before putting configs in `set[T]` parameters.
3. **RPyC serialization**: Keep task parameters simple (primitives, Paths, Config objects).
4. **Workspace locking**: Don't run multiple experiments in the same workspace simultaneously.
