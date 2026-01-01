# Tasks

A task is a special [configuration](./config.md) that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)

:::{admonition} Defining a task
:class: example

```python
from experimaestro import Config, Task, Param, Meta, PathGenerator, field

class ModelLearn(Task):
    epochs: Param[int] = 100
    model: Param[Model]
    parameters: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))

    def execute(self):
        """Called when this task is run"""
        pass
```
:::

## Task lifecycle

During task execution, the working directory is set to be the task directory,
and some special variables are defined:

- tags can be accessed as a dictionary using `self.__tags__`
- task directory can is `self.__taskdir__`
- when using [subparameters](./config.md#subparameters-and-partial-identifiers), `self.__maintaskdir__` is the directory of the main task


## Tasks outputs and dependencies

Task outputs can be re-used by other tasks. It is thus important
to properly define what is a task dependency (i.e. the task
should be run before) or not. To do so, the `task_outputs` method
of a `Task` takes one argument, `dep`, which can be used to mark a dependency
to the task.

By default, the task configuration is marked as a dependency as follows:

```python
from experimaestro import Task

class MyTask(Task):
    # (by default)
    def task_outputs(self, dep) -> Task:
        return dep(self)
```

For more complex cases, one can redefine the `task_outputs` method
and explicitly declare the dependencies.


:::{admonition} Task outputs
:class: example

In this example, we sample from a dataset composed of composed of queries
and documents. The documents are left untouched, but the topics are sampled.
In that case, we express the fact that:

- the returned object `Dataset` should be dependant on the task `RandomFold`
- the `topics` property of this dataset should also be dependant
- but the `documents` property should not (since we do not sample from it)

```python
from pathlib import Path
from experimaestro import Task, Param, field
from experimaestro.generators import PathGenerator

class RandomFold(Task):
    dataset: Param[Dataset]
    """The source dataset"""

    topics: Param[Path] = field(default_factory=PathGenerator("topics.tsv"))
    """Generated topics"""

    def task_outputs(self, dep) -> Adhoc:
        return dep(Dataset.C(
            topics=dep(Topics.C(path=self.topics)),
            documents=self.dataset.documents,
        ))
```
:::

## Common use case

A common use case is when we want to use a `Path` which contains the output of
the task A. Given how experimaestro works, just using it in a dependant task B
**will not work** since the ID will be the same whatever the task A
configuration. There are two ways to do so:

1. Wrap the task A output into a configuration object
2. Using initialization tasks


### Wrapping the task output

:::{admonition} Wrapping a task output
:class: example

```python
from pathlib import Path
from experimaestro import Config, Task, Param, Meta

class TaskA_Output(Config):
    path: Meta[Path]

class TaskA(Task):
    def task_outputs(self, dep) -> Task:
        return dep(MyTaskOutput.C(path=self.jobpath))

class TaskB(Task):
    task_a: Param[TaskA_Output]
```
:::

### Initialization tasks

The second solution is particularly suited when wanting to restore an object
state from disk, and we want to separate the loading mechanism from the
configuration logic; in that case, `LightweightTask` (a `Config` which must be
subclassed) can be used.


```python
from pathlib import Path
from experimaestro import Config, Param, Task, Meta, LightweightTask

class Model(Config):
    ...

class ModelLoader(LightweightTask):
    path: Meta[Path]
    model: Param[Model]

    def execute(self):
        ...

class Evaluate(Task):
    model: Param[Model]

    def execute(self):
        ...

class ModelLearner(Task):
    model: Param[Model]

    def task_outputs(self, dep):
        return dep(ModelLoader.C(model=model, path=path))

    def execute(self):
        ...


# We learn the model...
model_loader = learner.submit()

# ... and evaluate it, using the learned parameters
Evaluate.C(model=model).submit(init_tasks=[model_loader])
```



## Submit hooks

When a task is submitted, it is possible to modify the job/launcher environnement

```python
from experimaestro import Config
from experimaestro.core.types import SubmitHook
from experimaestro.scheduler.jobs import Job
from experimaestro.launchers import Launcher


class needs_java(SubmitHook):
    def __init__(self, version: int):
        self.version = version

    def spec(self):
        """Returns a hashable identifier for this hook (so it is only applied once)"""
        return self.version

    def process(self, job: Job, launcher: Launcher):
        """Apply the hook for a given job/launcher"""
        job.environ["JAVA_HOME"] = "THE_JAVA_HOME"
...

@needs_java(11)
class IndexCollection(Config):
    ...
```

(resumable-tasks)=
## Resumable Tasks

For long-running tasks that may be interrupted by scheduler timeouts (e.g., SLURM job time limits), you can use `ResumableTask` instead of `Task`. Resumable tasks can automatically retry when they fail due to timeouts, allowing them to resume from checkpoints.

:::{admonition} Defining a resumable task
:class: example

```python
from experimaestro import ResumableTask, Param, Meta, PathGenerator, field
from pathlib import Path

class LongTraining(ResumableTask):
    epochs: Param[int] = 1000
    checkpoint: Meta[Path] = field(default_factory=PathGenerator("checkpoint.pth"))

    def execute(self):
        # Check if we're resuming from a checkpoint
        start_epoch = 0
        if self.checkpoint.exists():
            start_epoch = load_checkpoint(self.checkpoint)

        # Continue training from where we left off
        for epoch in range(start_epoch, self.epochs):
            train_one_epoch()
            save_checkpoint(self.checkpoint, epoch)
```
:::

### Automatic Retry on Timeout

When a resumable task times out (e.g., reaches SLURM walltime limit), the scheduler automatically restarts it up to a maximum number of retries. The retry limit can be configured:

1. **Globally**: Set `max_retries` in workspace settings (`~/.config/experimaestro/settings.yaml`):
   ```yaml
   workspaces:
     - id: my_workspace
       path: /path/to/workspace
       max_retries: 5  # Default is 3
   ```

2. **Per-task**: Override when submitting:
   ```python
   task = LongTraining.C(epochs=1000).submit(max_retries=10)
   ```

### Important Considerations

- **Checkpointing**: Your task should save progress to checkpoints and check for existing checkpoints on startup
- **Idempotency**: The task should produce the same results whether run once or restarted multiple times
- **Directory preservation**: Unlike regular tasks, the task directory is preserved between retries to maintain checkpoints
- **Non-timeout failures**: Only timeout failures trigger retries. Other errors (out of memory, bugs, etc.) will not retry

### Querying Remaining Time

Resumable tasks can query the remaining time before a job timeout using the `remaining_time()` method. This is useful for deciding whether to start another iteration or checkpoint before the scheduler kills the job.

:::{admonition} Using remaining_time()
:class: example

```python
from experimaestro import ResumableTask, GracefulTimeout, Param, Meta, PathGenerator, field
from pathlib import Path

class LongTraining(ResumableTask):
    epochs: Param[int] = 1000
    checkpoint: Meta[Path] = field(default_factory=PathGenerator("checkpoint.pth"))

    def execute(self):
        start_epoch = 0
        if self.checkpoint.exists():
            start_epoch = load_checkpoint(self.checkpoint)

        for epoch in range(start_epoch, self.epochs):
            # Check if we have enough time for another epoch
            remaining = self.remaining_time()
            if remaining is not None and remaining < 300:  # 5 min buffer
                save_checkpoint(self.checkpoint, epoch)
                raise GracefulTimeout("Not enough time for another epoch")

            train_one_epoch()
            save_checkpoint(self.checkpoint, epoch)
```
:::

The `remaining_time()` method returns:

- **Remaining seconds** (as `float`): When running on a launcher with time limits (e.g., SLURM)
- **`None`**: When there is no time limit, or the launcher doesn't support querying remaining time

:::{note} Launcher Support
Currently, `remaining_time()` is supported for:

- **SLURM**: Queries the remaining walltime using `squeue`
- **Direct (local)**: Always returns `None` (no time limit)

The remaining time is cached internally, so repeated calls are efficient.
:::

### Graceful Timeout

Sometimes a task knows it won't have enough time to complete another processing step before the scheduler kills it (e.g., SLURM walltime). In this case, the task can raise `GracefulTimeout` to stop cleanly and trigger a retry.

:::{admonition} Using GracefulTimeout with remaining_time()
:class: example

```python
from experimaestro import ResumableTask, GracefulTimeout, Param, Meta, PathGenerator, field
from pathlib import Path

class LongTraining(ResumableTask):
    epochs: Param[int] = 1000
    checkpoint: Meta[Path] = field(default_factory=PathGenerator("checkpoint.pth"))

    def execute(self):
        start_epoch = 0
        if self.checkpoint.exists():
            start_epoch = load_checkpoint(self.checkpoint)

        for epoch in range(start_epoch, self.epochs):
            # Check remaining time before starting an epoch
            remaining = self.remaining_time()
            if remaining is not None and remaining < 300:  # 5 min buffer
                raise GracefulTimeout("Not enough time for another epoch")

            train_one_epoch()
            save_checkpoint(self.checkpoint, epoch)
```
:::

When `GracefulTimeout` is raised:

1. The task is marked as failed with reason "timeout"
2. If `retry_count < max_retries`, the task is automatically resubmitted
3. The task directory is preserved, allowing the task to resume from its checkpoint

This is useful when:

- Training epochs have variable duration and you want to stop before being killed
- You're monitoring remaining walltime via scheduler environment variables (e.g., `SLURM_JOB_END_TIME`)
- You want to ensure checkpoints are saved cleanly before termination

## Handling task events

Callbacks can be registered to accomplish some actions e.g. on task completion.

- `task.on_completed(callback: Callable[[], None])` register a callback that is
  called when the task terminates successfully

(dynamic-task-outputs)=
## Dynamic Task Outputs

For tasks that produce outputs during execution (e.g., checkpoints during training), you can use `watch_output` to register callbacks that are triggered when outputs are produced. This is particularly useful for triggering evaluation jobs on intermediate checkpoints.

:::{admonition} Defining dynamic outputs
:class: example

```python
from experimaestro import ResumableTask, Config, Param, DependentMarker

class Checkpoint(Config):
    step: Param[int]
    model: Param[Model]

class Validation(Config):
    model: Param[Model]

    def checkpoint(self, dep: DependentMarker, *, step: int) -> Checkpoint:
        """Method that produces dynamic outputs"""
        return dep(Checkpoint.C(model=self.model, step=step))

    def compute(self, step: int):
        """Called during task execution to register an output"""
        self.register_task_output(self.checkpoint, step=step)

class Learn(ResumableTask):
    model: Param[Model]
    validation: Param[Validation]

    def execute(self):
        for step in range(100):
            train_step()
            if step % 10 == 0:
                self.validation.compute(step)  # Triggers callbacks
```
:::

:::{admonition} Watching dynamic outputs
:class: example

```python
def on_checkpoint(checkpoint: Checkpoint):
    # Called when a checkpoint is produced
    Evaluate.C(checkpoint=checkpoint).submit()

learn = Learn.C(model=model, validation=validation)
learn.watch_output(validation.checkpoint, on_checkpoint)
learn.submit()
```
:::

### Key Features

- **ResumableTask only**: Only `ResumableTask` can use `register_task_output` (checked at runtime)
- **Automatic replay**: When a task is restarted, callbacks are replayed for previously produced outputs (events stored in `.experimaestro/task-outputs.jsonl`)
- **Multiple callbacks**: Multiple callbacks can watch the same output method
- **Separate thread**: Callbacks run in a dedicated worker thread

### How It Works

1. During task execution, `register_task_output()` writes events to a JSONL file
2. The experiment monitors this file and triggers registered callbacks
3. When a task is resubmitted after a failure, existing events are replayed to new callbacks
4. This enables seamless resumption of training with checkpoint-based evaluation

## Lightweights tasks using `@cache`

Sometimes, a configuration might need to compute some output that might be interesting to cache, but without relying on a fully-fledged task (because it can be done on the fly). In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

```python
from pathlib import Path
import pickle
import numpy as np
from experimaestro import Config, cache

class Terms(Config):
    @cache("terms.npy")
    def load(self, path: Path):
        if path.is_file():
            return pickle.load(path)

        # Value which can be long to compute
        weights = self.compute_weights()

        np.save(path, weights)
        return terms
```

# Task directory

When the task, submitted to the scheduler with `.submit(...)`,  is run,

- `[EXPERIMENT_FOLDER]` is the main experiment folder
- `[NAME]` is task name (by default, the python class name in lowercase)
- `[QUALIFIED_NAME]` is the qualified task name (by default, the python
  qualified name in lower case)
- `[JOB_ID]` is the [unique identifier](./config.md#configuration-identifiers)


All the task output should be located in the directory
`[EXPERIMENT_FOLDER]/jobs/[QUALIFIED_NAME]/[JOB_ID]`


Inside this directory, we have

- `__xpm__`, the folder that stores information about the process
- `[NAME].py` and (for some launchers) `[NAME].sh`  that contains the code that will be executed (through a cluster
  scheduler, e.g. slurm, or directly)
- `[NAME].err` and `[NAME].out` that stores the standard output and error
  streams
