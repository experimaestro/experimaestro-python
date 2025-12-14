# Tasks

A task is a special [configuration](./config.md) that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)

!!! example "Defining a task"

    ```py3
    from experimaestro import Config, Task, Param, Meta, PathGenerator, field

    class ModelLearn(Task):
        epochs: Param[int] = 100
        model: Param[Model]
        parameters: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))

        def execute(self):
            """Called when this task is run"""
            pass
    ```

## Task lifecycle

During task execution, the working directory is set to be the task directory,
and some special variables are defined:

- tags can be accessed as a dictionary using `self.__tags__`
- task directory can is `self.__taskdir__`
- when using [sub-parameters](./config.md#sub-parameters), `self.__maintaskdir__` is the directory of the main task


## Tasks outputs and dependencies

Task outputs can be re-used by other tasks. It is thus important
to properly define what is a task dependency (i.e. the task
should be run before) or not. To do so, the `task_outputs` method
of a `Task` takes one argument, `dep`, which can be used to mark a dependency
to the task.

By default, the task configuration is marked as a dependency as follows:

```py3

    class MyTask(Task):
        # (by default)
        def task_outputs(self, dep) -> Task:
            return dep(self)
```

For more complex cases, one can redefine the `task_outputs` method
and explicitly declare the dependencies.


!!! example "Task outputs"

    In this example, we sample from a dataset composed of composed of queries
    and documents. The documents are left untouched, but the topics are sampled.
    In that case, we express the fact that:

    - the returned object `Dataset` should be dependant on the task `RandomFold`
    - the `topics` property of this dataset should also be dependant
    - but the `documents` property should not (since we do not sample from it)

    ```py3

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

## Common use case

A common use case is when we want to use a `Path` which contains the output of
the task A. Given how experimaestro works, just using it in a dependant task B
**will not work** since the ID will be the same whatever the task A
configuration. There are two ways to do so:

1. Wrap the task A output into a configuration object
2. Using initialization tasks


### Wrapping the task output

!!! example "Wrapping a task output"
    ```py
    from experimaestro import Config, Task, Param, Meta

    class TaskA_Output(Config):
        path: Meta[Path]

    class TaskA(Task):
        def task_outputs(self, dep) -> Task:
            return dep(MyTaskOutput.C(path=self.jobpath))

    class TaskB(Task):
        task_a: Param[TaskA_Output]
    ```

### Initialization tasks

The second solution is particularly suited when wanting to restore an object
state from disk, and we want to separate the loading mechanism from the
configuration logic; in that case, `LightweightTask` (a `Config` which must be
subclassed) can be used.


```py3
    from experimaestro import Config, Param, Task, Meta, LightweightTask

    class Model(Config):
        ...

    class ModelLoader(LightweightTask):
        path: Meta[Path]
        model: Param[Model]

        def execute(self):
            ...

    def Evaluate(Task):
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

```py3
from experimaestro import SubmitHook, Job, Launcher, submit_hook_decorator


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

## Resumable Tasks

For long-running tasks that may be interrupted by scheduler timeouts (e.g., SLURM job time limits), you can use `ResumableTask` instead of `Task`. Resumable tasks can automatically retry when they fail due to timeouts, allowing them to resume from checkpoints.

!!! example "Defining a resumable task"

    ```py3
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
   ```py3
   task = LongTraining.C(epochs=1000).submit(max_retries=10)
   ```

### Important Considerations

- **Checkpointing**: Your task should save progress to checkpoints and check for existing checkpoints on startup
- **Idempotency**: The task should produce the same results whether run once or restarted multiple times
- **Directory preservation**: Unlike regular tasks, the task directory is preserved between retries to maintain checkpoints
- **Non-timeout failures**: Only timeout failures trigger retries. Other errors (out of memory, bugs, etc.) will not retry

## Handling task events

Callbacks can be registered to accomplish some actions e.g. on task completion.

- `task.on_completed(callback: Callable[[], None])` register a callback that is
  called when the task terminates successfully

## Lightweights tasks using `@cache`

Sometimes, a configuration might need to compute some output that might be interesting to cache, but without relying on a fully-fledged task (because it can be done on the fly). In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

```py3
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

- `.experimaestro`, the folder that stores information about the process
- `[NAME].py` and (for some launchers) `[NAME].sh`  that contains the code that will be executed (through a cluster
  scheduler, e.g. slurm, or directly)
- `[NAME].err` and `[NAME].out` that stores the standard output and error
  streams
