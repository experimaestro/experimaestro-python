# Tasks

A task is a special configuration that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)

!!! example "Defining a task"

    ```py3
    from experimaestro import Config, Task, Param

    class ModelLearn(Task):
        epochs: Param[int] = 100
        model: Param[Model]
        parameters: Annotated[Path, pathgenerator("parameters.pth")]

        def execute(self):
            """Called when this task is run"""
            pass
    ```

## Task lifecycle

During task execution, the working directory
is set to be the task directory, and
some special variables are defined:

- tags can be accessed as a dictionary using `self.__tags__`
- task directory can is `self.__taskdir__`
- when using [sub-parameters](../config#sub-parameters), `self.__maintaskdir__` is the directory of the main task


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

            topics: Annotated[Path, pathgenerator("topics.tsv")]
            """Generated topics"""

            def task_outputs(self, dep) -> Adhoc:
                return dep(Dataset(
                    topics=dep(Topics(path=self.topics)),
                    documents=self.dataset.documents,
                ))

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
