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

## Tasks outputs

It is possible to generate a configuration when submitting a task.

!!! example "Task outputs"

    It is possible to use classes if variables need to be defined,
    or if a configuration should be returned (here, `Model`)

    ```py3
    from experimaestro import Serialized, Task, Param

    class ModelLoader(Serialized):
        @staticmethod
        def fromJSON(path):
            return unserialize(path)

    class ModelLearn(Task):
        epochs: Param[int] = 100
        model: Param[Model]
        parameters: Annotated[Path, pathgenerator("parameters.pth")]

        def taskoutputs(self) -> Model:
            return SerializedConfig(self.model, ModelLoader(str(self.parameters)))

        def execute(self):
            """Called when this task is run"""
            pass

    # Building
    learn = ModelLearn(model=model)

    # learnedmodel is a Model configuration
    learnedmodel = learn.submit()

    ```

## Output configurations

Sometimes it is convenient to use a configuration object as one of the output of
a task. In this case, it is possible to use _output configurations_ nested
within tasks

```py3
class TaskValidation(Config):
    modelpath: Annotated[Path, pathgenerator("model.pth")]

class MyTask(Task):
    val: Param[TaskValidation]
```

When submitting a `MyTask` instance `mytask`, the `val` parameter is recognized as
an output configuration (because of the path generator), which means
that `mytask.val` can be used as a dependency and inherits the tags of `mytask`.

## Lightweights tasks using `@cache`

Sometimes, a configuration might need to compute some output that might be interesting to cache, but without relying on a fully-fledged task (because it can be done on the fly). In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

```py3
@config()
class Terms():
    @cache("terms.npy")
    def load(self, path: Path):
        if path.is_file():
            return pickle.load(path)

        # Value which can be long to compute
        weights = self.compute_weights()

        np.save(path, weights)
        return terms


```
