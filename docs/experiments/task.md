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

    # learnedmodel can be used as a model
    learnedmodel = learn.submit()

    ```

## Task output object

Calling submit returns a `TaskOutput` object, which is necessary to keep track
of dependencies. This task output is a proxy for the returned object, i.e.
accessing an attribute wraps it into another `TaskOutput` object.

`TaskOutput` objects have two specific methods and variables:

- `__xpm__` that is a `TaskOutputInformation` containing the `task` that was submitted
- `__unwrap__` that returns the wrapped value (warning: unwrapping might prevent dependency tracking from working)

For `TaskOutputInformation`, we have:

- `stdout()` and `stderr()` that return a `Path` to the file that contains the standard output/error
- `tags()` that returns the tags of the wrapped variable
- `wait()` that waits until the task is finished

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
