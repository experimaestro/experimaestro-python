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

## Generating new configurations

It is possible to generate a configuration when submitting a task.

!!! example "Defining a task as a class"

    It is possible to use classes if variables need to be defined,
    or if a configuration should be returned (here, `Model`)

    ```py3
        class ModelLearn(Task):
            epochs: Param[int] = 100
            model: Param[Model]
            parameters: Annotated[Path, pathgenerator("parameters.pth")]

            def config(self) -> Model:
                return self.model

            def execute(self):
                """Called when this task is run"""
                pass

        # Building
        learn = ModelLearn(model=model)

        # learnedmodel is a Model configuration
        learnedmodel = learn.submit()

    ```

## Lightweights tasks using `@cache`

Sometimes, a config can compute some output that might be interesting to cache, but without relying on a fully-fledge task (because it can be done on the fly). In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

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
