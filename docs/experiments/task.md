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


## Complex initializations

Sometimes, it is necessary to restore an object state from disk, and we want
to separate the loading mechanism from the configuration logic; in that case,
`LightweightTask` (which must be subclassed) can be used:

```py3
from experimaestro import Config, LightweightTask

class Model(Config):
    ...

class SerializedModel(LightweightTask):
    ...

    def execute(self):
        # Access the configuration through self.config
        self.config.initialized = True
```


The most often use case is when the state can be recovered from disk. In that case,
`PathSerializationLWTask` can be used

```py3
from experimaestro import Config, DataPath, LightweightTask

class Model(Config):
    ...

class SerializedModel(PathSerializationLWTask):
    def initialize(self):
        # Loads the model from disk
        data = torch.load(self.path)
        self.config.load_state_dict(data)
```

## Tasks outputs

It is possible to generate a configuration when submitting a task.

!!! example "Task outputs"

    It is possible to use classes if variables need to be defined,
    or if a configuration should be returned (here, `Model`)

    ```py3
    from experimaestro import Serialized, Task, Param

    class ModelLoader(Serialized):
        @staticmethod
        def fromJSON(path: Path):
            return unserialize(path)

    class ModelLearn(Task):
        epochs: Param[int] = 100
        model: Param[Model]
        parameters: Annotated[Path, pathgenerator("parameters.pth")]

        def task_outputs(self) -> Model:
            return SerializedModel(config=self.model, path=ModelLoader(str(self.parameters)))

        def execute(self):
            """Called when this task is run"""
            pass

    # Building
    learn = ModelLearn(model=model)

    # learnedmodel can be used as a model
    learnedmodel = learn.submit()

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
