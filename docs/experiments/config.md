Defining experiments is based on _config(urations)_ and _tasks_. Tasks are configurations that can be executed.

## Configurations

```py3
@config(identifier=None)
```

defines a configuration with identifier `identifier`, which could be any string.
If not given,it is the concatenation of the module full name with the class/function
name (lowercased).

!!! example

    ```py3
    from experimaestro import Param, config
        @config("my.model")
        class MyModel:
            gamma: Param[float]
    ```

defines a configuration with name `my.model` and one argument `gamma` that has the type `float`.

### Object life cycle

During task execution,

- The object is constructed using `__init__(self)`
- The attributes are set (e.g. `gamma` in the example above)
- `self.__postinit__()` is called

## Tasks

A task is a special configuration that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)

!!! example "Defining a task as a class"

    It is possible to use classes if variables need to be defined,
    or if a configuration should be returned (here, `Model`)

    ```py3
    from experimaestro import Config, Task, Param

        @config()
        class Model:
            parameters: Param[Path]

            def __postinit__(self):
                # Called once the object has been set up
                print(self.parameters)

        @task()
        class ModelLearn():
            epochs: Param[int] = 100
            model: Param[Model]
            parameters: PathOption = "parameters.pth"

            def config(self) -> Model:
                return {
                    "model": self.model,
                    "parameters": self.parameters
                }

            def execute(self):
                """Called when this task is run"""
                pass
    ```

Tags can be accessed as a dictionary using `self.__tags__`.

## Types

Types can be any simple type `int`, `float`, `str`, `bool` or `pathlib.Path` or *config*urations/tasks, as well as list of those (using `typing.List[T]`).

## Parameters

```python
class MyConfig:
    """
    Attributes:
        x: The parameter x
        y: The parameter y
    """
    # With default value
    x: Param[type] = default

    # Without default value
    y: Param[type]
```

- `name` defines the name of the argument, which can be retrieved by the instance `self` (class) or passed as an argument (function)
- `type` is the type of the argument
- `default` default value of the argument (if any). _If the value equals to the default, the argument will not be included in the signature computation_.

### Options

Options are parameters which are ignored during the signature computation. For instance, the human readable name of a model would be an option. They are declared as parameters, but using the `Option` type hint

```python
class MyConfig:
    """
    Attributes:
        x: The option x
    """
    x: Option[type]
```

### Path option

```py3

class MyTask:
    name: PathOption = Path("path")
```

- `name` defines the name of the argument, which can be retrieved by the instance `self` (class) or passed as an argument (function)
- `path` is the path within the task directory

## Lightweights tasks using `@cache`

Sometimes, a config can compute some output that might be interesting to cache, but without relying on a fully-fledge task (because it can be done on the fly). In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

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

## Validation

If a configuration or task has a `__validate__` method, it is called to validate
the values before a task is submitted. This allows to fail fast when parameters
are not valid.

```py3
class ModelLearn(Task):
    batch_size: Param[int] = 100
    micro_batch_size: Param[int] = 100
    parameters: PathOption = "parameters.pth"

    def __validate__(self):
        assert self.batch_size % self.micro_batch_size == 0
```

## Sub-parameters

When one to re-use partial results from a previous task,
e.g. for instance when running a model with a different number of epochs,
one can use _sub-parameters_. Tasks with the same parameters
but with different _sub-parameters_ are run sequentially.

For instance, given this task definition

```py3
class ModelLearn(Task):
    epoch: SubParam[int] = 100
    learning_rate: Param[float] = 1e-3
    def execute(self):
        pass
```

when the learning rate is the same, only one task is run at the same time.
