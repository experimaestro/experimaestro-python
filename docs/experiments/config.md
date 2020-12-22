Defining experiments is based on _config(urations)_ and _tasks_. Tasks are configurations that can be executed.

## Configurations

```python
@config(identifier=None, description=None, parents=[])
```

defines a configuration with identifier `identifier`, which could be any string.
If not given,it is the concatenation of the module full name with the class/function
name (lowercased).

!!! example

````
from experimaestro import config, argument

    @param("gamma", type=float, required=False)
    @config("my.model")
    class MyModel:
        pass
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

```py3
@task()
````

In most cases, it is easier to use a function
!!! example "Defining a task"

````
from experimaestro import task, argument

    @param("epochs", type=int, default=100)
    @param("model", type=Model, required=True)
    @task("model.learn")
    def modellearn(epochs: int, model: Model):
        pass
    ```

It is possible to use classes if variables need to be defined,
or if a configuration should be returned (here, `Model`)

!!! example
````

from experimaestro import task, argument

    @param("parameters", type=Path)
    @config()
    class Model:
        def __postinit__(self):
            # Called once the object has been set up
            print(self.parameters)

    @param("epochs", type=int, default=100)
    @param("model", type=Model, required=True)
    @pathoption("parameters", "parameters.pth")
    @task()
    class ModelLearn:
        def config(self) -> Model:
            return {
                "model": self.model,
                "parameters": self.parameters
            }

        def execute(self):
            """Called when this task is run"""
            pass
    ```

## Types

Types can be any simple type `int`, `float`, `str`, `bool` or `pathlib.Path` or *config*urations/tasks, as well as list of those (using `typing.List[T]`).

## Parameters

```python
@param(name: str, type: Any = None, default: Any = None, required: bool = None,
          ignored = None, help = None)
```

- `name` defines the name of the argument, which can be retrieved by the instance `self` (class) or passed as an argument (function)
- `type` is the type of the argument; if not given, it will be inferred from `default` (if defined) or be `Any`
- `default` default value of the argument. _If the value equals to the default, the argument will not be included in the signature computation_.
- `ignored` to ignore the argument in the signature computation (whatever its value).
- `help` a string to document the option; it can be used when the argument is used in a command line or when generating a documentation (_planned_).

Instead of using annotations, it is possible to use class variables
and type hints (**warning**: experimental syntax), as follows:

!!! example

````python
from experimaestro import task, Param

    @task("model.learn")
    class ModelLearn:
        epochs: Param[int] = 100
        model: Param[Model]
    ```

### Options

Options are just a simple shortcut to define a parameter with the `ignored` flag set. I

### Path option

```python
@pathoption(name: str, path: str, help: Optional[str] = None)
````

- `name` defines the name of the argument, which can be retrieved by the instance `self` (class) or passed as an argument (function)
- `path` is the path within the task directory

## Lightweights tasks using `@cache`

Sometimes, a config can compute some output that might be interesting to cache, but without relying on a fully-fledge task (because it can be done on the fly). In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

```python

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

## Validation

If a configuration or task has a `__validate__` method, it is called to validate
the values before a task is submitted. This allows to fail fast when parameters
are not valid.

```py3
@param("batch_size", type=int, default=100)
@param("micro_batch_size", type=int, default=100)
@pathoption("parameters", "parameters.pth")
@task()
class ModelLearn:
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
@subparam("epoch", type=int, default=100)
@param("learning_rate", type=float, default=1e-3)
@task()
class ModelLearn:
    def execute(self):
        pass
```

when the learning rate is the same, only one task is run at the same time.
