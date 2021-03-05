# Configurations

Defining experiments is based on _config(urations)_ and _tasks_. Tasks are configurations that can be executed.

## Defining a configuration

```py3
@config(identifier=None)
```

defines a configuration with identifier `identifier`, which could be any string.
If not given, it is the concatenation of the module full name with the class/function
name (lowercased).

!!! example

    ```py3
    from experimaestro import Param, config
        @config("my.model")
        class MyModel():
            gamma: Param[float]
    ```

defines a configuration with name `my.model` and one argument `gamma` that has the type `float`.

When an identifier is not given, it is computed as `__module__.__qualname__`. In that case,
it is possible to shorten the definition using the `Config` class as a base class.

!!! example

    ```py3
    from experimaestro import Param, Config

    class MyModel(Config):
        gamma: Param[float]
    ```

### Object life cycle

During [task](../task) execution, the objects are constructed following
this algorithm:

- The object is constructed using `self.__init__()`
- The attributes are set (e.g. `gamma` in the example above)
- `self.__postinit__()` is called (if the method exists)

## Types

Types can be any simple type `int`, `float`, `str`, `bool` or `pathlib.Path` or *config*urations/tasks, as well as list of those (using `typing.List[T]`).

## Parameters

```python
class MyConfig(Config):
    """
    Attributes:
        x: The parameter x
        y: The parameter y
    """
    # With default value
    x: Param[type] = value

    # Alternative syntax, useful to avoid class properties
    x: Annotated[type, default(value)]

    # Without default value
    y: Param[type]
```

- `name` defines the name of the argument, which can be retrieved by the instance `self` (class) or passed as an argument (function)
- `type` is the type of the argument
- `value` default value of the argument (if any). _If the value equals to the default, the argument will not be included in the signature computation_. This allows to add new parameters without changing the signature of past experiments (if the configuration is equivalent with the default value of course, otherwise do not use a default value!).

### Constants

Constants are special parameters that cannot be modified. They are useful to note that the
behavior of a configuration/task has changed, and thus that the signature should not be the
same (as the result of the processing will differ).

```py3
class MyConfig(Config):
    # Constant
    version: Constant[str] = "2.1"
```

### Options

Options are parameters which are ignored during the signature computation. For instance, the human readable name of a model would be an option. They are declared as parameters, but using the `Option` type hint

```py3
class MyConfig(Config):
    """
    Attributes:
        x: The option x
    """
    x: Option[type]
```

### Path option

It is possible to define special options that will be set
to paths relative to the task directory. For instance,

```py3
class MyConfig(Config):
    output: Annotated[Path, pathgenerator("output.txt")]
```

defines the instance variable `path` as a path `./output.txt` within
the task directory.

!!! warning
In case of conflicts (two configuration defining
the same path), the name of the file is changed using a suffix such as
`output.1.txt`.

## Validation

If a configuration has a `__validate__` method, it is called to validate
the values before a task is submitted. This allows to fail fast when parameters
are not valid.

```py3
class ModelLearn(Config):
    batch_size: Param[int] = 100
    micro_batch_size: Param[int] = 100
    parameters: Annotated[Path, pathgenerator("parameters.pth")]

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
