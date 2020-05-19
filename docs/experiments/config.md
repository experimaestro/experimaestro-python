Defining experiments is based on *config(urations)* and *tasks*. Tasks are configurations that can be executed.

## Configurations

```python
@config(identifier=None, description=None, parents=[])
```
defines a configuration with identifier `identifier`, which could be any string. 
If not given,it is the concatenation of the module full name with the class/function
name (lowercased).


!!! example
    ```
    from experimaestro import config, argument

    @argument("gamma", type=float, required=False)
    @config("my.model")
    class MyModel: pass
    ```

    defines a configuration with name `my.model` and one argument `gamma` that has the type `float`.

## Tasks

A task is a special configuration that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)


```py3
@task()
```


In most cases, it is easier to use a function
!!! example "Defining a task"
    ```
    from experimaestro import task, argument

    @argument("epochs", type=int, default=100)
    @argument("model", type=Model, required=True)
    @task("model.learn")
    def modellearn(epochs: int, model: Model):
        pass
    ```

It is possible to use classes if variables need to be defined

!!! example
    ```
    from experimaestro import task, argument

    @argument("epochs", type=int, default=100)
    @argument("model", type=Model, required=True)
    @task("model.learn")
    class ModelLearn:
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
- `default` default value of the argument. *If the value equals to the default, the argument will not be included in the signature computation*.
- `ignored` to ignore the argument in the signature computation.
- `help` a string to document the option; it can be used when the argument is used in a command line or when generating a documentation (*planned*).

Instead of using annotations, it is possible to use class variables
and type hints (**warning**: experimental syntax), as follows:

!!! example
    ```python
    from experimaestro import task, Param

    @task("model.learn")
    class ModelLearn:
        epochs: Param[int] = 100
        model: Param[Model]
    ```



## Path arguments

```python
@pathargument(name: str, path: str, help: Optional[str] = None)
```

- `name` defines the name of the argument, which can be retrieved by the instance `self` (class) or passed as an argument (function)
- `path` is the path within the task directory


## Lightweights tasks using `@cache`

Sometimes, a config can compute some output which is might be interesting to cache, but without relying on a fully-fledge task. In those cases, the annotation `@cache` can be used. Behind the curtain, a config cache is created (using the configuration unique identifier) and the `path` is locked (avoiding problems if the same configuration is used in two running tasks):

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