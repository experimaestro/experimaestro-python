Defining experiments is based on *config(urations)* and *tasks*. Tasks are configurations that can be executed.

## Configurations



```
from experimaestro import config

@argument("gamma", type=float, required=False)
@config("my.model")
class MyModel: pass
```

defines a configuration with name `my.model` and one argument `gamma` that has the type `float`.

## Defining a task

A task is a special configuration that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)

```
from experimaestro import *

@argument("epochs", type=int, default=100)
@argument("model", type=Model, required=True)
@task("model.learn")
class ModelLearn:
    def execute(self):
        """Called when this task is run"""
        pass
```


