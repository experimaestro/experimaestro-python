Python types and tasks can be defined using annotations

## Defining a type

Types can be either used by tasks or other types

```
from experimaestro import *

@TypeArgument("gamma", type=float, required=False)
@RegisterType("my.model")
class MyModel:
    def _init(self):
        """Called when the object has been created and arguments set"""
        pass
```

## Defining a task

A task is a special type that can be:

1. Submitted to the task scheduler using `submit` (preparation of the experiment)
1. Executed with the method `execute` (running a specific task within the experiment)

```
from experimaestro import *

@TypeArgument("epochs", type=int, default=100)
@TypeArgument("model", type=Model, required=True)
@RegisterTask("model.learn")
class ModelLearn:
    def _init(self):
        """Called when the object has been created and arguments set"""
        pass

    def execute(self):
        """Called when this task is run"""
        pass
```


## Command line parsing

### Click

Easily define arguments with click

```python

@TypeArgument("epochs", type=int, default=100, help="Number of learning epochs")
@RegisterType("mymodel")
class MyModel: ...

@forwardoption(MyModel, "epochs")
@main.command()
def experiments(epochs):
    pass
```

This will automatically use the type, help and default value of the matching option


## Running


```
import experimaestro as xpm

# Put your definitions here: either load a YAML or import the definitions

if __name__ == "__main__":
    xpm.register.parse()
```

