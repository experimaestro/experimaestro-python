# Experimental plans

## Defining plans

Once tasks and configurations are defined, experiments are defined imperatively by combining them. 
In experimaestro, plans are defined imperatively which gives a lot of freedom.

Configuration and task arguments can be set by using a constructor or assigning a member within
the config/task instance, as in the example below:

!!! example
    ```py3 linenums="1"
    model = Model1(layers=3)
    learnedmodel = Learn(epochs=100)
    learnedmodel.model = model
    learnedmodel.submit()
    ```

    - line 1: the `Model1` configuration is set with argument `layers` set to `3`.
    - line 2: the `Learn` task is configured with parameter `epochs` set to 100
    - line 3: Set the parameter `model` of the `Learn` task instance to `model`
    - line 4: Submit the task to the job scheduler

Once a task is submitted, the value of its arguments cannot be changed (it is **sealed**). 
This allows to re-use configuration/tasks latter.

### Unique job identifier

When a task is submitted, a unique id is computed based on the value of its arguments.
Parameters are ignored if:

- They were defined with `ignored` set to `True`
- They have a type `Path`








## Tags

Tags allow to monitor specific experimental parameters.

### Tagging a value

Tagging a value can be done easily by using the `tag` function from `experimaestro`
```
tag(value: Union[str, int, float, bool])
```

For example,
```python
model = MyModel(epochs=tag(100))
```

To retrieve tags, use the `tags` method().
In the above example, `model.tags()` will return `{ "epochs": 100 }`

### Adding a tag

Adding a tag can be done by using a configuration instance method:
```python
tag(name: str, value: Union[str, int, float, bool])
```

### Paths based on tags

*Planned*



## Misc

### Click integration

You can easily define command line arguments with [click](https://click.palletsprojects.com)
by using the `forwardoption` command

```python

from experimaestro import argument, type
from experimaestro.click import forwardoption
import click

@argument("epochs", type=int, default=100, help="Number of learning epochs")
@type("mymodel")
class MyModel: ...

@forwardoption(MyModel.epochs)
@click.command()
def cli(epochs):
    model = MyModel(epochs=epochs)
```

This will automatically use the type, help and default value of the matching option

