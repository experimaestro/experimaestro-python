# Experimental plans

Once tasks and configurations are defined, experiments are defined imperatively by combining them.

## Tags

Tags allow to monitor specific experimental parameters.

## Tagging a value

Tagging a value can be done easily by using the `tag` function from `experimaestro`
```
tag(value: Union[str, int, float, bool])
```

For example,
```python
model = MyModel(epochs=tag(100))
```

then `model.tags()` will return `{"epochs": 100}`

## Adding a tag

Adding a tag can be done by using a configuration instance method:
```python
tag(name: str, value: Union[str, int, float, bool])
```



# Misc

## Click integration

Easily define arguments with click

```python

from experimaestro import argument, type
from experimaestro.click import forwardoption
import click

@argument("epochs", type=int, default=100, help="Number of learning epochs")
@type("mymodel")
class MyModel: ...

@forwardoption(MyModel, "epochs")
@click.command()
def cli(epochs):
    model = MyModel(epochs=epochs)
```

This will automatically use the type, help and default value of the matching option

