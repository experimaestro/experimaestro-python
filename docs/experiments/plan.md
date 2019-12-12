# Experimental plans

Once tasks and configurations are defined, experiments are defined imperatively by combining them.

## Tags

Tags allow to monitor specific experimental parameters.

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

