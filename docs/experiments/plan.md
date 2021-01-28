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

    ## Tokens

    Tokens can be used to restrict the number of running jobs.

    ```py3
    # Creates a token with 4 available slots
    token = connector.createtoken("cpu", 4)

    # Add a task that needs 2 slots from the token
    task = token(1, task)
    ```

## Tags

Tags allow to monitor specific experimental parameters.

### Tagging a value

Tagging a value can be done easily by using the `tag` function from `experimaestro`

```py3
tag(value: Union[str, int, float, bool])
```

For example,

```py3
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

Use `tagspath(value)`

## Summarizing experimental results

For each experiment (identified by its name), a folder is created automatically. Using both

```py3
with experiment("...main experimental folder path...", "experiment ID", port=12346) as xp:
    model = c.Model()

    # Experimental plan
    models = {}
    for dlen_max, n_tokens in product([50, 200], [100, 1000]):
        data = c.Data(n_tokens=tag(n_tokens))
        learn = c.Learn(data=data, model=model, dlen_max=tag(dlen_max))
        learn.add_dependencies(token.dependency(1))
        models[tagspath(learn)] = learn.submit().jobpath


    # Build a central "runs" directory to plot easily the metrics
    runpath = xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    for key, learn in models.items():
        (runpath / key).symlink_to(learn.jobpath / "runs")


    # Wait until experiment completes
    xp.wait()
    for key, learn in models.items():
        process(models)

```

## Conditional tasks

_Planned_

Sometimes, it can be useful to wait until a task completes - for instance, when exploring the hyperparameter
space, one can wait to launch new tasks based on the outcome. This can be achieved using the `check_results` callback:

```py3

task.submit().on_completed(check_results)
```

## Misc

### Command Line Arguments

You can easily define command line arguments with [click](https://click.palletsprojects.com)
by using the `forwardoption` command

```py3

from experimaestro import argument, type
from experimaestro.click import forwardoption, arguments
import click

@argument("epochs", type=int, default=100, help="Number of learning epochs")
@type("mymodel")
class MyModel: ...

@forwardoption(arguments(MyModel).epochs)
@click.command()
def cli(epochs):
    model = MyModel(epochs=epochs)
```

This will automatically use the type, help and default value of the matching option
