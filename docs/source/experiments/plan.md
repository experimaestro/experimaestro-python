# Experimental plans

## Defining plans

Once tasks and configurations are defined, experiments are defined imperatively by combining them.
In experimaestro, plans are defined imperatively which gives a lot of freedom.

Configuration and task arguments can be set by using a constructor or assigning a member within
the config/task instance, as in the example below:

:::{admonition} Example
:class: example

```python
model = Model1.C(layers=3)
learnedmodel = Learn.C(epochs=100)
learnedmodel.model = model
learnedmodel.submit()
```
:::

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

```python
# Creates a token with 3 "resources"
# (resources are whatever you want)
token = connector.createtoken("cpu", 3)

# Add two tasks that needs 2 "resources" from the token
# Those task won't be able to run at the same time
# and one will wait until there are enough "resources"
# before being run
task1 = token(2, task)
task2 = token(2, task)
```

(tags)=
## Tags

Tags allow to monitor specific experimental parameters.

### Tagging a value

Tagging a value can be done easily by using the `tag` function from `experimaestro`:

`tag(value: Union[str, int, float, bool])`

For example,

```python
from experimaestro import tag

model = MyModel.C(epochs=tag(100))
```

will create a tag with key "epochs" and value "100".

Adding a tag can be also be done by using a configuration instance method:

`tag(name: str, value: Union[str, int, float, bool])`

### Retrieving tags

To retrieve tags, use the `tags` method().
In the above example, `model.tags()` will return `{ "epochs": 100 }`

### Paths based on tags

Use `tagspath(config: Config)` to create a unique path where
all the tags associated with configuration will be associated with
their values. The keys are ordered to ensure the uniqueness of the path.

## Summarizing experimental results

For each experiment (identified by its name), a folder is created automatically. This
can be used to store additional experimental results as shown in the example below:

```python
from experimaestro import tag, tagspath

with experiment("...main experimental folder path...", "experiment ID", port=12346) as xp:
    model = Model.C()

    # Experimental plan
    models = {}
    for dlen_max, n_tokens in product([50, 200], [100, 1000]):
        data = Data.C(n_tokens=tag(n_tokens))
        learn = Learn.C(data=data, model=model, dlen_max=tag(dlen_max))
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

## Event handlers


Callbacks can be registered to accomplish some actions e.g. on task completion.

- `task.on_completed(callback: Callable[[], None])` register a callback that is
  called when the task terminates successfully


## Misc

### Command Line Arguments

You can easily define command line arguments with [click](https://click.palletsprojects.com)
by using the `forwardoption` command

```python
from experimaestro import Config, Param
from experimaestro.click import forwardoption, arguments
import click

class MyModel(Config):
    epochs: Param[int] = 100
    "Number of learning epochs"

@forwardoption(arguments(MyModel).epochs)
@click.command()
def cli(epochs):
    model = MyModel.C(epochs=epochs)
```

This will automatically use the type, help and default value of the matching option
