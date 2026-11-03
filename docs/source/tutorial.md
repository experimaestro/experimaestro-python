# Walkthrough

This page introduces the main concepts of experimaestro on a deliberately
small example: fitting a line to noisy points, and comparing a few learning
rates. Everything runs on a laptop in a few seconds, with no dependency other
than experimaestro, so the code stays out of the way of the concepts. The
[complete example](#the-complete-example) at the end of the page goes
further.

The example is a small project with four files:

```
linear/
├── tasks.py          # what can be computed: configurations and tasks
├── experiment.py     # what to compute: the experimental plan
├── params.yaml       # the values of the experimental parameters
└── results.py        # collecting the results
```

## Configurations

Everything experimaestro manipulates is described by **configurations**:
classes deriving from {py:class}`~experimaestro.Config` whose
**parameters** are declared with `Param[type]` annotations. Here, a dataset
of noisy points on the line `y = 2x + 1`:

```{literalinclude} ../../src/experimaestro/tests/tutorial/tasks.py
:language: python
:caption: tasks.py
:end-before: class LinearModel
```

A configuration is created with `.C(...)`, e.g. `Dataset.C(size=100, seed=1)`,
and behaves like a dataclass. The key property is its **identifier**: a hash
of its type and parameter values. Two configurations with the same parameters
have the same identifier, and changing any parameter changes it.

`field(ignore_default=0.1)` gives `noise` a default value that is *left out*
of the identifier when it is used, so that adding a parameter with a default
does not change the identifier of existing configurations (use
`field(default=...)` to always include it). The docstrings below parameters
document them, and can be used to generate documentation.
See [Configurations](experiments/config.md).

Since results are stored by identifier, **identifiers must stay stable** as
the code evolves: renaming a parameter or changing its type gives new
identifiers, and earlier results are silently computed again. Renaming a
class does too, unless its identifier is pinned with `__xpmid__`; and a
parameter that has to change can be kept readable with
{py:class}`~experimaestro.DeprecatedAttribute`. See
[Keep task identifiers stable](experiments.md#keep-task-identifiers-stable).

## Tasks

A **task** is a configuration that can be *run*: it derives from
{py:class}`~experimaestro.Task` and implements `execute()`. Here, learning
the line by gradient descent:

```{literalinclude} ../../src/experimaestro/tests/tutorial/tasks.py
:language: python
:caption: tasks.py
:pyobject: Learn
```

Three things to notice:

- **Parameters can be other configurations** (`data: Param[Dataset]`), so
  experimental components compose: the identifier of a `Learn` task depends on
  the dataset's parameters too.
- **`Meta` values are not parameters**: they don't change the identifier.
  `PathGenerator("model.json")` gives a path inside the task's own folder,
  where it writes its output.
- **Each task runs in its own folder**,
  `<workspace>/jobs/<task id>/<identifier>/`, with its parameters
  (`params.json`), its standard output and error (`learn.out`, `learn.err`,
  here with the epochs printed by `execute()`), and a `learn.done` marker once
  it has completed. Since the folder depends only on the parameters, a task
  that already ran with the same parameters is not run again.

### Task outputs

What a task produces is described by `task_outputs()`, whose result is what
`submit()` returns. `Learn` produces a model: a `LinearModel` configuration
pointing to the file it writes.

```{literalinclude} ../../src/experimaestro/tests/tutorial/tasks.py
:language: python
:caption: tasks.py
:pyobject: LinearModel
```

`dep(...)` marks this configuration as depending on the task. This is how
**dependencies** are expressed: a task with this model as a parameter only
starts once `Learn` has completed, and the model's identifier depends on the
`Learn` task that produced it (although its only parameter, the path, is
`Meta`). A second task evaluates a model on another dataset:

```{literalinclude} ../../src/experimaestro/tests/tutorial/tasks.py
:language: python
:caption: tasks.py
:pyobject: Evaluate
```

`Evaluate` takes any `LinearModel`: it does not need to know how the model
was obtained. See [Tasks](experiments/task.md).

### Configurations and objects

`.C(...)` builds a *configuration*: a description, from which nothing is
computed. In the experiment, the models are only configurations, and their
files don't exist yet. When a task runs, its parameters become real Python
objects: each one is created, its attributes are set, then its
`__post_init__()` method is called. This is when `LinearModel` reads its
file, and why `execute()` can call `self.data.points()` or
`self.model.predict(x)`. See
[Object life cycle](experiments/config.md#object-life-cycle).

## The experimental plan

The plan says what to compute. It lives in a `run(helper, cfg)` function,
whose configuration (`cfg`) is read from a YAML file:

```{literalinclude} ../../src/experimaestro/tests/tutorial/experiment.py
:language: python
:caption: experiment.py
```

```{literalinclude} ../../src/experimaestro/tests/tutorial/params.yaml
:language: yaml
:caption: params.yaml
```

- **`Configuration`** declares the values the YAML file can (or must) give,
  and checks their types. `id` names the experiment; `file` is the Python file
  containing `run`. See [The CLI experiment runner](experiments/cli-runner.md).
- **`submit()`** hands a task to the scheduler, and returns its output (here
  the `LinearModel`). The scheduler runs each task as soon as its dependencies
  are done: the three `Learn` tasks run in parallel, and each `Evaluate` waits
  for the `Learn` that produced its model. The training set is shared by the
  three `Learn` tasks, but it is a plain configuration, not a task: nothing
  has to run for it.
- **`tag(lr)`** marks the learning rate as a value worth showing: tags are
  displayed in the monitoring interfaces and kept with the results, and are
  inherited by the tasks that depend on a tagged one (each `Evaluate` is
  tagged with the `lr` of its model). They don't change identifiers.
  See [Plans](experiments/plan.md), and [Grid search](experiments/grid_search.md)
  to declare parameter sweeps in the YAML file instead of loops.
- **`find_launcher("cpu(cores=1)")`** turns a resource requirement into a way
  to run the task (locally, on a SLURM cluster, …), as described by a
  `launchers.py` file. Without that file, tasks run as local processes, which
  is all this example needs. `experimaestro launchers direct generate`
  writes one for your machine, and `experimaestro launchers slurm generate`
  for a cluster: the experiment itself does not change.
  See [Launchers](launchers/index.md).

## Running the experiment

Tasks are run in a **workspace**, the folder where the job folders and the
experiment records are written. Give it on the command line with
`--workdir`, or define it once in `~/.config/experimaestro/settings.yaml`,
where `triggers` select it automatically from the experiment ID:

```yaml
workspaces:
  - id: tutorial
    path: ~/experiments/tutorial
    triggers: ["linear"]
```

See [Settings](settings.md). First check what would run, without running
anything:

```bash
uv run experimaestro run --run-mode dry_run params.yaml
```

This lists each task with its identifier, its tags, its launcher and its
dependencies. Then run it, following it in the terminal interface (or in the
browser with `--web`):

```bash
uv run experimaestro run --console params.yaml
```

Run it a second time: nothing runs, since all the tasks are already done. Add
a learning rate to `params.yaml` and only its two tasks run. A finished
experiment can be browsed at any time with `experimaestro monitor --console`
(or `--web`). See [Running experiments](experiments.md) and
[Interfaces](interfaces.md).

## Collecting the results

Experimaestro records the configurations each experiment submitted, so
results can be gathered without keeping track of job folders by hand:

```{literalinclude} ../../src/experimaestro/tests/tutorial/results.py
:language: python
:caption: results.py
```

The example is tested: the code on this page is the one of
`src/experimaestro/tests/tutorial/`, and the output below is checked by
its test.

```console
$ uv run python results.py ~/experiments/tutorial
{'lr': 0.001} mse=0.0098
{'lr': 0.01} mse=0.0088
{'lr': 0.1} mse=0.0092
```

See [Analysis](experiments/analysis.md), and
[Actions](experiments/actions.md) to attach such post-processing (e.g.
exporting the best model) to the experiment itself.

## When a task fails

Add `10` to `learning_rates` in `params.yaml` and run the experiment again.
Gradient descent diverges with such a learning rate, and `Learn` raises an
error. The evaluation of that model is not run, since the model does not
exist, and the experiment ends with `Experiment failed: 1 failed job (and 1 job
not run because of a failed dependency)`; `experimaestro run` then exits with
code 1. The error is in the job
folder: `learn.err` holds the traceback, and the job details of the
monitoring interfaces show the failure reason with the exception message and
where it was raised.

Running the experiment again only runs the failed tasks again: the finished
ones are kept. Here, `Learn` fails until the learning rate is removed from
`params.yaml`. (A task deriving from
{py:class}`~experimaestro.ResumableTask` is also restarted automatically when
it reaches the time limit of a cluster, see
[Resumable tasks](experiments/task.md).)

## The complete example

The [experimaestro-demo](https://github.com/experimaestro/experimaestro-demo)
repository applies the same concepts to a realistic project, a CNN trained on
MNIST, and introduces what this page leaves out:

- **A real model and dataset**: a PyTorch CNN, with the MNIST dataset
  described by [datamaestro](https://datamaestro.readthedocs.io/en/latest/),
  which downloads and caches datasets and gives them an identifier.
- **A full project**: a Python package with its `pyproject.toml`, a workspace
  definition to install in `settings.yaml`, and smoke tests run by the CI in
  dry-run mode. See [Experimaestro projects](experiments/projects.md).
- **GPU resources**: tasks requiring `cuda(mem=4G)`, and a generated
  `launchers.py` whose tokens keep concurrent tasks from oversubscribing the
  GPU memory.
- **Declarative grid search**: hyper-parameters declared as `GridSearch[int]`
  in the configuration, given as lists or ranges in the YAML file, and
  expanded with `generate_grid(cfg)` instead of nested loops.
  See [Grid search](experiments/grid_search.md).
- **Post-experiment actions**: an `ExportBestModel` action, registered by the
  experiment and run afterwards from the CLI or the TUI, which copies the best
  model. See [Actions](experiments/actions.md).
- **Analysis and the task graph**: collecting the accuracies of all the
  models, and drawing the dependency graph of the experiment.

## Going further

These features are worth knowing once the basic flow above is comfortable:

- **Long-running / preemptible jobs** — wrap `Learn` as a [`ResumableTask`](experiments/task.md), check `remaining_time()` and raise `GracefulTimeout` to checkpoint cleanly when a SLURM walltime is about to expire.
- **Shared workspace settings** — `imports:` in `settings.yaml` lets multiple project-level YAML files inherit a base workspace block. See [Settings](settings.md).
- **Archiving completed jobs** (beta) — declare auxiliary `folders` with `mode: backup` (or `move`) on a workspace to automatically copy/migrate finished job directories to long-term storage. See [Settings](settings.md).
- **Moving experiments across workspaces** — `experimaestro experiments copy` transfers a finished experiment (and its job directories) between workspaces, useful when promoting a laptop prototype to a cluster run. See [CLI](cli.md).

## Reference

- [Configurations](experiments/config.md) — deep dive into configuration objects
- [Tasks](experiments/task.md) — task definition, ResumableTask, dynamic outputs
- [Launchers](launchers/index.md) — direct and SLURM launchers and the requirement DSL
- [Settings](settings.md) — workspaces, triggers, imports, folders
- [CLI](cli.md) — command-line interface reference
