# Experimaestro projects

While you *can* drive Experimaestro from a single standalone script (see the
[hello-world example](../index.md)), real experiments are organised as a
**project**: a Python package whose experimental plan is launched and monitored
through the `experimaestro run-experiment` command line.

A project bundles together everything needed to reproduce a set of experiments:
the {py:class}`~experimaestro.Config`/{py:class}`~experimaestro.Task`
definitions, the orchestration code, the parameter files, and the project
metadata (dependencies, formatting, tests). This page describes how such a
project is laid out and how its pieces fit together.

## The experimaestro ecosystem

A number of libraries and tools are built around experimaestro. Most of them
live in the [experimaestro GitHub organisation](https://github.com/experimaestro):

**Core**

- [experimaestro-python](https://github.com/experimaestro/experimaestro-python) —
  the Python framework documented here.

**Datasets — [datamaestro](https://github.com/experimaestro/datamaestro)**

A companion dataset manager that pairs naturally with experimaestro tasks, with
domain-specific plugins:

- [datamaestro](https://github.com/experimaestro/datamaestro) — core dataset
  handling (download, caching, standardised access).
- [datamaestro_text](https://github.com/experimaestro/datamaestro_text) — NLP /
  information-access datasets.
- [datamaestro_image](https://github.com/experimaestro/datamaestro_image) —
  image datasets (e.g. MNIST, used by the demo).
- [datamaestro_ml](https://github.com/experimaestro/datamaestro_ml) — generic
  machine-learning datasets.
- [datamaestro_ir](https://github.com/xpmir/datamaestro_ir) — information
  retrieval datasets.

**Domain libraries**

Libraries that ship reusable {py:class}`~experimaestro.Config`/
{py:class}`~experimaestro.Task` definitions and a specialised experiment helper
for a whole field (see [Building a domain-specific library](#building-a-domain-specific-library)):

- [xpm-torch](https://github.com/experimaestro/xpm-torch) — an add-on with
  building blocks for PyTorch-based experiments.
- [experimaestro-ir (xpmir)](https://github.com/experimaestro/experimaestro-ir) —
  information-retrieval tasks and configurations (built on top of xpm-torch).
  Each *specific* experiment is then its own
  project that depends on xpmir — many live in the
  [xpmir GitHub organisation](https://github.com/xpmir), e.g.
  [splade](https://github.com/xpmir/splade) (SPLADE paper reproductions),
  [cosplade](https://github.com/xpmir/cosplade),
  [cross-encoders](https://github.com/xpmir/cross-encoders) and
  [mice](https://github.com/xpmir/mice). The
  [xpmir/experiment-template](https://github.com/xpmir/experiment-template) is a
  skeleton for such paper projects.

**Tools & services**

- [xpm-mlboard](https://github.com/experimaestro/xpm-mlboard) — lightweight
  experimaestro [services](../services.md) to monitor ML learning curves
  (TensorBoard, …).

**Starting points**

- [experiment-template](https://github.com/experimaestro/experiment-template) —
  a minimal project skeleton to clone.
- [experimaestro-demo](https://github.com/experimaestro/experimaestro-demo) — a
  fuller worked example (the MNIST [tutorial](../tutorial.md)).

## The smallest possible project

The [`experiment-template`](https://github.com/experimaestro/experiment-template)
repository is a ready-to-clone skeleton with the minimum a project needs:

```
experiment-template/
├── pyproject.toml          # project + dependencies (depends on experimaestro)
├── my_project/
│   ├── __init__.py
│   ├── experiment.py       # the run(helper, cfg) entry point
│   └── normal.yaml         # experiment configuration (id, module, parameters)
└── README.md
```

The entry point is a `run` function taking an
{py:class}`~experimaestro.experiments.cli.ExperimentHelper` and a configuration
object:

```python
# my_project/experiment.py
from experimaestro.experiments import ConfigurationBase, ExperimentHelper, configuration


@configuration
class Configuration(ConfigurationBase):
    my_param: int
    """An experimental parameter"""


def run(helper: ExperimentHelper, cfg: Configuration):
    # Build and submit your experimental plan here
    ...
```

The matching YAML file selects the module to run and supplies parameter values:

```yaml
# my_project/normal.yaml
id: my-project
module: my_project.experiment
my_param: 42
```

and the experiment is launched with:

```bash
uv run experimaestro run-experiment my_project/normal
```

`run-experiment` reads the YAML, builds the `Configuration` object (validating
types), imports the module, and calls its `run(helper, cfg)` function. See the
[experiment configuration reference](../experiments.md) for the full list of
YAML options (`id`, `file`/`module`, `pythonpath`, `imports`, `pre_experiment`,
`dirty_git`, …) and the [CLI documentation](../cli.md#running-experiments) for
the command-line flags (`--run-mode`, `-c`, `--pre-yaml`/`--post-yaml`,
`--show`, …).

## Anatomy of a fuller project

As a project grows, the experimental code is split into focused modules. The
[`experimaestro-demo`](https://github.com/experimaestro/experimaestro-demo)
repository (an MNIST CNN hyper-parameter search, also used as the
[tutorial](../tutorial.md)) is a representative example:

```
experimaestro-demo/
├── pyproject.toml            # dependencies + tooling (ruff, mypy, pytest)
├── xpm_settings.yaml         # workspace definitions (copied to ~/.config/experimaestro)
├── mnist_xp/
│   ├── learn.py              # Config / Task definitions (CNN model, Learn, Evaluate)
│   ├── data.py               # dataset configurations (via datamaestro)
│   ├── experiment.py         # run(helper, cfg): builds and submits the plan
│   ├── params.yaml           # configuration values (id, launcher, hyper-parameters)
│   ├── actions.py            # post-experiment Actions (e.g. export best model)
│   ├── analyze.py            # post-hoc result analysis (load_xp_info)
│   └── tensorboard_service.py# a custom experiment service
└── tests/                    # CI smoke tests (DRY_RUN / prepare run modes)
```

The roles of the pieces:

- **Component modules** (`learn.py`, `data.py`) hold the reusable
  {py:class}`~experimaestro.Config` and {py:class}`~experimaestro.Task` classes
  — the experimental building blocks. See
  [Configurations](config.md) and [Tasks](task.md).
- **`experiment.py`** is the orchestration layer: its `run(helper, cfg)`
  function composes those components into an
  [experimental plan](plan.md) and submits the tasks. `helper.xp` is the
  underlying {py:class}`~experimaestro.experiment` context (for
  `helper.xp.add_service(...)`, `helper.xp.add_action(...)`,
  `helper.xp.wait()`, …).
- **YAML files** (`params.yaml`) provide the values for the project's
  `Configuration` (a {py:class}`~experimaestro.experiments.ConfigurationBase`
  subclass) and the experiment metadata. Multiple YAML files can be layered with
  `imports:` or `--pre-yaml`/`--post-yaml`, and values overridden on the command
  line with `-c`.
- **`pyproject.toml`** declares the package and its dependencies (at least
  `experimaestro`, plus extras such as `experimaestro[carbon]`), and configures
  the tooling (ruff, mypy with the `experimaestro.mypy` plugin, pytest).
- **`xpm_settings.yaml`** is an optional, project-local copy of the
  [workspace settings](workspace.md): it declares *where* job outputs go and,
  via `triggers:`, which workspace is auto-selected for a given experiment id.
  Users copy it into `~/.config/experimaestro/settings.yaml`.
- **`actions.py` / `analyze.py`** handle what happens *after* the plan runs —
  see [Actions](actions.md) and [Analysing results](analysis.md).
- **`tests/`** run the plan in `DRY_RUN` (or `prepare`) mode so CI can validate
  the experiment wiring without launching real jobs.

## Workspaces and launchers

A project describes *what* to run; two pieces of per-user configuration describe
*where* it runs, and are intentionally kept outside the project:

- **Workspaces** (`~/.config/experimaestro/settings.yaml`, optionally seeded
  from the project's `xpm_settings.yaml`) decide where job directories are
  created. See [Workspaces](workspace.md) and [Settings](../settings.md).
- **Launchers** (`~/.config/experimaestro/launchers.py`) map a hardware
  requirement string such as `"duration=1h & gpu(mem=4G)*1 & cpu(cores=4)"` to a
  concrete way of running tasks (local, SLURM, OAR, …). Generate one with
  `experimaestro launchers direct generate` (single host) or
  `experimaestro launchers slurm generate` (cluster). See the
  [launchers documentation](../launchers/index.md).

Keeping these outside the project is what lets the *same* experiment code run
unchanged on a laptop and on a cluster.

## Building a domain-specific library

A project can also be a **library** that factors common experimental components
and a specialised experiment helper for a whole research domain, which other
projects then depend on. The pattern is to subclass
{py:class}`~experimaestro.experiments.ConfigurationBase` for domain-wide
settings and wrap `run-experiment` with a custom decorator/helper.

[experimaestro-ir (xpmir)](https://github.com/experimaestro/experimaestro-ir) is
the reference example. It provides:

- domain configuration bases such as `NeuralIRExperiment(ConfigurationBase)`
  (adding `gpu`, `seed`, a shared `random`, …);
- experiment decorators `@ir_experiment()` / `@learning_experiment()` and
  matching helpers (`IRExperimentHelper`, `LearningExperimentHelper`) that add
  evaluation, Tensorboard logging and model upload on top of the generic
  helper.

A downstream project then depends on `xpmir` and writes a much shorter
`experiment.py`:

```python
from experimaestro.experiments import configuration
from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper


@configuration
class Configuration:
    #: Default learning rate
    learning_rate: float = 1e-3


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: Configuration) -> PaperResults:
    ...
    return PaperResults(models=..., evaluations=..., tb_logs=...)
```

See xpmir's own [experiments documentation](https://experimaestro-ir.readthedocs.io/en/latest/experiments.html)
for the full helper API.

## See also

- [Tutorial](../tutorial.md) — a guided walk-through of the demo project.
- [Running experiments](../experiments.md) — the YAML configuration reference.
- [Experimental plan](plan.md) — composing and submitting tasks.
- [Workspaces](workspace.md) & [Settings](../settings.md) — where outputs go.
