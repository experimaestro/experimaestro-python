# Walkthrough

The [experimaestro-demo](https://github.com/experimaestro/experimaestro-demo)
repository is a complete working example: a hyper-parameter search for a CNN
on MNIST. Its [README](https://github.com/experimaestro/experimaestro-demo#readme)
is the step-by-step tutorial; this page summarises what it covers.

```bash
git clone https://github.com/experimaestro/experimaestro-demo.git
cd experimaestro-demo
```

## Setup

- **Workspace**: where the job folders are created. The demo ships a
  `settings.yaml` whose workspace uses `triggers:` (here `"MNIST_*"`), so the
  `MNIST_train` experiment picks it without any `--workspace` / `--workdir`
  flag. See [Settings](settings.md).
- **Launchers** (optional for this small experiment): how tasks are run.
  `experimaestro launchers direct generate` writes a `launchers.py` for the
  local machine (CPU, CUDA, MPS, with memory tokens);
  `experimaestro launchers slurm generate` does the same for a SLURM cluster.
  See [Launchers](launchers/index.md).

## The experiment

The code lives in `mnist_xp/`:

- `learn.py` defines the **configurations** and **tasks**. `CNN` is a
  `Config` whose `Param`s (number of layers, hidden dimension, kernel size)
  determine its identifier; `Learn` and `Evaluate` are `Task`s, whose
  `execute()` method is what the scheduler runs, and whose parameters can be
  other configurations (the model, the data).
  See [Configurations](experiments/config.md) and [Tasks](experiments/task.md).
- `data.py` describes the MNIST dataset with
  [datamaestro](https://datamaestro.readthedocs.io/en/latest/).
- `experiment.py` builds the plan: it reads the configuration, picks a
  launcher from a requirement string (`find_launcher("duration=1h &
  cuda(mem=4G)*1 & cpu(cores=4)")`) and submits one `Learn` task (then its
  evaluation) per combination of hyper-parameters.
- `params.yaml` holds the experiment ID, the launcher requirements and the
  hyper-parameter values of the grid.

## Running it

```bash
# List what would run (done / not done / failed), and with which launcher
uv run experimaestro run mnist_xp/params.yaml --run-mode dry_run

# Run it, monitoring with the terminal UI (or --web)
uv run experimaestro run mnist_xp/params.yaml --console
```

Each task gets a folder `<workspace>/jobs/<task id>/<hash>/`, the hash being
computed from its parameters: a task already run with the same parameters is
not run again. The experiment can also be followed afterwards with
`experimaestro monitor --console` (or `--web`).

The demo's `ADVANCED.md` goes on with declarative grid search (`GridSearch`),
post-experiment actions (exporting the best model) and task dependency graphs.

## Going further

These features extend the demo and are worth knowing once the basic flow above is comfortable:

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
