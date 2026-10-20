# Guide to the documentation

Where to find what, depending on what you are trying to do.

## I want to…

| Goal | Page |
|------|------|
| Run my first experiment | [Tutorial](./tutorial.md) |
| Describe parameters and nested objects | [Configurations](./experiments/config.md) |
| Write the code of a task, and its dependencies | [Tasks](./experiments/task.md) |
| Compare many parameter combinations | [Plans](./experiments/plan.md) and [Grid search](./experiments/grid_search.md) |
| Launch an experiment, and do it right | [Running experiments](./experiments.md) |
| Drive experiments from `YAML` files | [The CLI experiment runner](./experiments/cli-runner.md) |
| Run on a cluster, or on another machine | [Launchers](./launchers/index.md) and [Connectors](./connectors/index.md) |
| Watch what is running | [Interfaces](./interfaces.md) |
| Share a workspace with colleagues | [Sharing a workspace](./experiments.md#sharing-a-workspace) |
| Collect and plot results | [Analysis](./experiments/analysis.md) |
| Understand what is written on disk | [Workspace layout](./experiments/workspace.md) |
| Look up a class or a command | [API reference](./api/index.md) and [CLI reference](./cli.md) |

## 🏁 Getting started

If you are new to the project, start with the [Tutorial](./tutorial.md). It
walks you through setting up your first workspace and running a basic
experiment: training a CNN on MNIST.

## 🧪 Building experiments

Learn how to define your workflow:

- [Configurations](./experiments/config.md): the heart of Experimaestro. Define
  parameters, nested structures, and value classes.
- [Tasks](./experiments/task.md): define the execution logic and manage
  dependencies — including resumable tasks and dynamic outputs.
- Experimental [Plans](./experiments/plan.md): compose tasks into complex
  matrices and track them using tags.
- [Experimaestro projects](./experiments/projects.md): how to lay out a project
  — the package, the `run(helper, cfg)` entry point, the YAML configuration, and
  the surrounding ecosystem.

## ▶️ Running and monitoring

Take an experiment from your editor to results:

- [Running experiments](./experiments.md): the two ways to launch an
  experiment, and the practices that keep results reusable — workspaces, shared
  workspaces, identifier stability, cluster runs, recovery.
- [The CLI experiment runner](./experiments/cli-runner.md): the `YAML`
  configuration format and `experimaestro run-experiment`.
- [Interfaces](./interfaces.md): follow experiments from the terminal UI, the
  web UI, or over SSH.
- [Actions](./experiments/actions.md) and [Analysis](./experiments/analysis.md):
  act on jobs, then collect and compare their results.
- [Workspace layout](./experiments/workspace.md): what experimaestro writes on
  disk, and how to read it.

## ⚙️ Execution and infrastructure

Control where and how your code runs:

- [Launchers](./launchers/index.md): manage execution environments (Direct,
  Slurm).
- [Connectors](./connectors/index.md): abstract file access and command
  execution (Local, SSH).
- [Settings](./settings.md): workspaces, auxiliary folders, remote workspaces,
  and lock file permissions.

## 🛠️ Advanced tools

- [Services](./services.md): expose long-running services from an experiment.
- [Serialization](./serialization.md): save and reload configurations.
- [Jupyter integration](./jupyter.md): interact with your experiments from
  notebooks.
- [Documenting](./documenting.md): document your own configurations.
- [Utilities](./utilities.md): miscellaneous helpers.

## 📚 Reference

- [API reference](./api/index.md): the classes and methods.
- [CLI reference](./cli.md): every `experimaestro` command.
- [FAQ](./faq.md) and [Changelog](./changelog.md).
