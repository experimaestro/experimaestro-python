# Introduction

```{toctree}
---
maxdepth: 1
caption: "Getting Started"
---
tutorial
```

```{toctree}
---
maxdepth: 2
caption: "Experiments"
---
experiments/config
experiments/task
experiments/plan
experiments/workspace
experiments
```

```{toctree}
---
maxdepth: 1
caption: "Execution"
---
launchers/index
connectors/index
```

```{toctree}
---
maxdepth: 1
caption: "Misc"
---
serialization
settings
services
interfaces
```

```{toctree}
---
maxdepth: 1
caption: "Integration"
---
jupyter
documenting
```

```{toctree}
---
maxdepth: 1
caption: "Reference"
---
api/index
cli
changelog
faq
```

Experimaestro is a versatile tool for designing and managing complex workflows.
It enables the definition of tasks and their dependencies, ensuring orderly
execution within a workflow. Key features of Experimaestro include:

- **Task Automation**: Automates repetitive tasks, facilitating large-scale
  experiments, especially useful when varying parameters or datasets.
- **Extensibility**: Designed for flexibility, Experimaestro can easily be integrated with
  existing libraries, serving the diverse needs of data science and research.
- **Reproducibility**: Maintains comprehensive records of experiments, including
  parameters and environments, supporting the essential research principle of
  reproducibility.
- **User Interface**: Offers a user interface for workflow management and
  visualization, complementing its primary back-end functionality.

## Difference with Other Projects

Experimaestro differentiates itself from traditional job scheduling software
like [OAR](https://oar.imag.fr) and [Slurm](https://slurm.schedmd.com), which
focus more on resource allocation than on managing experimental workflows. It
also stands apart from other experiment management tools like
[Comet](https://www.comet.ml), [Sacred](https://github.com/IDSIA/sacred),
[FGLab](https://github.com/Kaixhin/FGLab), and
[Sumatra](http://neuralensemble.org/sumatra/). For instance, Comet emphasizes
collaboration and note-taking for machine learning experiments but is not
open-source and focuses on single-shot experiments. Sumatra and FGLab, based on
parameter files, offer less flexibility. Sacred, though open-source and allowing
for pre-processing steps, doesn't support the construction of complex
experimental plans like Experimaestro.

Experimaestro's distinct features include:

1. **Comprehensive Task Composition**: It allows for the composition of types
   and tasks within an experimental plan.
2. **Parameter Monitoring**: Offers a clear method to monitor experimental
   parameters using tags.
3. **Automated Output Organization**: Efficiently manages task outputs in the
   file system, simplifying result storage.
4. **Imperative Experiment Definition**: Unlike other tools that define
   experiments declaratively, Experimaestro adopts an imperative approach,
   enhancing flexibility in complex experimental planning.

## Outline of the documentation

### Tutorial

You can follow the [tutorial](./tutorial.md)

### Experimental plan

An experimental plan is based on [configuration and tasks](./experiments/config.md), and define which tasks should be run with which parameters. Within an experiment, [tags](./experiments/plan.md#tags) can be used to track experimental parameters.

### Connectors

A [connector](./connectors/index.md) allow to specify how to access files on the computer where a task will be launched, and how to run processes on this computer. Two basic connectors exist, for localhost and SSH accesses (_alpha_).

### Launcher

A [launcher](./launchers/index.md) specifies how a given task can be run. The most basic method is direct execution, but experimaestro can launch and monitor oar (_planned_) and slurm (_planned_) jobs.


:::{note}
Experimaestro and datamaestro are described in the following paper

Benjamin Piwowarski. 2020.
[Experimaestro and Datamaestro: Experiment and Dataset Managers (for IR).](https://doi.org/10.1145/3397271.3401410)
*In Proceedings of the 43rd International ACM SIGIR*
:::
