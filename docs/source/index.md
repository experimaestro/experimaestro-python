# Experimaestro 

[![GitHub Release](https://img.shields.io/github/v/release/experimaestro/experimaestro-python)](https://github.com/experimaestro/experimaestro-python)

## Introduction

Experimaestro is a versatile tool for designing and managing complex workflows.
It enables the definition of tasks and their dependencies, ensuring orderly
execution within a workflow. Key features of Experimaestro include:

- **Task Automation and Caching**: Automates repetitive tasks, facilitating large-scale
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

## Guide to the Documentation

**🏁 Getting Started**
If you are new to the project, start with the [Tutorial](./tutorial.md). It walks you through setting up your first workspace and running a basic experiment: training a CNN on MNIST.

**🧪 Building Experiments**
Learn how to define your workflow:
  - [Configurations](./experiments/config.md): The heart of Experimaestro. Define parameters, nested structures, and value
    classes.
  - [Tasks](./experiments/task.md): Define the execution logic and manage dependencies.
  - Experimental [Plans](./experiments/plan.md): Compose tasks into complex matrices and track them using tags.

**⚙️ Execution & Infrastructure**
Control where and how your code runs:
  - [Launchers](./launchers/index.md): Manage execution environments (Direct, Slurm).
  - [Connectors](./connectors/index.md): Abstract file access and command execution (Local, SSH).

**🛠️ Advanced Tools**
  - [Jupyter Integration](./jupyter.md): Interact with your experiments from notebooks.
  - [API Reference](./api/index.md): Deep dive into the classes and methods.

# Documentation Outline

```{toctree}
---
maxdepth: 2
caption: "Experiments"
---
experiments/config
experiments/task
experiments/plan
experiments/workspace
experiments/actions
experiments/analysis
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
utilities
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

```{toctree}
---
maxdepth: 1
caption: "Development"
---
development/experiment
development/api
```


:::{note}
Experimaestro and datamaestro are described in the following paper

Benjamin Piwowarski. 2020.
[Experimaestro and Datamaestro: Experiment and Dataset Managers (for IR).](https://doi.org/10.1145/3397271.3401410)
*In Proceedings of the 43rd International ACM SIGIR*
:::
