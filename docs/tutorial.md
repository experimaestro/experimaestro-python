# Introduction

This tutorial will illustrate the different components of Experimaestro using a simple experimental
project to illustrate the various aspects.

# Installation

First install the package using

```sh
pip install experimaestro
```

# Configurations

A configuration object in Experimaestro serves as a structured template to define parameters and settings for tasks and experiments. Key aspects include:

- **Parameter Definition**: Specifies essential inputs needed for task execution, like file paths, numerical values, etc.
- **Types and Validation**: Ensures parameters are in the correct format with type specifications and validation rules.
- **Default Values**: Provides default settings for optional or commonly used parameters.
- **Documentation**: Includes explanations for each parameter, aiding in user understanding and usability.
- **Hierarchy and Nesting**: Allows organization of parameters in a structured manner, especially useful in complex tasks.
- **Task Linking**: Directly associated with specific tasks or experiments to provide necessary inputs.
- **Flexibility and Extensibility**: Adaptable to changing requirements by allowing modifications and additions.
- **Serialization**: Can be saved and loaded for sharing.

In essence, configuration objects in Experimaestro facilitate the automation, and reproducibility of experiments by providing a detailed and validated framework for task parameters.

An example of a configuration of an optimizer in machine learning:

```py3
from experimaestro import Config, Param

class Optimizer(Config):
    @abstractmethod
    def__call__(self, parameters):
        ...

class Adam(Optimizer):
    """Wrapper for Adam optimizer"""

    lr: Param[float] = 1e-3
    """Learning rate"""

    weight_decay: Param[float] = 0.0
    """Weight decay (L2)"""

    eps: Param[float] = 1e-8

    def __call__(self, parameters):
        # Returns an optimizer for the given parameters
        ...
```

## Configuration identifiers


Experimaestro has an automated process that generates a unique signature for
each configuration depending on experimental parameters – this idea is used for
instance in [PlanOut](https://github.com/facebookarchive/planout) to uniquely
identify the system parameters in A/B testing. This identifier plays a crucial
in identifying a unique set of parameters. Here's a detailed description:

1. **Uniqueness**: A configuration identifier is unique for each configuration
   instance. This uniqueness ensures that each configuration can be distinctly
   identified and referenced, avoiding confusion or overlap with other
   configurations.

1. **MD5 Hashes**: Experimaestro utilizes MD5 hashes as configuration
    identifiers. These hashes are unique to each configuration, ensuring a
    distinct and consistent identifier for every set of parameters.

1. **Run-Once Guarantee**: The unique MD5 hash identifiers ensure that each task
   associated with a specific configuration is executed only once. This is
   particularly important in avoiding redundant computations and ensuring the
   efficiency of the workflow.

Taking the configuration class `Adam` defined above, we have:

- `Adam.C().__identifier__()` returns `261c5...`
- `Adam.C(lr=1e-3).__identifier__()` returns the same identifier since `lr` has a default value of `1e-3`
- `Adam.C(lr=1e-2).__identifier__()` returns `71848...` (different set of parameters)


# Tasks

When it comes to actually running code, experimaestro allows to define
[tasks](./experiments/task.md) that are special kinds of configurations. The
task defined below learns a model. It also defines parameters which do not
 change the outcome but rather (1) the processing (e.g. number of GPUs to use)
and are marked as `Meta`, (2) the output location on disk (e.g. index_path). In
both cases, the parameter value should be ignored when computing the signature
of the experiment. The method execute is called when the task is effectively
run, with the different parameters accessible through self in the execute
method.


```py3
class LearnedModel(Config):
    model: Param[Model]
    """The model"""

    path: Param[Path]
    """The path to the serialized parameters"""

    @cached_property
    def instance(self):
        """Returns the model with the learned parameters"""
        self.model.load(path)
        return self.model

class Learn(Task):
    """Index documents"""

    model: Param[Model]
    """The model to use"""

    data: Param[Dataset]
    """The dataset to use"""

    optimizer: Param[Optimizer]
    """The optimizer"""

    epochs: Param[int]
    """Number of epochs"""

    gpus: Meta[int] = 2
    """Number of GPUs to use (note the `Meta`)"""

    model_path: Meta[Path] = field(default_factory=PathGenerator("model.pth"))
    """A path relative to the task directory"""


    def execute(self):
        # Learns and save the model in self.model_path
        ...

    def task_outputs(self, dep: Callable):
        """Output of this task when submitted

        :param dep: A function that marks any configuration object as a dependency
        """
        # Construct the returned configuration object
        learned_model = LearnedModel(model=model, path=self.model_path)

        # The learned model is an dependent on this task, so we use dep
        return dep(learned_model)
```

# Launchers and connectors

When running experiments, it might be useful to specify the material constraint
– especially when running on clusters like slurm. This can be done easily with a
[configuration file](./launchers/index.md) (that specificies how to launch a
task given some specifications), and the `find_launcher` function:

```py3
from experimaestro.launcherfinder import find_launcher

learn_launcher = find_launcher(
    """duration=2 days & cuda(mem=16G) * 4  & cpu(mem=400M, cores=4)"""
    """ | duration=4 days & cuda(mem=16G) * 2  & cpu(mem=400M, cores=4)"""
)
evaluation_launcher = find_launcher(
    """duration=6 hours & cuda(mem=16G) * 4 & cpu(mem=2G, cores=16)"""
)
```

# Experiments


When configurations and tasks are defined, it is possible to assemble
them by defining an experimental plan. Contrarily to all the other
frameworks, Experimaestro has adopted an imperative style to define an
experiment. This makes it particularly easy to define complex
experimental plans. The code below shows a simple
but full experimental plan.

Let start with the experimental file `experiment.py` that
describes the experiment:

```py3

from experimaestro.experiments import ExperimentHelper, configuration

@configuration
class Configuration:
    epochs: int
    n_layers: List[int]

def run(
    helper: ExperimentHelper, cfg: Configuration
):
    # Experimental code
    optimizer = Adam.C(lr=1e-4)
    dataset = MyDataset.C()
    models = [AwesomeModel.C(layers=tag(n_layer)) for n_layer in cfg.n_layers]
    learned = {}

    for model in models:
        # Learn the model
        learner = Learner.C(optimizer=optimizer, dataset=dataset.train, model=model)
        learned_model = learner.submit(launcher=learn_launcher, epochs=cfg.epochs)

        # Keeps track of the learned models
        # e.g. here tagspath returns `f"layers={n_layer}"`
        learned[tagspath(learned_model)] = learned_model

        # and evaluate (another task, not shown here)
        Evaluate.C(dataset=dataset.test, model=learned_model).submit(launcher=evaluation_launcher)
```

With `debug.yaml` located in the same folder as `experiment.py`

```yaml
    # Uses experiment.py
    file: experiment

    # Just debugging
    epochs: 16

    # Experimental parameters
    n_layers: [3, 5]
```

The experiment can be started with

```sh
    experimaestro run-experiment --run-mode dry-run debug.yaml
```

The run mode controls the experiment: `dry-run` is used to just test that the
script runs until the end, `generate` generates job directories, and `normal`
launches the jobs.

The `ExperimentHelper` API is described in [this document](./experiments.md).

Finally, while many parameters can have an effect on the process outcome, only a
subset of those are monitored during a typical experiment. These are specially
marked using tagging with the `tag` function. In the code above, one tag is
used. These tags can be easily retrieved (e.g. when generating the final
report), and are also easily accessible when interacting with the command line
and web interfaces.

## Unique task ID

Notice that there is no indication of the folder where tasks are run and store
results is given in the experimental plan, beside the location of the main
experiment directory (not shown here). This is one of the strength of
Experimaestro, i.e. the exact location is determined when a task is submitted,
and **is unique for a given set of experimental parameters** – this allows to
avoid running twice the same task and the painful creation of unique folder
names for each experiment, which are error-prone and time-consuming.

When `.submit()` is called, Experimaestro automatically computes the task byte
string, and its signature. The identifier will be composed of the task ID and of
the identifier, e.g. `my.module.learner/133778acb`.... All the artifacts
generated by this task are contained within this folder (e.g. the argument
model_path), allowing easy task management (e.g. lookup results, cleaning up old
experiments, etc.).
