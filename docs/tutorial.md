# Introduction

This tutorial will illustrate the different components of Experimaestro using a simple experimental
project to illustrate the various aspects.

**...WORK IN PROGRESS...**

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

class Adam(Config):
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

- `Adam().__identifier__()` returns `261c5...`
- `Adam(lr=1e-3).__identifier__()` returns the same identifier since `lr` has a default value of `1e-3`
- `Adam(lr=1e-2).__identifier__()` returns `71848...` (different set of parameters)


# Tasks

When it comes to actually running code, Experimaestro allows to define
[tasks](./experiments/task.md) that are special kinds of configurations. The task defined below
allows to index a data collection retrieved from
Datamaestro, based on various experimental parameters (storePositions,
storeDocvectors, storeRawDocs, storeTransformedDocs and the collection
documents). It also defines parameters which do not change the outcome
but rather (1) the processing (e.g. threads) and are marked with an
ignored flag, (2) the output location on disk (e.g. index_path). In
both cases, the parameter value should be ignored when computing the
signature of the experiment. The method execute is called when the task
is effectively run, with the different parameters accessible through
self in the execute method.

```py3
class IndexCollection:
    """Index documents"""

    storePositions: Param[bool] = False
    storeDocvectors: Param[bool] = False
    storeRawDocs: Param[bool] = False
    storeTransformedDocs: Param[bool] = False
    documents: Param[Documents]

    # An option doesn't change the outcome, just the processing
    threads: Option[int] = 8

    # A path relative to the task directory
    index_path: Annotated[Path, pathgenerator("index")]

    def execute(self):
        # Calls java program and report progress
        pass
```


## Experiments


When configurations and tasks are defined, it is possible to assemble
them by defining an experimental plan. Contrarily to all the other
frameworks, Experimaestro has adopted an imperative style to define an
experiment. This makes it particularly easy to define complex
experimental plans. The code below shows a simple
but full experimental plan.

```py3
# Prepare the collection
random = Random()
wordembs = prepare_dataset("edu.stanford.glove.6b.50")
vocab = WordvecUnkVocab(data=wordembs, random=random)
robust = RobustDataset.prepare().submit()

# Train with OpenNIR DRMM model
ranker = Drmm(vocab=vocab).tag("ranker", "drmm")
predictor = Reranker()
trainer = PointwiseTrainer()
learner = Learner(trainer=trainer, random=random, ranker=ranker,
    valid_pred=predictor, train_dataset=robust.subset('trf1'),
    val_dataset=robust.subset('vaf1'), max_epoch=max_epoch)
model = learner.submit()

# Evaluate
Evaluate(dataset=robust.subset('f1'), model=model, predictor=predictor).submit()
```

The different tasks (whose definition is not
shown here) are used to perform various parts of the experiment:
(i) Word embeddings are downloaded and used to defined a vocabulary
(line 3-4); (ii) The robust collection index downloaded and
pre-processed for OpenNir (line 8); (iii) The DRMM is defined (l. 8) and
learned (l. 9-11) (iv) The learned model is evaluated on a held out set
(l. 15) Each task is submitted with .submit() (lines 5, 12 and 15), and
handled to a job scheduler that monitors and runs the tasks (on the
local machine, or in future versions through SSH or schedulers like OAR)
by running the execute() method.

Finally, while many parameters can have an effect on the process
outcome, only a subset of those are monitored during a typical
experiment. These are specially marked using tagging. In the code above,
one tag is used (line 8). These tags can be easily retrieved (e.g. when
generating the final report), and are also easily accessible when
interacting with the command line and web interfaces through a local
server which can be launched for any experiment.

### Unique task ID

Notice that there is no indication of the folder where tasks are run and
store results is given in the experimental plan, beside the location of
the main experiment directory (not shown here). This is one of the
strength of Experimaestro, i.e. the exact location is determined when a
task is submitted, and **is unique for a given set of experimental
parameters** – this allows to avoid running twice the same task and the
painful creation of unique folder names for each experiment (such as in
e.g. Capreolus or OpenNIR), which are error-prone and time-consuming.

When .submit() is called, Experimaestro automatically computes the task
byte string, and its signature. The identifier will be composed of the
task ID and of the identifier, e.g. ir.model.bm25/133778acb.... All the
artifacts generated by this task are contained within this folder (e.g.
the argument index_path), allowing easy task management (e.g. lookup
results, cleaning up old experiments, etc.).

# Launchers and connectors

# Configuring experiments
