# Overview

Types and tasks are the main components of an experimental plan.
Defining types and tasks is akin to defining structures in any
strongly-typed programming language. Data types can be either simple
(real, integer, boolean, string or path) or complex (arrays or
dictionaries).

## Configurations

[Configurations](../config) and tasks are defined as Python objects,
and types have properties such as a default value or an ignored
flag which are useful when computing the signature of an experiment.

The example below a is
a simple configuration for the BM25 model, defining two parameters (k1
and b) with their default values. It has a type identified a unique ID
(e.g. ir.model.bm25).

```py3
class Model(Config):
    pass

class BM25(Model):
    k1: Param[float] = 0.9
    b: Param[float] = 0.4
```

Configurations are used as... configuration units (e.g. a dataset from
[Datamaestro](), a stemmer configuration, or the optimizer to use for a
gradient descent).

## Tasks

When it comes to actually running code, Experimaestro allows to define
[tasks](../task) that are special kinds of configurations. The task defined below
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
    documents: Param[AdhocDocuments]

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

Experimaestro has a automated process that generates a unique signature
for each task depending on experimental parameters – this idea is used
for instance in [PlanOut]() to uniquely identify the system
parameters in A/B testing. First, any value can be associated with a
unique byte string: the byte string is obtained by outputting the type
of the value (e.g. string, ir.adhoc.dataset) and the value itself as a
binary string. A special handling of configurations
and tasks (objects) is performed by sorting keys in ascending lexicographic
order, thus ensuring the uniqueness of the representation. Moreover,

- Default values are removed (e.g. k1 when set to 0.9). This allows to
  handle the situation where one adds a new experimental parameter
  (e.g. a new loss component). In that case, using a default parameter
  allows to add this parameter without invalidating all the previously
  ran experiments.
- Ignored values are removed (e.g. the number of threads when
  indexing, the path where the index is stored)

When .submit() is called, Experimaestro automatically computes the task
byte string, and its signature. The identifier will be composed of the
task ID and of the identifier, e.g. ir.model.bm25/133778acb.... All the
artifacts generated by this task are contained within this folder (e.g.
the argument index_path), allowing easy task management (e.g. lookup
results, cleaning up old experiments, etc.).
