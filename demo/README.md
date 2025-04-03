# Experimaestro demo

This tutorial aims at showcasing how to launch and monitor Python experiments
using [experimaestro](https://github.com/experimaestro/experimaestro-python).

We will use experimaestro in a basic setup performing hyper-parameters search
for a deep learning model (_here we will train CNN trained on the MNIST
dataset_).

In this situation, people usually have to run a lot of different trainings with
various hyper-parameters. This can rapidly become a pain to:
- Launch training jobs in parallel on your cluster, and monitor them in real time.
- Track the hyper-parameters already tested. And avoid launching a job with same parameters twice.
- Save results in proper _unique and not conflicting directories_ for further loading and analysis.

### Luckily, `experimaestro` can do all of that for you !

By the end of this tutorial, you will understand the core structure of
experimaestro. And you will be able to launch and monitor your own experiments.

## Installation

### First clone this repository

```bash
git clone https://github.com/experimaestro/experimaestro-demo.git
```

### Install the dependancies within your python environment

 You may first install pytorch along with the specific version of CUDA from [here](https://pytorch.org/get-started/locally/).
```bash
pip install -r requirements.txt
```

# The experiment structure

We will now have a closer look at the key files of this demo repository. In short, in the `mnist_xp` folder we have

- `learn.py` contains the core code, here a CNN configuration and the learning and evaluation tasks
- `data.py` contains the code that provides dataset
- `experiment.py` orchestrates the experiment
- `params.yaml`contains configuration parameters for our experiment

We will also point out the most important objects that allow us to run the experiment.

## `learn.py`

This file contains the code that defines a CNN model, and specifies how to learn
and evaluate an image classification model.

The most important concept at this stage is that of a **configuration** object,
that serves as a structured template to define parameters and settings for tasks
and experiments.

Let's see a first configuration that defines a CNN model

```py
class CNN(Config, nn.Module):
    """Defines a CNN model"""

    n_layers: Param[int] = 2
    """Number of Hidden layers"""

    hidden_dim: Param[int] = 64
    """Number of hidden units"""

    kernel_size: Param[int] = 3
    """Kernel size of the CNN"""
	...
```

A configuration is characterized by:

1. Deriving from the `experimaestro.Config` class
2. Defining **experimental parameters** (number of layer, hidden dimension, kernel size) that can **change** the outcome of an experiment.
3. Use docstring to document the parameters – which can be used to generate a
   documentation when the number of experimental configuration becomes high (e.g. [the documentation of a cross-scorer in the IR library](https://experimaestro-ir.readthedocs.io/en/latest/neural.html#xpmir.neural.cross.CrossScorer))

This configuration can latter be used as a dataclass when configuring the
experiments, e.g. `CNN(n_layers=3)`.For each instance of a configuration, we can
compute a **unique identifier** that changes only if **one or more experimental
parameter changes**. For instance, `CNN()` and `CNN(n_layers=2)` have the same
identifier, which is different from `CNN(n_layers=1)`.


Another important type of object are `Task` objects. They correspond to the code that can be run, e.g. on a SLURM cluster or locally, to perform a part of the experiment, e.g. learning the CNN model from data. Let us take a closer look at some bits of the code:

```py
class Learn(Task):
    """Learn to classify an image into a pre-defined set of classes"""

    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""

    data: Param[LabelledImages]
    """Train data are labelled images"""

    model: Param[CNN]
    """The model we are training"""

	...

	def execute(self):
		...
```

You can notice that tasks are specific types of configuration. You can also notice that
parameters can be other configurations, allowing to compose experimental components easily.

The main difference between `Config` and `Task` is the `execute` method. The
latter contain the code to be run when the task is run by the task scheduler.
Another important thing in the above example

Please have a look at the `Task`
[documentation](https://experimaestro-python.readthedocs.io/en/latest/experiments/task/)
for more details.


## Dataset handling: `data.py`

The first code in this file defines a labelled image dataset, which is a light
abstraction of the basic datasets we manipulate in our experiment. We use the
[datamaestro](https://datamaestro.readthedocs.io/en/latest/) library, which is
tightly associated with experimaestro, for this purpose.

```py
class LabelledImages(Base, ABC):
    @abstractmethod
    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        ...
```

The `datamaestro.Base` class is the central class that any dataset class should
derive from.

Leveraging [torchvision](https://pytorch.org/vision/stable/index.html), We then define the MNIST data with the following code:

```py
class MNISTLabelledImages(LabelledImages):
    root: Meta[Path]
    train: Param[bool]

    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        return MNIST(self.root, train=self.train, **kwargs)


def download_mnist(context: Context, root: Path, force=False):
    logging.info("Downloading in %s", root)
    for train in [False, True]:
        MNIST(root, train=train, download=True)


@custom_download("root", download_mnist)
@dataset(id="com.lecun.mnist")
def mnist(root: Path) -> Supervised[LabelledImages, None, LabelledImages]:
    """This corresponds to a dataset with an ID `com.lecun.mnist`"""
    return Supervised(
        train=MNISTLabelledImages(root=root, train=True),
        test=MNISTLabelledImages(root=root, train=False),
    )
```




**🚀 Going further** [datamaestro](https://datamaestro.readthedocs.io/en/latest/) can help managing many datasets by providing a unified interface to many datasets, as well as providing many utilities to download files from various sources. One such example is [datamaestro-image](https://github.com/experimaestro/datamaestro_image) that contains e.g. the MNIST dataset and [datamaestro-text](https://github.com/experimaestro/datamaestro_text) that contains NLP and IR datasets. Feel free to contribute!

<details>

<summary>👓 See how MNIST is defined in datamaestro-image</summary>

```py
from datamaestro_image.data import ImageClassification, LabelledImages, Base, IDXImage
from datamaestro.download.single import filedownloader
from datamaestro.definitions import dataset
from datamaestro.data.tensor import IDX


@filedownloader(
    "train_images.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
)
@filedownloader(
    "train_labels.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
)
@filedownloader(
    "test_images.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
)
@filedownloader(
    "test_labels.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
)
@dataset()
def mnist(train_images, train_labels, test_images, test_labels) -> ImageClassification:
    """The MNIST database

    The MNIST database of handwritten digits, available from this page, has a
    training set of 60,000 examples, and a test set of 10,000 examples. It is a
    subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image.
    """
    return ImageClassification(
        train=LabelledImages(
            images=IDXImage(path=train_images), labels=IDX(path=train_labels)
        ),
        test=LabelledImages(
            images=IDXImage(path=test_images), labels=IDX(path=test_labels)
        ),
    )
```

</details>


## `experiment.py`
This file describe the experiment, an experiment may load and launch several Tasks ..
There are two key elements in `experiment.py`:

### Configuration
The most important concept in Experimaestro is that of a configuration.
In Experimaestro, a configuration object is a fundamental concept used to specify parameters and settings for tasks and experiments.
It acts as a structured way to input the necessary details to execute a task or a series of tasks.

> 💡 **Have you noticed ?**
> Under the hood, the `Task` that we defined above contains a Configuration ! This enables experimaestro to create a unique Task ID for each set of `Params`.

Here, we first define a `ConfigurationBase` object that describes our experiment settings.
```python
from experimaestro import ConfigurationBase
...
@configuration
class Configuration(ConfigurationBase):
	# experimaestro Task parameters
	## Model
	n_layers: int = 3 # number of Hidden layers
	...
	# Misc
	launcher: str = """duration=3h & cuda(mem=4G)*1 & cpu(cores=2)"""
...
```
This class describes the configuration needed to run our experiment, The values for this configuration
Please have a look at the `Config` [documentation](https://experimaestro-python.readthedocs.io/en/latest/experiments/config/) for more details.

### Launchers
When operating with clusters using workload managers like [`Slurm`](https://slurm.schedmd.com/quickstart.html), experimaestro can manage the sumbission of your Tasks. This can be done easily with a [configuration file](https://experimaestro-python.readthedocs.io/en/latest/launchers/) (that specificies how to launch a task given some specifications), and the `find_launcher` function:

Here, we specified in the configuration above what hardware contraints we need for our Task: training a CNN on MNIST. We can then find a launcher with:
```python
...
gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])
```
You can have a look at the `Launchers` [documentation](https://experimaestro-python.readthedocs.io/en/latest/launchers/) for more details.

### Running Tasks
Now we are ready to launch our tasks !
we use the

```python
 for n_layer in cfg.n_layers:
        for hidden_dim in cfg.hidden_dim:
            for kernel_size in cfg.kernel_size:

                task = TrainOnMNIST(...)
				...task.submit(launcher=gpulauncher)... # Submit the Task to Slurm
```

## `params.yaml`
This file contains the values of the parameters that will be used to run our experiment.

```yaml
id: MNIST_train
title: "MNIST training"
description: "Training a simple CNN model on MNIST"
# what experiment file to run
file: experiment # will run experiment.py

# Launcher configuration: what resources we need to run a Task
launcher: "duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"

# Experimental parameters
hidden_dim:        # number of hidden units in the model
    - 32
n_layers:          # number of layers in the model
    - 1
    - 2
kernel_size:       # kernel size of the convolutions
    - 3

# Training parameters
epochs: 1       # number of epochs to train
lr: 1e-3        # learning rate
batch_size: 64  # batch size
n_val: 100      # number of steps between validation and logging
```
We will launch one job for each possible combination of `hidden_dim`,`n_layers` and `kernel_size`.

# Running the Experiment
Now that everything is set up, we can run our experiment
```bash
experimaestro run-experiment debug.yaml
```

Now experimaestro will:
- Lock the experiment `MNIST_train` so that you cannot relaunch it while this one is running.
- Run `experiment.py` with the configuration values read in `params.yaml`
- For each Task submitted in the experiment:
	- A unique hash ID is created depending on the parameters given to the task.
		- This ensures that you don't run the Task with the same params twice.
	- A folder is created in the `workspace/jobs/task-id` , it will be the working directory for the Task
		- here you can find the logs and the outputs of the running Task

## Monitoring your jobs
Your jobs are now launched. You can display all launched jobs with the following command:
```bash
experimaestro jobs list --tags
```
it yields:
```bash
RUNNING    task.trainonmnist/12feeb6cb9f5f7aad5d0fdcaac5ee057673f7c82485126d8903ecc119b567451 MNIST_train n_layers=1 hidden_dim=32 kernel_size=3
RUNNING    task.trainonmnist/c9420a1de91830ff467397bd3e1aa592535eac931c9dff7efbad7c0e759c0be3 MNIST_train n_layers=2 hidden_dim=32 kernel_size=3
```

> 💡 **Bonus**:
> For better readability, we used `tags` when creating our Tasks, and used the `--tag` flag in the command above. This is why the parameters are diplayed above. See the [docs](https://experimaestro-python.readthedocs.io/en/latest/experiments/plan/#tags) for more details.

## TODOs
Installation:
- [ ] Setup the workspace directory, default install ??
- [ ] Add details for results processing.
- [ ] Detail how to install the launcher file as well.

Don't know what is the best choice here
