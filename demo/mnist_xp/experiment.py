"""Author : Victor MORAND"""

import logging
import pandas as pd
from shutil import rmtree
from datamaestro import prepare_dataset
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, tags
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from .learn import CNN, Learn, Evaluate
from .data import mnist

logging.basicConfig(level=logging.INFO)


# Configuration of the whole experiment
# The values will be read from ./params.yaml
# and (optionally) the command line
@configuration
class Configuration(ConfigurationBase):
    # --- Model
    n_layers: list[int] = [3]  # number of Hidden layers
    hidden_dim: list[int] = [64]  # number of hidden units
    kernel_size: list[int] = [3]  # kernel size of the CNN

    # --- Training
    epochs: int = 5  # number of epochs to train the model
    n_val: int = 100  # number of steps between validation and logging
    lr: float = 1e-2  # learning rate
    batch_size: int = 32  # batch size

    # --- Misc
    launcher: str = """duration=3h & cuda(mem=4G)*1 & cpu(cores=2)"""


def run(helper: ExperimentHelper, cfg: Configuration):
    logging.debug(cfg)
    # Find a launcher to run our tasks given the given cfg.launcher
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])

    logging.info(f"Will Launch Tasks using launcher: {gpulauncher}")

    evaluations = []
    logging.info(
        "Experimaestro will launch tasks for " "each combination of parameters"
    )

    # This downloads the dataset if needed
    ds_mnist = prepare_dataset(mnist)

    # This path will contain all the tensorboard data
    runpath = (
        helper.xp.resultspath / "runs"
    )  # using pathlib.Path for cross-platform compatibility
    if runpath.is_dir():
        rmtree(runpath)
    runpath.mkdir(exist_ok=True, parents=True)

    # GridSearch: Launch a task for each combination of parameters
    for n_layer in cfg.n_layers:
        for hidden_dim in cfg.hidden_dim:
            for kernel_size in cfg.kernel_size:
                # Create a task with the given parameters
                model = CNN(
                    # Model params are 'tagged' for later monitoring
                    hidden_dim=tag(hidden_dim),
                    kernel_size=tag(kernel_size),
                    n_layers=tag(n_layer),
                )

                task = Learn(
                    # Defines the data and model used for training
                    data=ds_mnist.train,
                    model=model,
                    # Training params are not tagged
                    epochs=cfg.epochs,
                    n_val=cfg.n_val,
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                )

                # Submit the task
                loader = task.submit(launcher=gpulauncher)

                # Symlink so we can watch all this on tensorboard
                (runpath / tagspath(task)).symlink_to(task.runpath)

                # Evaluate the model on the test set
                evaluate = Evaluate(model=model, data=ds_mnist.test)
                evaluate.submit(init_tasks=[loader])
                evaluations.append(evaluate)

    # Wait that everything finishes
    helper.xp.wait()

    # OK, now we can look at the results
    dfs = []
    for evaluation in evaluations:
        df = pd.read_csv(evaluation.results_path)
        for key, value in tags(evaluation).items():
            df[key] = value
        dfs.append(df)

    df = pd.concat(dfs)
    print(df)  # noqa: T201
