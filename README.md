[![PyPI version](https://badge.fury.io/py/experimaestro.svg)](https://badge.fury.io/py/experimaestro)
[![RTD](https://readthedocs.org/projects/experimaestro-python/badge/?version=latest)](https://experimaestro-python.readthedocs.io)

Experimaestro helps in designing and managing complex workflows. It allows for the definition of tasks and their dependencies, ensuring that each step in a workflow is executed in the correct order. Some key aspects of Experimaestro are:

- **Task Automation**: The tool automates repetitive tasks, making it easier to run large-scale experiments. It's particularly useful in scenarios where experiments need to be repeated with different parameters or datasets.
- **Resource Management**: It efficiently manages computational resources, which is critical when dealing with data-intensive tasks or when running multiple experiments in parallel.
- **Extensibility**: Experimaestro is designed to be flexible and extensible, allowing users to integrate it with various programming languages and tools commonly used in data science and research.
- **Reproducibility**: By keeping a detailed record of experiments, including parameters and environments, it aids in ensuring the reproducibility of scientific experiments, which is a fundamental requirement in research.
- **User Interface**: While primarily a back-end tool, Experimaestro also offers a user interface to help in managing and visualizing workflows.

The full documentation can be read by going to the following URL: [https://experimaestro-python.readthedocs.io](https://experimaestro-python.readthedocs.io)


# Install

## With pip

You can then install the package using `pip install experimaestro`

## Develop

Checkout the git directory, then

```
pip install -e .
```

# Example

This very simple example shows how to submit two tasks that concatenate two strings.
Under the curtain,

- A directory is created for each task (in `workdir/jobs/helloworld.add/HASHID`)
  based on a unique ID computed from the parameters
- Two processes for `Say` are launched (there are no dependencies, so they will be run in parallel)
- A tag `y` is created for the main task

<!-- SNIPPET: MAIN ARGS[%WORKDIR% --port 0 --sleeptime=0.0001] -->

```python
# --- Task and types definitions

import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path
from experimaestro import Task, Param, experiment, progress
import click
import time
import os
from typing import List

# --- Just to be able to monitor the tasks

def slowdown(sleeptime: int, N: int):
    logging.info("Sleeping %ds after each step", sleeptime)
    for i in range(N):
        time.sleep(sleeptime)
        progress((i+1)/N)


# --- Define the tasks

class Say(Task):
    word: Param[str]
    sleeptime: Param[float]

    def execute(self):
        slowdown(self.sleeptime, len(self.word))
        print(self.word.upper(),)

class Concat(Task):
    strings: Param[List[Say]]
    sleeptime: Param[float]

    def execute(self):
        says = []
        slowdown(self.sleeptime, len(self.strings))
        for string in self.strings:
            with open(string.__xpm_stdout__) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))


# --- Defines the experiment

@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.option("--sleeptime", type=float, default=2, help="Sleep time")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, sleeptime):
    """Runs an experiment"""
    # Sets the working directory and the name of the xp
    with experiment(workdir, "helloworld", port=port) as xp:
        # Submit the tasks
        hello = Say(word="hello", sleeptime=sleeptime).submit()
        world = Say(word="world", sleeptime=sleeptime).submit()

        # Concat will depend on the two first tasks
        Concat(strings=[hello, world], sleeptime=sleeptime).tag("y", 1).submit()


if __name__ == "__main__":
    cli()
```

which can be launched with `python test.py /tmp/helloworld-workdir`
