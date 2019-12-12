[![CircleCI](https://circleci.com/gh/experimaestro/experimaestro-python.svg?style=svg)](https://circleci.com/gh/experimaestro/experimaestro-python)
[![PyPI version](https://badge.fury.io/py/experimaestro.svg)](https://badge.fury.io/py/experimaestro)

Experimaestro is a computer science experiment manager whose goals are:

* To decompose experiments into a set of parameterizable tasks
* Schedule tasks and handle dependencies between tasks
* Avoids to re-run the same task two times by computing unique task IDs dependending on the parameters
* Handle experimental parameters through tags

Full documentation can be found in [https://experimaestro.github.io/experimaestro-python/](https://experimaestro.github.io/experimaestro-python/)

# Install

## With pip

You can then install the package using `pip install experimaestro`

## Develop

Checkout the git directory, then

```
pip install -e .
```

# Example

This very simple example shows how to submit two tasks that add two numbers.
Under the curtain, 

- A directory is created for each task (in `workdir/jobs/helloworld.add`)
  based on a unique ID computed from the parameters
- Two processes are launched (there are no dependencies, so they will be run in parallel)
- A tag `y` is created for each task - tags are experimental parameters you vary in your experiments,
  so you can easily keep track of them


```python
# --- Task and types definitions

import logging
from pathlib import Path
from experimaestro import *
import click
import time
from typing import List

# --- Just to be able to monitor the tasks

def slowdown(N: int):
    for i in range(N):
        time.sleep(2)
        progress((i+1)/N)


# --- Define the tasks

hw = Typename("helloworld")

@argument("word", type=str, required=True, help="Word to generate")
@task(hw.say)
def Say(word: str):
    slowdown(len(word))
    print(word.upper(),)

@argument("strings", type=List[Say], help="Strings to concat")
@task(hw.concat)
def Concat(strings: List[Say]):
    # We access the file where standard output was stored
    says = []
    slowdown(len(strings))
    for string in strings:
        with open(string.stdout()) as fp:
            says.append(fp.read().strip())
    print(" ".join(says))


# --- Defines the experiment

@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, debug):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Sets the working directory and the name of the xp
    with experiment(workdir, "helloworld", port=12345) as xp:
        # Submit the tasks
        hello = Say(word="hello").submit()
        world = Say(word="world").submit()

        # Concat will depend on the two first tasks
        Concat(strings=[hello, world]).submit()


if __name__ == "__main__":
    cli()
```

which can be launched with `python test.py xp /tmp/helloworld-workdir`