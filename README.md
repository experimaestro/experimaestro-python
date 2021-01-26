[![PyPI version](https://badge.fury.io/py/experimaestro.svg)](https://badge.fury.io/py/experimaestro)

Experimaestro is a computer science experiment manager whose goals are:

- To decompose experiments into a set of parameterizable tasks
- Schedule tasks and handle dependencies between tasks
- Avoids to re-run the same task two times by computing unique task IDs dependending on the parameters
- Handle experimental parameters through tags

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

<!-- SNIPPET: MAIN ARGS[%WORKDIR% --port 0] ENV[XPM_EXAMPLE_WAIT=0.001] -->

```python
# --- Task and types definitions

import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path
from experimaestro import *
import click
import time
import os
from typing import List

# --- Just to be able to monitor the tasks

def slowdown(N: int):
    for i in range(N):
        time.sleep(float(os.environ.get("XPM_EXAMPLE_WAIT", 2)))
        progress((i+1)/N)


# --- Define the tasks

hw = Identifier("helloworld")

@task(hw.say)
class Say:
    word: Param[str]

    def execute(self):
        slowdown(len(self.word))
        print(self.word.upper(),)

@task(hw.concat)
class Concat:
    strings: Param[List[Say]]

    def execute(self):
        says = []
        slowdown(len(self.strings))
        for string in self.strings:
            with open(string.__xpm_stdout__) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))


# --- Defines the experiment

@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir):
    """Runs an experiment"""
    # Sets the working directory and the name of the xp
    with experiment(workdir, "helloworld", port=port) as xp:
        # Submit the tasks
        hello = Say(word="hello").submit()
        world = Say(word="world").submit()

        # Concat will depend on the two first tasks
        Concat(strings=[hello, world]).tag("y", 1).submit()


if __name__ == "__main__":
    cli()
```

which can be launched with `python test.py /tmp/helloworld-workdir`
