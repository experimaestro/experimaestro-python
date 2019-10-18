[![CircleCI](https://circleci.com/gh/experimaestro/experimaestro-python.svg?style=svg)](https://circleci.com/gh/experimaestro/experimaestro-python)
[![PyPI version](https://badge.fury.io/py/experimaestro.svg)](https://badge.fury.io/py/experimaestro)

Experimaestro is a computer science experiment manager whose goals are:

* To decompose experiments into a set of parameterizable tasks
* Schedule tasks and handle dependencies between tasks
* Avoids to re-run the same task two times by computing unique task IDs dependending on the parameters
* Handle experimental parameters through tags

Full documentation can be found in [https://experimaestro.github.io/experimaestro-python/](https://experimaestro.github.io/experimaestro-python/)

# Install

Experimaestro depends on external libraries [Poco](https://pocoproject.org/) and [libssh](https://www.libssh.org/). 
You can install them using

- on Mac or Linux (user) `brew install poco libssh` 
- on Debian/Ubuntu Linux (root) `apt-get install libpoco-dev libssh-dev`

You can then install the package using `pip install experimaestro`

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
from experimaestro import *
from experimaestro.click import cli, TASK_PREFIX
import click

# --- Define the tasks

hw = Typename("helloworld")

@Type("word", type=str, required=True, help="Word to generate")
@RegisterTask(hw.say, prefix_args=TASK_PREFIX)
class Say:
    def execute(self):
        print(self.word.upper(),)

@Type("strings", type=ArrayOf(Say), help="Strings to concat")
@RegisterTask(hw.concat, prefix_args=TASK_PREFIX)
class Concat:
    def execute(self):
        # We access the file where standard output was stored
        says = []
        for string in self.strings:
            with open(string._stdout()) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))


# --- Defines the experiment

@click.argument("workdir", type=str)
@cli.command()
def xp(workdir):
    # Sets the working directory and the name of the xp
    experiment(workdir, "helloworld")

    # Submit the tasks
    hello = Say(word="hello").submit()
    world = Say(word="world").submit()

    # Concat will depend on the two first tasks
    Concat(strings=[hello, world]).submit()

```
