# --- Task and types definitions

import logging
from experimaestro import *
from experimaestro.click import cli, TASK_PREFIX
import click

# --- Define the tasks

hw = Typename("helloworld")

@Argument("word", type=str, required=True, help="Word to generate")
@Task(hw.say, prefix_args=TASK_PREFIX)
class Say:
    def execute(self):
        print(self.word.upper(),)

@Argument("strings", type=Array(Say), help="Strings to concat")
@Task(hw.concat, prefix_args=TASK_PREFIX)
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

