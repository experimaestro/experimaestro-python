# --- Task and types definitions

import logging
import click
from experimaestro import *
from experimaestro.click import cli, CliRegisterTask

# --- Define the tasks

hw = Typename("helloworld")

@TypeArgument("word", type=str, required=True, help="Word to generate")
@CliRegisterTask(hw.say)
class Say:
    def execute(self):
        print(self.word.upper(),)

@TypeArgument("strings", type=ArrayOf(Say), help="Strings to concat")
@CliRegisterTask(hw.concat)
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

cli()

