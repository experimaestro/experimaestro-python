import click
from experimaestro import *
from experimaestro.click import cli, forwardoption


@TypeArgument("epochs", type=int, default=100, help="Number of learning epochs")
@RegisterType("mymodel")
class MyModel: pass

@forwardoption(MyModel, "epochs")
@cli.command()
def experiment(epochs):
    print(epochs)

cli()
