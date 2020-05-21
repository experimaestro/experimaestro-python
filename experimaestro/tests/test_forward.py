from experimaestro import argument, config
from experimaestro.core.objects import Config
from experimaestro.click import cli, forwardoption
import click


@argument("epochs", type=int, default=100, help="Number of learning epochs")
@config("mymodel")
class MyModel(Config):
    pass


@forwardoption.epochs(MyModel, "my-epochs")
@cli.command()
def experiment(my_epochs):
    return my_epochs


def test_main():
    epochs = cli(["experiment", "--my-epochs", "100"], standalone_mode=False)
    assert epochs == 100


@forwardoption.epochs(MyModel)
@cli.command()
def experiment2(epochs):
    return epochs


def test_implicit_forward():
    epochs = cli(["experiment2", "--epochs", "100"], standalone_mode=False)
    assert epochs == 100
