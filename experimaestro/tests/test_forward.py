from experimaestro import argument, config, api
from experimaestro.click import cli, forwardoption
import click

@argument("epochs", type=int, default=100, help="Number of learning epochs")
@config("mymodel")
class MyModel(api.Config): pass

@forwardoption(MyModel.epochs, "my-epochs")
@cli.command()
def experiment(my_epochs):
    return(my_epochs)

def test_main():
    epochs = cli(["experiment", "--my-epochs", "100"], standalone_mode=False)
    assert epochs == 100



@forwardoption(MyModel.epochs)
@cli.command()
def experiment2(epochs):
    return(epochs)

def test_implicit_forward():
    epochs = cli(["experiment2", "--epochs", "100"], standalone_mode=False)
    assert epochs == 100
