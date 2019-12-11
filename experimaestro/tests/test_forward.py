from experimaestro import Argument, Type, api
from experimaestro.click import cli, forwardoption
import click

@Argument("epochs", type=int, default=100, help="Number of learning epochs")
@Type("mymodel")
class MyModel(api.PyObject): pass

@forwardoption(MyModel.epochs, "epochs")
@cli.command()
def experiment(epochs):
    return(epochs)

def test_main():
    epochs = cli(["experiment", "--epochs", "100"], standalone_mode=False)
    assert epochs == 100
