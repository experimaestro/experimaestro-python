from experimaestro import argument, config
from experimaestro.click import forwardoption
import click


def test_main():
    @argument("epochs", type=int, default=100, help="Number of learning epochs")
    @config("mymodel")
    class MyModel:
        pass

    @forwardoption.epochs(MyModel)
    @click.command()
    def cli(epochs):
        return epochs

    epochs = cli(["--epochs", "100"], standalone_mode=False)
    assert epochs == 100


def test_rename():
    @argument("epochs", type=int, default=100, help="Number of learning epochs")
    @config("mymodel")
    class MyModel:
        pass

    @forwardoption.epochs(MyModel, "my-epochs")
    @click.command()
    def cli(my_epochs):
        return my_epochs

    epochs = cli(["--my-epochs", "100"], standalone_mode=False)
    assert epochs == 100
