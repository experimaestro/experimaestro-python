from experimaestro import Param, Config
from experimaestro.click import forwardoption
import click


def test_main():
    class MyModel(Config):
        epochs: Param[int] = 100
        """Number of learning epochs"""

    @forwardoption.epochs(MyModel)
    @click.command()
    def cli(epochs):
        return epochs

    epochs = cli(["--epochs", "100"], standalone_mode=False)
    assert epochs == 100


def test_rename():
    class MyModel(Config):
        epochs: Param[int] = 100
        """Number of learning epochs"""

    @forwardoption.epochs(MyModel, "my-epochs")
    @click.command()
    def cli(my_epochs):
        return my_epochs

    epochs = cli(["--my-epochs", "100"], standalone_mode=False)
    assert epochs == 100
