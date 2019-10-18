import unittest
from experimaestro import *
from experimaestro.click import cli, forwardoption
import click

@Argument("epochs", type=int, default=100, help="Number of learning epochs")
@Type("mymodel")
class MyModel: pass

@forwardoption(MyModel, "epochs")
@cli.command()
def experiment(epochs):
    return(epochs)


class MainTest(unittest.TestCase):
    def test_main(self):
        epochs = cli(["experiment", "--epochs", "100"], standalone_mode=False)
        self.assertEqual(epochs, 100)

if __name__ == '__main__':
    import sys
    logging.warn(sys.path)
    unittest.main()

