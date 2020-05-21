import click
from experimaestro import parse_commandline

"""Defines the task command line argument prefix for experimaestro-handled command lines"""




@click.group()
def cli():
    """Main entry point for CLI"""
    pass


@cli.command(context_settings={"allow_extra_args": True})
@click.pass_context
def xpm(context):
    """Command used to run a task"""
    parse_commandline(context)




class forwardoptionMetaclass(type):
  def __getattr__(self, key):
        """Access to a class field"""
        return forwardoption([key])

class forwardoption(metaclass=forwardoptionMetaclass):
    """Allows to access an argument of the configuration

    This allows to refer to a path of a class in a "python" syntax, e.g.
    `@forwardoption.ranker.optimizer.epsilon(MyConfig)` or `@forwardoption.ranker.optimizer.epsilon(MyConfig, "option-name")`
    """
    def __init__(self, path=[]):
        self.path = path

    def __call__(self, cls, option_name=None):
        argument = cls.__xpm__.arguments[self.path[0]]
        for c in self.path[1:]:
            argument = getattr(current, c)

        xpmtype = argument.type
        name = "--%s" % (option_name or argument.name.replace("_", "-"))
        default = argument.default
        # TODO: set the type of the option when not a simple type
        return click.option(name, help=argument.help or "", default=default)

    def __getattr__(self, key):
        """Access to a class field"""
        return forwardoption([key])
