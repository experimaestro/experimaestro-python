import click
from experimaestro import parse_commandline, Environment

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
    `@forwardoption.ranker.optimizer.epsilon(MyConfig)` or
    `@forwardoption.ranker.optimizer.epsilon(MyConfig, "option-name")`
    """

    def __init__(self, path=[]):
        self.path = path

    def __call__(self, cls, option_name=None):
        argument = cls.__xpmtype__.arguments[self.path[0]]
        for c in self.path[1:]:
            argument = getattr(argument, c)

        name = "--%s" % (option_name or argument.name.replace("_", "-"))
        default = argument.default
        # TODO: set the type of the option when not a simple type
        return click.option(name, help=argument.help or "", default=default)

    def __getattr__(self, key):
        """Access to a class field"""
        return forwardoption([key])


def environment(name: str):
    def annotate(f):
        def callback_env(ctx, name, value):
            if value:
                assert name not in ctx.params, "Environment has already been set"
            else:
                return ctx.params.get(name, None)
            return Environment.get(value)

        def callback(ctx, param, value):
            if value:
                if name not in ctx.params:
                    ctx.params[name] = Environment()
                ctx.params[name].workdir = value

        f = click.option(
            f"--{name}-workdir",
            type=str,
            callback=callback,
            expose_value=False,
            help="Experimaestro environment",
        )(f)
        f = click.option(
            f"--{name}",
            type=str,
            callback=callback_env,
            help="Experimaestro environment",
        )(f)
        return f

    return annotate
