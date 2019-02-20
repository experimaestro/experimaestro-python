import click
from experimaestro import register
from experimaestro import RegisterTask, Type, Value

class CliRegisterTask(RegisterTask):
    """Registers a task when the tasks and main command line are in the same file"""
    def __init__(self, *args, **kwargs):
        kwargs["prefix_args"] = ["xpm", "--"]
        super().__init__(*args, **kwargs)

@click.group()
def cli():
    """Main entry point for CLI"""
    pass

@cli.command(context_settings={"allow_extra_args": True})
@click.pass_context
def xpm(context):
    """Command used to run a task"""
    register.parse(context.args)


def forwardoption(xpmtype, option_name):
    """Helper function
    
    Arguments:
        xpmtype {class or typename} -- The experimaestro type name or a class corresponding to a type
        option_name {str} -- The name of the option
    
    Raises:
        Exception -- Raised if the option `option_name` does not exist in the type, or if the type is not defined
    
    Returns:
        click.option -- Returns a click option annotation
    """

    xpmtype = register.getType(xpmtype)
    a = xpmtype.getArgument(option_name)
    if a is None:
        raise Exception("No argument with name %s in %s" % (option_name, xpmtype))

    name = "--%s" % a.name.replace("_", "-")
    
    ptype = Type.topython(a.type)
    default = Value.toPython(a.defaultvalue) if a.defaultvalue else None
    return click.option(name, help=a.help, type=ptype, default=default)