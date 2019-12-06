import click
import logging

# --- Command line main options

logging.basicConfig(level=logging.INFO)

class Config: 
    def __init__(self):
        self.traceback = False


def pass_cfg(f):
    """Pass configuration information"""
    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj, *args, **kwargs)
    return update_wrapper(new_func, f)


@click.group()
@click.option("--quiet", is_flag=True, help="Be quiet")
@click.option("--debug", is_flag=True, help="Be even more verbose (implies traceback)")
@click.option("--traceback", is_flag=True, help="Display traceback if an exception occurs")
@click.pass_context
def cli(ctx, quiet, debug, traceback, data, keep_downloads):
    if quiet:
        logging.getLogger().setLevel(logging.WARN)
    elif debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ctx.obj = Config()
    ctx.obj.traceback = traceback

def main():
    cli(obj=None)


