from pathlib import Path

# Annotations
from .annotations import Type, Task, Argument, PathArgument, ConstantArgument, Typename, Array
from .workspace import Workspace
from .scheduler import Scheduler, experiment, progress
from .register import parse_commandline
from .meta import __version__

# Deprecated Annotation
from .annotations import RegisterType, RegisterTask, TypeArgument

def set_launcher(launcher):
    Workspace.CURRENT.launcher = launcher


# Signal handling

# import atexit
# import signal

# EXIT_MODE = False

# def handleKill(*args):
#     EXIT_MODE = True
#     logger.warning("Received SIGINT or SIGTERM")

# signal.signal(signal.SIGINT, handleKill)
# signal.signal(signal.SIGTERM, handleKill)
# signal.signal(signal.SIGQUIT, handleKill)

