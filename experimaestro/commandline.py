"""Command line jobs"""

import os
from pathlib import Path
from .scheduler import Job, JobError
class AbstractCommandComponent(): pass

class CommandContext: pass

class CommandPath(AbstractCommandComponent):
    def __init__(self, path: [Path, str]):
        self.path = Path(path)

class CommandString(AbstractCommandComponent):
    def __init__(self, string: str):
        self.string = string

class CommandParameters(AbstractCommandComponent):
    def output(self, context: CommandContext): 
        pass

class AbstractCommand(): pass

class Command(AbstractCommand):
    def __init__(self):
        self.components = []

    def add(self, component: AbstractCommandComponent):
        self.components.append(component)

class CommandLine():
    """A command line is composed of one or more commands"""
    def __init__(self):
        self.commands = []

    def add(self, command: Command):
        self.commands.append(command)




class CommandLineJob(Job): 

    def run(self):
        scriptbuilder = self.launcher.scriptbuilder()
        processbuilder = self.launcher.processbuilder()
        donepath = self.donepath

        def check():
            if self.donepath.is_file():
                self.state = Job.STATE_DONE
                return True
                
        os.spawnl(os.P_DETACH, 'some_long_running_command')
        raise NotImplementedError()


class CommandLineTask():
    def __init__(self, commandline: CommandLine):
        self.commandline = commandline
        

    def __call__(self, pyobject, *, launcher=None, workspace=None) -> Job:
        return CommandLineJob(pyobject, launcher=launcher, workspace=workspace)

