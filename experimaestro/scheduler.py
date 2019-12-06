from pathlib import Path
from .workspace import Workspace
from .api import PyObject

class JobContext():
    """Context of a job"""
    def __init__(self, workspace: Workspace, object: PyObject):
        assert workspace and object
        self.workspace = workspace
        self.object = object   
        self.type = object.__class__.__xpm__

    @property
    def jobpath(self):
        return self.workspace.jobspath  / str(self.type.typename) / self.object.__xpm__.identifier.hex()

class Job():   
    def state(self):
        raise NotImplementedError()

    def wait(self):
        raise NotImplementedError()

    def codePath(self):
        return self.code

    def stdoutPath(self):
        return self.stdout

    def stderrPath(self):
        return self.stderr


class Register():
    def __init__(self):
        self.tasks = {}

register = Register()

