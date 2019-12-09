"""All classes related to localhost management
"""
import subprocess

from . import Connector, ProcessBuilder, RedirectType, Redirect

class LocalProcessBuilder(ProcessBuilder):
    def start(self):

        creationflags = 0
        if self.detach:
            creationflags = subprocess.DETACHED_PROCESS

        subprocess.Popen(self.command, creationflags=creationflags)

class LocalConnector(Connector):
    pass
