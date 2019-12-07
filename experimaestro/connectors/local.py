"""All classes related to localhost management
"""
import os

from .connectors import Connector
from .process import ProcessBuilder, RedirectType, Redirect

class LocalProcessBuilder(ProcessBuilder):
    pass

class LocalConnector(Connector):
    def start(self):
        os.
