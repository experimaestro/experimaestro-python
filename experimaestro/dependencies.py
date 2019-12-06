"""Dependency between tasks and tokens"""

from pathlib import Path

class Dependency(): pass

class Token(): pass

class CounterToken(Token):
    def __init__(self, path: Path, tokens: int=-1):
        self.path = path
        self.tokens = tokens

    def createDependency(self, count: int):
        return Dependency()