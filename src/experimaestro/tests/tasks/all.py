from pathlib import Path
import time
from typing import List
from experimaestro import (
    Meta,
    Param,
    field,
    Task,
    PathGenerator,
    Config,
    STDOUT,
    cache,
)


class SimpleTask(Task):
    x: Param[int]

    def execute(self):
        print(self.x)  # noqa: T201


class Say(Task):
    out: Meta[Path] = field(default_factory=PathGenerator(STDOUT))
    word: Param[str]

    def execute(self):
        self.out.write_text(self.word.upper())


class Concat(Task):
    strings: Param[List[Say]]

    def execute(self):
        # We access the file where standard output was stored
        says = []
        for string in self.strings:
            with open(string.out) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))  # noqa: T201


class ForeignClassB1(Config):
    x: Param[int]


class ForeignTaskA(Task):
    b: Param[ForeignClassB1]

    def execute(self):
        print(self.b.x)  # noqa: T201


class Fail(Task):
    wait: Meta[Path] = field(default_factory=PathGenerator("wait"))

    def execute(self):
        while not self.wait.is_file():
            time.sleep(0.01)
        raise AssertionError("Failing")

    def touch(self):
        while self.__xpm__.job.state.notstarted():
            time.sleep(0.01)

        with open(self.wait, "w") as out:
            out.write("hello")


class FailConsumer(Task):
    fail: Param[Fail]

    def execute(self):
        return True


class Method(Task):
    a: Param[int]

    def execute(self):
        assert self.a == 1


class SetUnknown(Task):
    def execute(self):
        self.abc = 1


"""Check that config works properly"""


class CacheConfig(Config):
    @cache("cached")
    def get(self, path):
        if not path.is_file():
            path.write_text("hello")
        return path.read_text()


class CacheConfigTask(Task):
    data: Param[CacheConfig]

    def execute(self):
        assert self.data.get() == "hello"
