import time
from typing import List
from experimaestro import (
    param,
    Param,
    Task,
    Config,
    pathoption,
    STDOUT,
    cache,
)


class SimpleTask(Task):
    x: Param[int]

    def execute(self):
        print(self.x)  # noqa: T201


@pathoption("out", STDOUT)
class Say(Task):
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


@param("x", type=int)
class ForeignClassB1(Config):
    pass


@param("b", type=ForeignClassB1)
class ForeignTaskA(Task):
    def execute(self):
        print(self.b.x)  # noqa: T201


@pathoption("wait", "wait")
class Fail(Task):
    def execute(self):
        while not self.wait.is_file():
            time.sleep(0.01)
        raise AssertionError("Failing")

    def touch(self):
        while self.__xpm__.job.state.notstarted():
            time.sleep(0.01)

        with open(self.wait, "w") as out:
            out.write("hello")


@param("fail", Fail)
class FailConsumer(Task):
    def execute(self):
        return True


@param("a", int)
class Method(Task):
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


@param("data", type=CacheConfig)
class CacheConfigTask(Task):
    def execute(self):
        assert self.data.get() == "hello"
