import time
from typing import List
from experimaestro import (
    param,
    Param,
    config,
    task,
    pathoption,
    Identifier,
    STDOUT,
    cache,
)

tasks = Identifier("tasks")


@task()
class SimpleTask:
    x: Param[int]

    def execute(self):
        print(self.x)


@pathoption("out", STDOUT)
@task(tasks.say)
class Say:
    word: Param[str]

    def execute(self):
        print(
            self.word.upper(),
        )


@param("strings", type=List[Say], help="Strings to concat")
@task(tasks.concat)
class Concat:
    def execute(self):
        # We access the file where standard output was stored
        says = []
        for string in self.strings:
            with open(string.out) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))


@param("x", type=int)
@config()
class ForeignClassB1:
    pass


@param("b", type=ForeignClassB1)
@task()
class ForeignTaskA:
    def execute(self):
        print(self.b.x)


@pathoption("wait", "wait")
@task(tasks.fail)
class Fail:
    def execute(self):
        while not self.wait.is_file():
            time.sleep(0.1)
        raise AssertionError("Failing")

    def touch(self):
        while self.__xpm__.job.state.notstarted():
            time.sleep(0.05)
        with open(self.wait, "w") as out:
            out.write("hello")


@param("fail", Fail)
@task(tasks.failconsumer)
class FailConsumer:
    def execute(self):
        return True


@param("a", int)
@task(tasks.method)
class Method:
    def execute(self):
        assert self.a == 1


@task(tasks.setunknown)
class SetUnknown:
    def execute(self):
        self.abc = 1


"""Check that config works properly"""


@config()
class CacheConfig:
    @cache("cached")
    def get(self, path):
        if not path.is_file():
            path.write_text("hello")
        return path.read_text()


@param("data", type=CacheConfig)
@task()
class CacheConfigTask:
    def execute(self):
        assert self.data.get() == "hello"
