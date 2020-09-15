import time
from experimaestro import *

tasks = Identifier("tasks")


@argument("x", type=int)
@task()
class SimpleTask:
    def execute(self):
        print(self.x)


@argument("word", type=str, required=True, help="Word to generate")
@pathoption("out", STDOUT)
@task(tasks.say)
class Say:
    def execute(self):
        print(
            self.word.upper(),
        )


@argument("strings", type=Array(Say), help="Strings to concat")
@task(tasks.concat)
class Concat:
    def execute(self):
        # We access the file where standard output was stored
        says = []
        for string in self.strings:
            with open(string.out) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))


@argument("x", type=int)
@config()
class ForeignClassB1:
    pass


@argument("b", type=ForeignClassB1)
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

    @configmethod
    def touch(self):
        while self.__xpm__.job.state.notstarted():
            time.sleep(0.05)
        with open(self.wait, "w") as out:
            out.write("hello")


@argument("fail", Fail)
@task(tasks.failconsumer)
class FailConsumer:
    def execute(self):
        return True


@argument("a", int)
@task(tasks.method)
def Method(a: int):
    assert a == 1


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
