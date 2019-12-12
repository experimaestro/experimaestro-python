import time
from experimaestro import *

tasks = Typename("tasks")

@Argument("word", type=str, required=True, help="Word to generate")
@Task(tasks.say)
class Say:
    def execute(self):
        print(self.word.upper(),)

@Argument("strings", type=Array(Say), help="Strings to concat")
@Task(tasks.concat)
class Concat:
    def execute(self):
        # We access the file where standard output was stored
        says = []
        for string in self.strings:
            with open(string.stdout()) as fp:
                says.append(fp.read().strip())
        print(" ".join(says))

@PathArgument("wait", "wait")
@Task(tasks.fail)
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

@Argument("fail", Fail)
@Task(tasks.failconsumer)
class FailConsumer:
    def execute(self):
        return True

@Argument("a", int)
@Task(tasks.method)
def Method(a: int):
    assert a == 1


if __name__ == "__main__":
    parse_commandline()
