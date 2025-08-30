"""Test for task outputs"""

from experimaestro import Config, Task, Param
from experimaestro.scheduler.workspace import RunMode


class B(Config):
    x: Param[int] = 1


class A(Config):
    b: Param[B]


class Main(Task):
    a: Param[A]

    def task_outputs(self, dep):
        return self.a, {
            "a": self.a,
        }

    def execute(self):
        print(self.a.b.x)  # noqa: T201


class MainB(Task):
    b: Param[B]

    def execute(self):
        pass


def test_output_taskoutput():
    a = A.C(b=B.C())
    output, ioutput = Main.C(a=a).submit(run_mode=RunMode.DRY_RUN)

    # Direct
    Main.C(a=output)

    # Via getattr
    Main.C(a=A.C(b=output.b))

    # Via getitem
    Main.C(a=ioutput["a"])

    # Now, submits
    Main.C(a=output).submit(run_mode=RunMode.DRY_RUN)
