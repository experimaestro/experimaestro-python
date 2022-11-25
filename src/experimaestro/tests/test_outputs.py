"""Test for task outputs"""

from experimaestro import Config, Task, Param
from experimaestro.core.objects import SerializedConfig, Serialized, TaskOutput
from experimaestro.tests.utils import TemporaryExperiment


class B(Config):
    x: Param[int] = 1


class A(Config):
    b: Param[B]


class LoaderA(Serialized):
    @staticmethod
    def fromJSON(x) -> A:
        return A(b=B(x=x)).instance()


class Main(Task):
    a: Param[A]

    def taskoutputs(self):
        return self.a, {
            "a": self.a,
            "serialized": SerializedConfig(self.a, LoaderA(self.a.b.x)),
        }

    def execute(self):
        print(self.a.b.x)


class MainB(Task):
    b: Param[B]

    def execute(self):
        pass


def test_output_taskoutput():
    a = A(b=B())
    output, ioutput = Main(a=a).submit(dryrun=True)

    assert isinstance(ioutput["serialized"], TaskOutput)
    assert isinstance(output, TaskOutput), "outputs should be task proxies"

    # Direct
    Main(a=output)

    # Via getattr
    Main(a=A(b=output.b))

    # Via getitem
    Main(a=ioutput["a"])

    # Now, submits
    Main(a=output).submit(dryrun=True)


def test_output_serialiation():
    """Test output serialization"""

    with TemporaryExperiment("output_serialization", maxwait=5) as xp:
        a = A(b=B(x=2))

        main0 = Main(a=a)
        output, ioutput = main0.submit()

        # Direct
        serialized_a = ioutput["serialized"]
        main1 = Main(a=serialized_a)
        main1.submit()

        # Indirect (via attribute)
        serialized_a = ioutput["serialized"]
        main2 = Main(a=A(b=serialized_a.b))
        main2.submit()

        xp.wait()

        for main in (main1, main2):
            assert main.__xpm__.job.stdout.read_text().strip() == "2"
            assert len(main.__xpm__.job.dependencies) == 1
            dep = next(iter(main.__xpm__.job.dependencies))
            assert dep.origin.config is main0
