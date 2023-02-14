# Test post-experimental serialization

from pathlib import Path
from experimaestro import Config, DataPath, Task, Param
from experimaestro.core.objects import ConfigInformation
from experimaestro.scheduler.workspace import RunMode


class A(Config):
    path: DataPath[Path]


class TaskA(Task):
    id: Param[str]

    def taskoutputs(self):
        return A()


def test_serialization_simple(tmp_path_factory):
    dir = tmp_path_factory.mktemp("ser")

    a = A(path=Path(__file__))
    a.__xpm__.serialize(dir)

    des_a = ConfigInformation.deserialize(dir)
    assert des_a.path != Path(__file__)
    assert des_a.path.read_text() == Path(__file__).read_text()


def test_serialization_identifier(tmp_path_factory):
    dir = tmp_path_factory.mktemp("ser")

    a = TaskA(id="id").submit(run_mode=RunMode.DRY_RUN)
    a = a.__unwrap__()
    a.__xpm__.serialize(dir)

    des_a = ConfigInformation.deserialize(dir)

    des_a_id = des_a.__identifier__()

    assert des_a_id.all == a.__identifier__().all, (
        "Identifier don't match: "
        f"expected {a.__identifier__().all.hex()}, got {des_a_id.all.hex()}"
    )
