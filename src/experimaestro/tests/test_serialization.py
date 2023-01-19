# Test post-experimental serialization

from pathlib import Path
from experimaestro import Config, DataPath
from experimaestro.core.objects import ConfigInformation


class A(Config):
    path: DataPath[Path]


def test_serialization_simple(tmp_path_factory):
    dir = tmp_path_factory.mktemp("ser")

    a = A(path=Path(__file__))
    a.__xpm__.serialize(dir)

    des_a = ConfigInformation.deserialize(dir)
    assert des_a.path != Path(__file__)
    assert des_a.path.read_text() == Path(__file__).read_text()
