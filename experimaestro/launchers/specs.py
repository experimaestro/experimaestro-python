# Allows to specify how to retrieve

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Union
from pathlib import Path
from experimaestro import typingutils
from dataclass_wizard import YAMLWizard
import humanfriendly
import yaml
from yaml import Loader, YAMLObject


class LauncherSpec:
    """Specifications for a launcher"""

    cuda_mem: Union[str, List[str]]
    """CUDA memory"""


class LauncherLoader(Loader):
    pass


class YAMLDataClass(YAMLObject):
    yaml_loader = LauncherLoader

    @classmethod
    def from_yaml(cls, loader, node):
        kwargs = {}
        # {key.value: value.value for key, value in node.value}
        for key, value in node.value:
            assert isinstance(key.value, str)
            fieldtype = cls.__dataclass_fields__[key.value].type
            v = None
            if list_type := typingutils.get_list(fieldtype):
                assert isinstance(value.value, list), f"{value.value} is not a list"
                v = []
                for element in value.value:
                    v.append(loader.construct_object(value.value[0]))
                    assert isinstance(
                        v[-1], list_type
                    ), f"Element of {v[-1]} is not of type {list_type}"
            elif isinstance(value.value, (str, int, float)):
                v = value.value
            else:
                v = loader.construct_object(value.value)

            # assert isinstance(v, fieldtype), f"{v} is not of type {fieldtype}"
            kwargs[key.value] = v

        return cls(**kwargs)


@dataclass
class GPU(YAMLDataClass):
    yaml_tag = "!gpu"
    type: str
    count: str
    memory: int = field(init=humanfriendly.parse_size)


@dataclass
class Host(YAMLDataClass):
    yaml_tag = "!host"
    name: str
    gpus: List[GPU]
    launchers: List[str]


LauncherLoader.add_path_resolver("!gpu", ["hosts", None, "gpus", None], dict)
LauncherLoader.add_path_resolver("!host", ["hosts", None], dict)


class Configuration:
    INSTANCE: ClassVar[Optional["Configuration"]] = None

    @staticmethod
    def instance():
        if Configuration.INSTANCE is None:
            Configuration.INSTANCE = Configuration()

        return Configuration.INSTANCE

    def __init__(self):
        path = Path("~/.config/experimaestro/launchers.yaml").expanduser()
        self.configuration = yaml.load(path.read_text(), Loader=LauncherLoader)

    def find(self, spec: LauncherSpec):
        print(self.configuration)


def find_launcher(spec: LauncherSpec):
    """Find a launcher matching a given specification"""
    return Configuration.instance().find(spec)


if __name__ == "__main__":
    find_launcher(None)
