from omegaconf import MISSING
from typing import Optional
import attr

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform


@dataclass_transform(kw_only_default=True)
def configuration(*args, **kwargs):
    """Method to define keyword only dataclasses

    Configurations are keyword-only
    """

    return attr.define(*args, kw_only=True, slots=False, hash=True, eq=True, **kwargs)


@configuration()
class ConfigurationBase:
    """Base configuration for any experiment"""

    id: str = MISSING
    """ID of the experiment

    This ID is used by experimaestro when running as the experiment.
    """

    file: str = "experiment"
    """Relative path of the file containing a run function"""

    parent: Optional[str] = None
    """Relative path of a YAML file that should be merged"""

    title: str = ""
    """Short description of the experiment"""

    subtitle: str = ""
    """Allows to give some more details about the experiment"""

    paper: str = ""
    """Source paper for this experiment"""

    description: str = ""
    """Description of the experiment"""
