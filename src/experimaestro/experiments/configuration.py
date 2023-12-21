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
    id: str = MISSING
    """ID of the experiment"""

    description: str = ""
    """Description of the experiment"""

    file: str = "experiment"
    """qualified name (relative to the module) for the file containing a run function"""

    parent: Optional[str] = None
    """Relative path of a YAML file that should be merged"""
