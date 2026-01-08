from enum import Enum
from omegaconf import MISSING
from typing import Optional, List
import attr

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform


class DirtyGitAction(str, Enum):
    """Action to take when the git repository has uncommitted changes"""

    IGNORE = "ignore"
    """Don't check or warn about dirty git state"""

    WARN = "warn"
    """Warn about dirty git state (default)"""

    ERROR = "error"
    """Raise an error if git is dirty"""


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

    module: Optional[str] = None
    """Relative path of the file containing a run function"""

    pythonpath: Optional[List[str]] = None
    """Python path relative to the parent directory of the YAML file"""

    parent: Optional[str] = None
    """Relative path of a YAML file that should be merged"""

    pre_experiment: Optional[str] = None
    """Relative path to a Python file to execute before importing the experiment.

    This is useful for setting environment variables or mocking modules to speed up
    the experiment setup phase (e.g., mocking torch.compile or torch.nn).
    The actual job execution will use real modules."""

    title: str = ""
    """Short description of the experiment"""

    subtitle: str = ""
    """Allows to give some more details about the experiment"""

    paper: str = ""
    """Source paper for this experiment"""

    description: str = ""
    """Description of the experiment"""

    add_timestamp: bool = False
    """Adds a timestamp YYYY_MM_DD-HH_MM to the experiment ID"""

    dirty_git: DirtyGitAction = DirtyGitAction.WARN
    """Action when git repository has uncommitted changes: ignore, warn (default), error"""
