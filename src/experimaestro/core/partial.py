"""Partial identifier computation.

This module provides the `partial` function and `Partial` class for defining
parameter subsets that compute partial identifiers. This enables sharing
directories (like checkpoints) across tasks that differ only in excluded
parameter groups.

Example:
    iter_group = param_group("iter")

    class Learn(Task):
        checkpoints = partial(exclude_groups=[iter_group])

        max_iter: Param[int] = field(groups=[iter_group])
        learning_rate: Param[float]

        # Path will be in WORKSPACE/partials/TASK_ID/checkpoints/PARTIAL_ID/
        checkpoints_path: Param[Path] = field(
            default_factory=PathGenerator(partial=checkpoints)
        )
"""

from dataclasses import dataclass, field as dataclass_field
from typing import Set, Optional


@dataclass(frozen=True)
class ParameterGroup:
    """A parameter group with a name

    The name is just for reference, what is important is the identity of the
    object. The dataclass is frozen to make it hashable.
    """

    name: str


def param_group(name: str) -> ParameterGroup:
    """Create a parameter group for use with partial identifiers.

    Parameter groups allow computing partial identifiers that exclude
    certain parameters, enabling shared directories across related tasks.

    Example::

        training_group = param_group("training")

        class MyTask(Task):
            model_size: Param[int]
            learning_rate: Param[float] = field(groups=[training_group])

    :param name: Unique name for this parameter group
    :return: A ParameterGroup object
    """
    return ParameterGroup(name)


@dataclass
class Partial:
    """Defines a subset of parameters for partial identifier computation.

    A Partial instance defines which parameter groups to include or exclude
    when computing a partial identifier. This enables sharing directories
    (like checkpoints) across experiments that only differ in excluded groups.

    The inclusion/exclusion logic follows these rules:
    1. If `exclude_all` is True, all parameters are excluded by default
    2. Parameters in `exclude_groups` are excluded
    3. Parameters with no group are excluded if `exclude_no_group` is True
    4. Parameters in `include_groups` are always included (overrides exclusion)

    Attributes:
        exclude_groups: Set of group names to exclude from identifier computation
        include_groups: Set of group names to always include (overrides exclusion)
        exclude_no_group: If True, exclude parameters with no group assigned
        exclude_all: If True, exclude all parameters by default
        name: The name of this parameter set (auto-set from class attribute name)
    """

    #: Set of group names to exclude from identifier computation
    exclude_groups: Set[ParameterGroup] = dataclass_field(default_factory=set)

    #: Set of group names to always include (overrides exclusion)
    include_groups: Set[ParameterGroup] = dataclass_field(default_factory=set)

    #: If True, exclude parameters with no group assigned
    exclude_no_group: bool = False

    #: If True, exclude all parameters by default (use include_groups to select)
    exclude_all: bool = False

    #: Name of this parameter set (auto-set from class attribute)
    name: Optional[ParameterGroup] = None

    def __post_init__(self):
        # Ensure groups are sets
        if not isinstance(self.exclude_groups, set):
            self.exclude_groups = set(self.exclude_groups)
        if not isinstance(self.include_groups, set):
            self.include_groups = set(self.include_groups)

    def is_excluded(self, groups: Set[ParameterGroup]) -> bool:
        """Check if a parameter with the given groups should be excluded.

        Args:
            groups: The set of groups the parameter belongs to (empty if no groups).

        Returns:
            True if the parameter should be excluded from partial identifier.
        """
        # Include always overrides exclude - if any group is in include_groups
        if groups and (groups & self.include_groups):
            return False

        # Check exclusion rules
        if self.exclude_all:
            return True
        if not groups and self.exclude_no_group:
            return True
        if groups and (groups & self.exclude_groups):
            return True

        return False


def partial(
    *,
    exclude_groups: list[ParameterGroup] | None = None,
    include_groups: list[ParameterGroup] | None = None,
    exclude_no_group: bool = False,
    exclude_all: bool = False,
) -> Partial:
    """Create a partial specification for partial identifier computation.

    Partials allow tasks to share directories when they differ only
    in certain parameter groups (e.g., training hyperparameters).

    Example::

        training_group = param_group("training")

        class Train(Task):
            model: Param[Model]
            epochs: Param[int] = field(groups=[training_group])

            checkpoint: Meta[Path] = field(
                default_factory=PathGenerator(
                    "model.pt",
                    partial=partial(exclude_groups=[training_group])
                )
            )

    :param exclude_groups: Parameter groups to exclude from identifier
    :param include_groups: Parameter groups to always include (overrides exclusion)
    :param exclude_no_group: If True, exclude parameters with no group assigned
    :param exclude_all: If True, exclude all parameters by default
    :return: A Partial object
    """
    return Partial(
        exclude_groups=set(exclude_groups or []),
        include_groups=set(include_groups or []),
        exclude_no_group=exclude_no_group,
        exclude_all=exclude_all,
    )
