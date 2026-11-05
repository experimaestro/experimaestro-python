"""Deprecated module: use :mod:`experimaestro.experiments.specification`"""

import warnings

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform

from .specification import (  # noqa: F401
    MISSING,
    DirtyGitAction,
    ExperimentSpecification,
    specification,
)


@dataclass_transform(kw_only_default=True)
def configuration(*args, **kwargs):
    """Deprecated: use :func:`~experimaestro.experiments.specification`"""
    warnings.warn(
        "experimaestro.experiments.configuration is deprecated, "
        "use experimaestro.experiments.specification",
        DeprecationWarning,
        stacklevel=2,
    )
    if len(args) == 1 and not kwargs and isinstance(args[0], type):
        # Used as @configuration (without parentheses)
        return specification(args[0])
    return specification(*args, **kwargs)


class ConfigurationBase(ExperimentSpecification):
    """Deprecated: use :class:`~experimaestro.experiments.ExperimentSpecification`"""

    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            "experimaestro.experiments.ConfigurationBase is deprecated, "
            "use experimaestro.experiments.ExperimentSpecification",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
