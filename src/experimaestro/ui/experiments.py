"""Grouping and search of the experiment list, shared by the UIs

The web UI does the same on the browser side (``app/src/Experiments.tsx``):
keep both in sync.
"""

import re
from collections.abc import Iterable, Sequence
from typing import Literal, TypeVar

from experimaestro.scheduler.interfaces import BaseExperiment

ExperimentGrouping = Literal["none", "prefix", "tag"]

#: Grouping modes, in the order a key cycles through them
EXPERIMENT_GROUPINGS: tuple[ExperimentGrouping, ...] = ("none", "prefix", "tag")

#: Group of the experiments without tags
UNTAGGED = "(untagged)"

E = TypeVar("E", bound=BaseExperiment)


#: Separators ending the prefix of an experiment ID
PREFIX_SEPARATORS = re.compile(r"[-_.]")


def experiment_prefix(experiment_id: str) -> str:
    """The prefix of an experiment ID: what comes before its first ``-``, ``_``
    or ``.`` (the whole ID when there is none)"""
    return PREFIX_SEPARATORS.split(experiment_id, maxsplit=1)[0]


def experiment_matches(experiment: BaseExperiment, text: str) -> bool:
    """Whether the experiment ID or one of its tags contains the text (ignoring
    case)"""
    text = text.lower()
    return text in experiment.experiment_id.lower() or any(
        text in tag.lower() for tag in experiment.experiment_tags
    )


def group_experiments(
    experiments: Iterable[E], grouping: ExperimentGrouping
) -> list[tuple[str, list[E]]]:
    """Groups of experiments, sorted by name (keeping the order of the
    experiments within a group)

    With the ``tag`` grouping, an experiment is in the group of each of its
    tags, and those without tags are in the last group (:data:`UNTAGGED`). With
    ``none``, there is a single group named ``""``.
    """
    experiments = list(experiments)
    if grouping == "none":
        return [("", experiments)]

    groups: dict[str, list[E]] = {}
    for experiment in experiments:
        keys: Sequence[str]
        if grouping == "prefix":
            keys = [experiment_prefix(experiment.experiment_id)]
        else:
            keys = sorted(set(experiment.experiment_tags)) or [UNTAGGED]
        for key in keys:
            groups.setdefault(key, []).append(experiment)

    return sorted(groups.items(), key=lambda item: (item[0] == UNTAGGED, item[0]))
