"""Grouping and search of the experiment list (shared by the UIs)"""

from types import SimpleNamespace

from experimaestro.ui.experiments import (
    UNTAGGED,
    experiment_matches,
    experiment_prefix,
    group_experiments,
)


def _xp(experiment_id: str, *tags: str):
    return SimpleNamespace(experiment_id=experiment_id, experiment_tags=list(tags))


def _names(groups):
    return [(name, [xp.experiment_id for xp in xps]) for name, xps in groups]


def test_prefix():
    assert experiment_prefix("msmarco-splade-v2") == "msmarco"
    assert experiment_prefix("baseline_ettin150m_long") == "baseline"
    assert experiment_prefix("v1.2-test") == "v1"
    assert experiment_prefix("baseline") == "baseline"


def test_group_by_prefix_keeps_order():
    xps = [_xp("trec-b"), _xp("msmarco-a"), _xp("trec-a"), _xp("solo")]
    assert _names(group_experiments(xps, "prefix")) == [
        ("msmarco", ["msmarco-a"]),
        ("solo", ["solo"]),
        ("trec", ["trec-b", "trec-a"]),
    ]


def test_group_by_tag():
    xps = [_xp("a", "eval", "mice"), _xp("b"), _xp("c", "eval")]
    assert _names(group_experiments(xps, "tag")) == [
        ("eval", ["a", "c"]),
        ("mice", ["a"]),
        (UNTAGGED, ["b"]),
    ]


def test_no_grouping():
    xps = [_xp("b"), _xp("a")]
    assert _names(group_experiments(xps, "none")) == [("", ["b", "a"])]


def test_matches_id_or_tag():
    xp = _xp("msmarco-splade", "Paper")
    assert experiment_matches(xp, "SPLADE")
    assert experiment_matches(xp, "pap")
    assert not experiment_matches(xp, "mice")
