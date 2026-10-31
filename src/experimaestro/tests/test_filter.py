"""Tests for the job filter DSL shared by the CLI, the TUI and the web UI"""

from pathlib import Path

import pyparsing as pp
import pytest

from experimaestro.cli.filter import (
    FilterContext,
    FilterVocabulary,
    complete_filter,
    createFilter,
    filter_jobs,
)
from experimaestro.scheduler.state_provider import MockJob


def _job(identifier: str, task_id: str, state: str) -> MockJob:
    return MockJob(
        identifier=identifier,
        task_id=task_id,
        path=Path("/workspace/jobs") / task_id / identifier,
        state=state,
        starttime=None,
        endtime=None,
        progress=[],
        updated_at="2026-01-01T00:00:00",
    )


JOBS = [
    _job("j1", "pkg.Learn", "done"),
    _job("j2", "pkg.Learn", "error"),
    _job("j3", "pkg.Evaluate", "running"),
]

TAGS = {
    "j1": {"model": "bm25", "lr": "0.1"},
    "j2": {"model": "splade"},
    "j3": {"model": "bm25"},
}


def _ids(query: str, tags=TAGS) -> list[str]:
    return [job.identifier for job in filter_jobs(query, JOBS, tags)]


def test_filter_by_state():
    assert _ids("@state = 'done'") == ["j1"]
    assert _ids("@state in ['done', 'running']") == ["j1", "j3"]
    assert _ids("@state not in ['done']") == ["j2", "j3"]


def test_filter_by_name_regex():
    assert _ids("@name ~ 'pkg.Learn'") == ["j1", "j2"]
    # Regexes are anchored at the start of the task id
    assert _ids("@name ~ 'Evaluate'") == []
    assert _ids("@name ~ '.*Evaluate'") == ["j3"]


def test_filter_by_tag_uses_context():
    assert _ids("model = 'bm25'") == ["j1", "j3"]
    assert _ids("model = 'bm25' and @state = 'running'") == ["j3"]
    assert _ids("lr = '0.1' or model = 'splade'") == ["j1", "j2"]
    # Without tags, tag conditions never match
    assert _ids("model = 'bm25'", tags={}) == []


TYPED_TAGS = {
    "j1": {"phase": 1, "lr": 0.1, "debug": True},
    "j2": {"phase": 2, "lr": 1e-3, "debug": False},
    "j3": {"phase": 0},
}


@pytest.mark.parametrize(
    "query,expected",
    [
        # Regex on a non-string tag used to raise a TypeError (#308)
        ("phase ~ '1'", ["j1"]),
        ("phase ~ '[12]'", ["j1", "j2"]),
        # A falsy value (0) is a value, not a missing tag
        ("phase ~ '0'", ["j3"]),
        ("phase = '1'", ["j1"]),
        ("phase = '0'", ["j3"]),
        ("phase in ['1', '0']", ["j1", "j3"]),
        ("phase not in ['1']", ["j2", "j3"]),
        ("lr = '0.1'", ["j1"]),
        ("lr ~ '0\\.001'", ["j2"]),
        ("debug = 'True'", ["j1"]),
        # Missing tags never match a regex
        ("lr ~ '.*'", ["j1", "j2"]),
    ],
)
def test_filter_non_string_tags(query, expected):
    """Tags are compared as text whatever their type"""
    assert _ids(query, tags=TYPED_TAGS) == expected


def test_filter_context_is_read_at_evaluation_time():
    """The TUI shares a context whose tags are updated after parsing"""
    ctx = FilterContext()
    filter_fn = createFilter("model = 'splade'", ctx)
    assert not any(filter_fn(job) for job in JOBS)
    ctx.tags_map = TAGS
    assert [job.identifier for job in JOBS if filter_fn(job)] == ["j2"]


def test_invalid_query_raises():
    with pytest.raises(pp.ParseException):
        filter_jobs("@state = ", JOBS, TAGS)


# --- Completion


def _vocabulary() -> FilterVocabulary:
    vocabulary = FilterVocabulary()
    vocabulary.add_jobs(JOBS, TAGS)
    return vocabulary


def _labels(query: str, cursor: int | None = None) -> list[str]:
    return [item.label for item in complete_filter(query, cursor, _vocabulary()).items]


def _accept(query: str, label: str, cursor: int | None = None) -> str:
    """The query once a suggestion is accepted"""
    completion = complete_filter(query, cursor, _vocabulary())
    (item,) = [item for item in completion.items if item.label == label]
    return query[: completion.start] + item.text + query[completion.end :]


def test_vocabulary_from_jobs():
    vocabulary = _vocabulary()
    assert vocabulary.names == {"pkg.Learn", "pkg.Evaluate"}
    assert vocabulary.tags == {"model": {"bm25", "splade"}, "lr": {"0.1"}}
    assert "done" in vocabulary.states and "error" in vocabulary.states


def test_complete_fields():
    # Special fields first, then the tag keys
    assert _labels("") == ["@state", "@name", "lr", "model"]
    assert _labels("@st") == ["@state"]
    assert _labels("mo") == ["model"]
    assert _accept("mo", "model") == "model "
    # In the middle of a query, the whole word is replaced
    assert _accept("@st = 'done'", "@state", cursor=2) == "@state  = 'done'"


def test_complete_operators():
    assert _labels("model ") == ["=", "~", "in", "not in"]
    assert _labels("model n") == ["not in"]
    assert _accept("model n", "not in") == "model not in ["
    assert _labels("model not ") == ["in"]
    # Unknown special fields are not completed
    assert _labels("@unknown ") == []


def test_complete_values():
    assert _labels("@state = ") == [
        "'unscheduled'",
        "'waiting'",
        "'ready'",
        "'scheduled'",
        "'running'",
        "'done'",
        "'error'",
    ]
    assert _labels("@state = 'ru") == ["'running'"]
    assert _labels("model = ") == ["'bm25'", "'splade'"]
    assert _accept("model = 'sp", "'splade'") == "model = 'splade' "
    # The quote typed by the user is kept, an unquoted word gets quotes
    assert _labels('model = "b') == ['"bm25"']
    assert _accept("model = bm", "'bm25'") == "model = 'bm25' "
    # Values of an unknown tag
    assert _labels("other = ") == []


def test_complete_values_substring():
    assert _labels("@name = 'Eval") == ["'pkg.Evaluate'"]
    # Not from a single character, which would match too many values
    assert _labels("@state = 'u") == ["'unscheduled'"]


def test_complete_regex_escapes_values():
    assert _labels("@name ~ 'pkg") == ["'pkg\\.Evaluate'", "'pkg\\.Learn'"]


def test_complete_in_list():
    assert _labels("model in ") == ["["]
    assert _labels("model in [") == ["'bm25'", "'splade'"]
    assert _accept("model in [", "'bm25'") == "model in ['bm25'"
    assert _labels("model in ['bm25'") == [",", "]"]
    assert _labels("model in ['bm25', ") == ["'bm25'", "'splade'"]


def test_complete_boolean_operators():
    assert _labels("model = 'bm25' ") == ["and", "or"]
    assert _labels("model = 'bm25' o") == ["or"]
    assert _accept("model = 'bm25' a", "and") == "model = 'bm25' and "
    # A space is added when needed
    assert _accept("model='bm25'", "and") == "model='bm25' and "
    assert _labels("model = 'bm25' and ") == ["@state", "@name", "lr", "model"]


def test_complete_replaces_up_to_the_closing_quote():
    query = "@state = 'do' and model = 'bm25'"
    assert _accept(query, "'done'", cursor=11) == (
        "@state = 'done'  and model = 'bm25'"
    )


def test_complete_invalid_prefix():
    assert _labels("model = = ") == []
    assert _labels("(") == []


def test_completions_build_valid_queries():
    """Accepting suggestions builds expressions the filter accepts"""
    steps = [
        ("model", "model "),
        ("in", "model in ["),
        ("'splade'", "model in ['splade'"),
        (",", "model in ['splade', "),
        ("'bm25'", "model in ['splade', 'bm25'"),
        ("]", "model in ['splade', 'bm25'] "),
        ("and", "model in ['splade', 'bm25'] and "),
        ("@state", "model in ['splade', 'bm25'] and @state "),
        ("=", "model in ['splade', 'bm25'] and @state = "),
        ("'done'", "model in ['splade', 'bm25'] and @state = 'done' "),
    ]
    query = ""
    for label, expected in steps:
        query = _accept(query, label)
        assert query == expected
    assert _ids(query) == ["j1"]
