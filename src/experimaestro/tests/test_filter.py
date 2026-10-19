"""Tests for the job filter DSL shared by the CLI, the TUI and the web UI"""

from pathlib import Path

import pyparsing as pp
import pytest

from experimaestro.cli.filter import FilterContext, createFilter, filter_jobs
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
