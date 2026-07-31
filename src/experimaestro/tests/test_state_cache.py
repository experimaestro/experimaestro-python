"""Tests for the state provider query cache

Read queries are answered from the cache by default; the state is read again
only when an event says it changed, when a mutation occurs, or when the caller
asks for a full refresh (``refresh=True``).
"""

from collections import Counter
from datetime import datetime

import pytest

from experimaestro.scheduler.query_cache import MISSING, QueryCache
from experimaestro.scheduler.state_provider import MockExperiment, MockJob
from experimaestro.scheduler.state_provider import OfflineStateProvider
from experimaestro.scheduler.state_status import (
    ExperimentJobStateEvent,
    ExperimentUpdatedEvent,
    JobStateChangedEvent,
    JobTag,
)


class CountingProvider(OfflineStateProvider):
    """State provider that counts how often the state is actually read"""

    def __init__(self, experiment_ids=("exp1", "exp2")):
        super().__init__()
        self.calls: Counter = Counter()
        self.experiment_ids = list(experiment_ids)

    def _make_job(self, experiment_id: str) -> MockJob:
        return MockJob(
            identifier=f"{experiment_id}-job",
            task_id="task",
            path=None,
            state="done",
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )

    def _fetch_experiments(self, since=None, *, refresh=False):
        self.calls["experiments"] += 1
        return [
            MockExperiment(workdir=None, run_id="run1", experiment_id_override=exp_id)
            for exp_id in self.experiment_ids
        ]

    def _fetch_experiment(self, experiment_id, *, refresh=False):
        self.calls[f"experiment:{experiment_id}"] += 1
        return MockExperiment(
            workdir=None, run_id="run1", experiment_id_override=experiment_id
        )

    def _fetch_experiment_runs(self, experiment_id, *, refresh=False):
        self.calls[f"runs:{experiment_id}"] += 1
        return []

    def _fetch_current_run(self, experiment_id, *, refresh=False):
        self.calls[f"current_run:{experiment_id}"] += 1
        return "run1"

    def _fetch_jobs(
        self,
        experiment_id=None,
        run_id=None,
        task_id=None,
        state=None,
        tags=None,
        since=None,
        *,
        refresh=False,
    ):
        self.calls[f"jobs:{experiment_id}"] += 1
        return [self._make_job(experiment_id or "all")]

    def _fetch_job(self, task_id, job_id, *, refresh=False):
        self.calls[f"job:{job_id}"] += 1
        return self._make_job(job_id)

    def _fetch_all_jobs(self, state=None, tags=None, since=None, *, refresh=False):
        self.calls["all_jobs"] += 1
        return [self._make_job(exp_id) for exp_id in self.experiment_ids]

    def _fetch_tags_map(self, experiment_id, run_id=None, *, refresh=False):
        self.calls[f"tags:{experiment_id}"] += 1
        return {f"{experiment_id}-job": {"tag": "value"}}

    def _fetch_dependencies_map(self, experiment_id, run_id=None, *, refresh=False):
        self.calls[f"deps:{experiment_id}"] += 1
        return {}

    def _fetch_experiment_job_info(self, experiment_id, run_id=None, *, refresh=False):
        self.calls[f"job_info:{experiment_id}"] += 1
        return {}

    def _fetch_orphan_jobs(self, *, refresh=False, **kwargs):
        self.calls["orphans"] += 1
        return []

    def _fetch_services_from_storage(self, experiment_id, run_id):
        self.calls[f"services:{experiment_id}"] += 1
        return []

    def _create_job(self, full_id, *args, **kwargs):
        return self._make_job(full_id)

    def _create_experiment(self, *args, **kwargs):
        return MockExperiment(workdir=None, run_id="run1")

    def kill_job(self, job, perform=False):
        return False

    def clean_job(self, job, perform=False):
        return False

    def close(self):
        pass


@pytest.fixture
def provider():
    return CountingProvider()


def test_queries_are_cached(provider):
    """The same query is answered from the cache"""
    for _ in range(3):
        provider.get_experiments()
        provider.get_jobs("exp1")
        provider.get_tags_map("exp1")
        provider.get_experiment_job_info("exp1")
        provider.get_orphan_jobs()

    assert provider.calls["experiments"] == 1
    assert provider.calls["jobs:exp1"] == 1
    assert provider.calls["tags:exp1"] == 1
    assert provider.calls["job_info:exp1"] == 1
    assert provider.calls["orphans"] == 1


def test_query_arguments_are_part_of_the_key(provider):
    """Different arguments are different queries"""
    provider.get_jobs("exp1")
    provider.get_jobs("exp2")
    provider.get_jobs("exp1", state="done")

    assert provider.calls["jobs:exp1"] == 2
    assert provider.calls["jobs:exp2"] == 1


def test_refresh_forces_a_reread(provider):
    """refresh=True bypasses the cache"""
    provider.get_jobs("exp1")
    provider.get_jobs("exp1")
    assert provider.calls["jobs:exp1"] == 1

    provider.get_jobs("exp1", refresh=True)
    assert provider.calls["jobs:exp1"] == 2

    # ... and the fresh value is cached again
    provider.get_jobs("exp1")
    assert provider.calls["jobs:exp1"] == 2


def test_job_state_event_does_not_invalidate(provider):
    """Job objects are updated in place, so the queries stay valid"""
    provider.get_jobs("exp1")
    provider.get_experiments()

    provider.apply_event(
        JobStateChangedEvent(job_id="exp1-job", task_id="task", state="running")
    )

    provider.get_jobs("exp1")
    provider.get_experiments()
    assert provider.calls["jobs:exp1"] == 1
    assert provider.calls["experiments"] == 1


def submitted(experiment_id="exp1", job_id="new-job", task_id="task", **kwargs):
    return ExperimentJobStateEvent(
        experiment_id=experiment_id,
        job_id=job_id,
        task_id=task_id,
        scheduler_state="submitted",
        **kwargs,
    )


def test_new_job_extends_the_cached_answers(provider):
    """A submitted job is added to the cached lists, without reading again"""
    provider.get_jobs("exp1")
    provider.get_jobs("exp2")
    provider.get_all_jobs()
    provider.get_experiments()

    provider.apply_event(submitted())

    assert [j.identifier for j in provider.get_jobs("exp1")] == [
        "exp1-job",
        "new-job",
    ]
    assert [j.identifier for j in provider.get_jobs("exp2")] == ["exp2-job"]
    assert "new-job" in [j.identifier for j in provider.get_all_jobs()]

    assert provider.calls["jobs:exp1"] == 1, "the answer was updated, not re-read"
    assert provider.calls["jobs:exp2"] == 1
    assert provider.calls["all_jobs"] == 1
    assert provider.calls["experiments"] == 1, "a job does not change the experiments"


def test_new_job_extends_the_cached_maps(provider):
    """The event payload feeds the tags/dependencies/job info maps"""
    provider.get_tags_map("exp1")
    provider.get_dependencies_map("exp1")
    provider.get_experiment_job_info("exp1")

    provider.apply_event(
        submitted(tags=[JobTag(key="fold", value="1")], depends_on=["exp1-job"])
    )

    assert provider.get_tags_map("exp1")["new-job"] == {"fold": "1"}
    assert provider.get_dependencies_map("exp1")["new-job"] == ["exp1-job"]
    assert provider.get_experiment_job_info("exp1")["new-job"].task_id == "task"

    assert provider.calls["tags:exp1"] == 1, "the answers were updated, not re-read"
    assert provider.calls["deps:exp1"] == 1
    assert provider.calls["job_info:exp1"] == 1


def test_new_job_respects_query_filters(provider):
    """A job is only added to the queries it belongs to"""
    provider.get_jobs("exp1", task_id="other-task")
    provider.get_jobs("exp1", state="done")
    provider.get_jobs("exp1", tags={"fold": "2"})
    provider.get_jobs("exp1", tags={"fold": "1"})

    provider.apply_event(submitted(tags=[JobTag(key="fold", value="1")]))

    assert "new-job" not in [
        j.identifier for j in provider.get_jobs("exp1", task_id="other-task")
    ]
    assert "new-job" not in [
        j.identifier for j in provider.get_jobs("exp1", state="done")
    ]
    assert "new-job" not in [
        j.identifier for j in provider.get_jobs("exp1", tags={"fold": "2"})
    ]
    # ... but it does belong to the query matching its tags
    assert "new-job" in [
        j.identifier for j in provider.get_jobs("exp1", tags={"fold": "1"})
    ]

    assert provider.calls["jobs:exp1"] == 4, "no query had to be read again"


def test_undecidable_query_is_dropped(provider):
    """A query the event cannot be checked against is read again"""
    provider.get_jobs("exp1", since=datetime(2026, 1, 1))
    provider.apply_event(submitted())

    provider.get_jobs("exp1", since=datetime(2026, 1, 1))
    assert provider.calls["jobs:exp1"] == 2


def test_job_of_another_run_is_ignored(provider):
    provider.get_jobs("exp1", run_id="run1")
    provider.apply_event(submitted(run_id="run2"))

    assert "new-job" not in [
        j.identifier for j in provider.get_jobs("exp1", run_id="run1")
    ]
    assert provider.calls["jobs:exp1"] == 1


def test_services_use_the_same_cache(provider):
    """Services are a read query like the others"""
    provider.get_services("exp1")
    provider.get_services("exp1")
    assert provider.calls["services:exp1"] == 1

    provider.get_services("exp1", refresh=True)
    assert provider.calls["services:exp1"] == 2

    # ... and they are dropped with the rest of the experiment
    provider.apply_event(ExperimentUpdatedEvent(experiment_id="exp1"))
    provider.get_services("exp1")
    assert provider.calls["services:exp1"] == 3


def test_experiment_event_invalidates_only_that_experiment(provider):
    provider.get_tags_map("exp1")
    provider.get_tags_map("exp2")

    provider.apply_event(ExperimentUpdatedEvent(experiment_id="exp1"))

    provider.get_tags_map("exp1")
    provider.get_tags_map("exp2")
    assert provider.calls["tags:exp1"] == 2
    assert provider.calls["tags:exp2"] == 1


def test_results_are_copies(provider):
    """Callers sorting or filtering a result must not corrupt the cache"""
    jobs = provider.get_jobs("exp1")
    jobs.clear()
    assert len(provider.get_jobs("exp1")) == 1

    tags = provider.get_tags_map("exp1")
    tags.clear()
    assert provider.get_tags_map("exp1") == {"exp1-job": {"tag": "value"}}

    assert provider.calls["jobs:exp1"] == 1, "no re-read happened"


def test_invalidate_caches(provider):
    provider.get_experiments()
    provider.get_jobs("exp1")

    provider.invalidate_caches("exp1")
    provider.get_experiments()
    provider.get_jobs("exp1")

    # exp1 and the workspace-wide queries are dropped
    assert provider.calls["jobs:exp1"] == 2
    assert provider.calls["experiments"] == 2


def test_none_is_a_cacheable_value():
    """A query answering None is cached, and not re-run as a miss"""

    class NoExperiment(CountingProvider):
        def _fetch_experiment(self, experiment_id, *, refresh=False):
            self.calls[f"experiment:{experiment_id}"] += 1
            return None

    provider = NoExperiment()
    assert provider.get_experiment("exp1") is None
    assert provider.get_experiment("exp1") is None
    assert provider.calls["experiment:exp1"] == 1


class TestQueryCache:
    def test_miss_returns_sentinel(self):
        cache = QueryCache()
        assert cache.get(("a",)) is MISSING

    def test_ttl_expiry(self, monkeypatch):
        cache = QueryCache(ttl=10.0)
        now = [1000.0]
        monkeypatch.setattr(
            "experimaestro.scheduler.query_cache.time.monotonic", lambda: now[0]
        )

        cache.put(("a",), "value", experiment_id=None)
        assert cache.get(("a",)) == "value"

        now[0] += 11.0
        assert cache.get(("a",)) is MISSING

    def test_invalidate_global_keeps_experiments(self):
        cache = QueryCache()
        cache.put(("experiments",), [], experiment_id=None)
        cache.put(("jobs", "exp1"), [], experiment_id="exp1")

        cache.invalidate_global()
        assert cache.get(("experiments",)) is MISSING
        assert cache.get(("jobs", "exp1")) == []

    def test_invalidate_experiment_drops_global_too(self):
        cache = QueryCache()
        cache.put(("experiments",), [], experiment_id=None)
        cache.put(("jobs", "exp1"), [], experiment_id="exp1")
        cache.put(("jobs", "exp2"), [], experiment_id="exp2")

        cache.invalidate("exp1")
        assert cache.get(("experiments",)) is MISSING
        assert cache.get(("jobs", "exp1")) is MISSING
        assert cache.get(("jobs", "exp2")) == []

    def test_hit_and_miss_counters(self):
        cache = QueryCache()
        cache.get(("a",))
        cache.put(("a",), 1, experiment_id=None)
        cache.get(("a",))
        assert (cache.hits, cache.misses) == (1, 1)


def test_datetime_arguments_are_hashable(provider):
    """Queries taking a datetime can be cached"""
    since = datetime(2026, 1, 1)
    provider.get_experiments(since=since)
    provider.get_experiments(since=since)
    assert provider.calls["experiments"] == 1
