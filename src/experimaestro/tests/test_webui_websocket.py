"""Tests for the WebUI job filter request (beyond the shared UI tests)

Kill/delete and the in-experiment filters are covered, for both interfaces,
by ``tests/ui/``.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from experimaestro.scheduler.interfaces import ExperimentStatus
from experimaestro.scheduler.state_provider import MockJob
from experimaestro.webui.websocket import WebSocketHandler


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


@dataclass
class FakeExperiment:
    experiment_id: str
    status: ExperimentStatus


class FakeProvider:
    """Minimal state provider: two experiments sharing one job"""

    def __init__(self):
        shared = _job("j3", "pkg.Evaluate", "running")
        self.experiments = {
            "xp-a": FakeExperiment("xp-a", ExperimentStatus.DONE),
            "xp-b": FakeExperiment("xp-b", ExperimentStatus.RUNNING),
        }
        self.jobs = {
            "xp-a": [
                _job("j1", "pkg.Learn", "done"),
                _job("j2", "pkg.Learn", "running"),
                shared,
            ],
            "xp-b": [shared, _job("j4", "pkg.Learn", "error")],
        }
        self.tags = {
            "xp-a": {"j1": {"model": "bm25"}, "j2": {"model": "splade"}},
            "xp-b": {"j3": {"model": "bm25"}, "j4": {"model": "bm25"}},
        }

    def get_experiments(self, refresh=False):
        return list(self.experiments.values())

    def get_experiment(self, experiment_id, refresh=False):
        return self.experiments.get(experiment_id)

    def get_jobs(self, experiment_id=None, run_id=None, refresh=False):
        return self.jobs.get(experiment_id, [])

    def get_tags_map(self, experiment_id, run_id=None, refresh=False):
        return self.tags.get(experiment_id, {})


class FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, message):
        self.sent.append(message)

    def of_type(self, msg_type: str) -> list[dict]:
        return [m["payload"] for m in self.sent if m["type"] == msg_type]


def _send(provider, message) -> FakeWebSocket:
    handler = WebSocketHandler(provider, token="t")
    websocket = FakeWebSocket()
    handler.connections.add(websocket)
    asyncio.run(handler.handle_message(websocket, message))
    return websocket


def _filter(provider, query, **payload):
    ws = _send(
        provider,
        {"type": "jobs.filter", "payload": {"query": query, "requestId": 7, **payload}},
    )
    (response,) = ws.of_type("jobs.filter")
    assert response["requestId"] == 7
    return response


def test_jobs_filter_within_experiment():
    provider = FakeProvider()
    response = _filter(provider, "model = 'bm25'", experimentId="xp-a")
    assert response["error"] is None
    assert response["jobIds"] == ["j1"]

    response = _filter(provider, "@state = 'running'", experimentId="xp-a")
    assert response["jobIds"] == ["j2", "j3"]


def test_jobs_filter_uses_per_experiment_tags():
    """Without an experiment, a job matches if it matches in any experiment"""
    provider = FakeProvider()
    response = _filter(provider, "model = 'bm25' and @name ~ 'pkg\\.'")
    assert sorted(response["jobIds"]) == ["j1", "j3", "j4"]


def test_jobs_filter_parse_error():
    response = _filter(FakeProvider(), "@state = ", experimentId="xp-a")
    assert response["jobIds"] is None
    assert response["error"].startswith("Invalid query")


def test_jobs_filter_empty_query():
    response = _filter(FakeProvider(), "  ", experimentId="xp-a")
    assert response["jobIds"] is None
    assert response["error"] is None
