"""Clients only receive the events of the experiments they watch

The SSH server filters the events before sending them, and so does the web
server for each connection (see ``scheduler/event_filter.py``).
"""

import io
import json

import pytest

from experimaestro.scheduler.event_filter import EventFilter
from experimaestro.scheduler.remote.server import SSHStateProviderServer
from experimaestro.scheduler.state_status import (
    ActionAddedEvent,
    ExperimentJobStateEvent,
    ExperimentUpdatedEvent,
    JobStateChangedEvent,
    JobSubmittedEvent,
    ServiceAddedEvent,
    ServiceStateChangedEvent,
)
from experimaestro.webui.state_bridge import StateBridge
from experimaestro.webui.websocket import WebSocketHandler

from .scenario import EVAL_WAITING, LEARN_DONE, LEARN_RUNNING, JobSpec

pytestmark = pytest.mark.anyio


def job_state(job_id: str) -> JobStateChangedEvent:
    return JobStateChangedEvent(job_id=job_id, state="running")


def service_state(experiment_id: str, state: str) -> ServiceStateChangedEvent:
    return ServiceStateChangedEvent(
        experiment_id=experiment_id, service_id="tensorboard", state=state
    )


# --- Event filter


def test_filter_watches_everything_by_default():
    event_filter = EventFilter()
    assert event_filter.accepts(job_state("j1"))
    assert event_filter.accepts(ActionAddedEvent(experiment_id="b"))


def test_filter_jobs():
    """Job events are sent for the jobs of the watched experiments, including
    the ones submitted afterwards"""
    event_filter = EventFilter()
    event_filter.watch({"a"}, ["j1"])
    assert event_filter.accepts(job_state("j1"))
    assert not event_filter.accepts(job_state("j2"))

    assert not event_filter.accepts(
        ExperimentJobStateEvent(experiment_id="b", job_id="j2")
    )
    assert not event_filter.accepts(job_state("j2"))
    assert event_filter.accepts(JobSubmittedEvent(experiment_id="a", job_id="j3"))
    assert event_filter.accepts(job_state("j3"))


def test_filter_experiment_events():
    """Experiment-level events are always sent, actions of watched experiments"""
    event_filter = EventFilter()
    event_filter.watch(set())
    assert event_filter.accepts(ExperimentUpdatedEvent(experiment_id="b"))
    assert event_filter.accepts(ServiceAddedEvent(experiment_id="b"))
    assert not event_filter.accepts(ActionAddedEvent(experiment_id="b"))


def test_filter_started_services():
    """The state of the other experiments' services is sent while they are
    started, up to the event telling they stopped"""
    event_filter = EventFilter()
    event_filter.watch({"a"})
    assert event_filter.accepts(service_state("a", "STOPPED"))
    assert not event_filter.accepts(service_state("b", "STOPPED"))
    assert event_filter.accepts(service_state("b", "STARTING"))
    assert event_filter.accepts(service_state("b", "RUNNING"))
    assert event_filter.accepts(service_state("b", "STOPPED"))
    assert not event_filter.accepts(service_state("b", "STOPPED"))


# --- SSH server


def notified(stdout: io.BytesIO) -> list[str]:
    """Types of the events notified by the server"""
    return [
        json.loads(line)["params"]["event_type"]
        for line in stdout.getvalue().decode().splitlines()
    ]


def test_server_sends_watched_events(provider, tmp_path):
    stdout = io.BytesIO()
    server = SSHStateProviderServer(tmp_path, stdin=io.BytesIO(), stdout=stdout)
    server._state_provider = provider

    # Older clients do not watch: they receive everything
    server._on_state_event(job_state(LEARN_RUNNING))
    assert notified(stdout) == ["JobStateChangedEvent"]

    stdout.seek(0)
    stdout.truncate()
    server._handle_watch({"experiments": ["baseline"]})
    server._on_state_event(job_state(LEARN_RUNNING))  # sweep
    server._on_state_event(ExperimentUpdatedEvent(experiment_id="sweep"))
    server._on_state_event(job_state(LEARN_DONE))  # baseline
    assert notified(stdout) == ["ExperimentUpdatedEvent", "JobStateChangedEvent"]


# --- Web server


class FakeWebSocket:
    """Records the messages sent to a connection"""

    def __init__(self):
        self.messages: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.messages.append(message)

    def types(self) -> list[str]:
        types = [m["type"] for m in self.messages]
        self.messages.clear()
        return types


class Watcher:
    """Records the experiments a provider is told to watch"""

    def __init__(self, provider):
        self.watched = []
        watch = provider.watch

        def record(owner, experiments):
            self.watched.append(experiments)
            watch(owner, experiments)

        provider.watch = record


async def test_web_connections_receive_watched_events(provider):
    watcher = Watcher(provider)
    handler = WebSocketHandler(provider, "token")
    bridge = StateBridge(provider, handler)
    baseline, sweep = FakeWebSocket(), FakeWebSocket()
    for websocket in (baseline, sweep):
        handler.connections.add(websocket)
        handler.filters[websocket] = EventFilter()
        handler.filters[websocket].watch(set())

    await handler._handle_jobs(baseline, {"experimentId": "baseline"})
    await handler._handle_jobs(sweep, {"experimentId": "sweep"})
    baseline.types(), sweep.types()
    # The provider watches what the connections show
    assert watcher.watched[-1] == {"baseline", "sweep"}

    provider.submit("sweep", JobSpec("e-sweep", "pkg.Evaluate", "waiting"))
    await bridge._handle_event_async(
        JobSubmittedEvent(
            experiment_id="sweep", job_id="e-sweep", task_id="pkg.Evaluate"
        )
    )
    await bridge._handle_event_async(
        JobStateChangedEvent(job_id=EVAL_WAITING, task_id="pkg.Evaluate")
    )
    assert baseline.types() == []
    assert sweep.types() == ["job.add", "job.update"]

    # The started services are shown everywhere
    await bridge._handle_event_async(
        ServiceStateChangedEvent(
            experiment_id="sweep", service_id="tensorboard", state="RUNNING"
        )
    )
    assert baseline.types() == sweep.types() == ["service.update"]

    await handler.disconnect(sweep)
    assert watcher.watched[-1] == {"baseline"}
    bridge.close()


# --- SSH client


def test_ssh_client_drops_unwatched_job_queries():
    """The cached job answers of the experiments no longer watched would go
    stale: they are dropped, but not the services (live local instances)"""
    from experimaestro.scheduler.remote.client import SSHStateProviderClient

    client = SSHStateProviderClient(host="host", remote_workspace="/workspace")
    cache = client._query_cache
    for experiment_id in ("a", "b"):
        cache.put(("jobs", experiment_id), [], experiment_id=experiment_id)
        cache.put(("services", experiment_id, "run"), [], experiment_id=experiment_id)
    cache.put(("all_jobs",), [], experiment_id=None)
    cache.put(("experiments", None), [], experiment_id=None)

    client.watch("ui", {"a"})
    assert {key for key, _ in cache.entries()} == {
        ("jobs", "a"),
        ("services", "a", "run"),
        ("services", "b", "run"),
        ("experiments", None),
    }


# --- TUI


async def test_tui_watches_shown_experiment(provider):
    """The TUI watches the experiment it shows, none on the experiment list"""
    from .tui_driver import TuiDriver

    async with TuiDriver(provider) as ui:
        assert provider._watchers[ui.app] == frozenset()
        await ui.open_experiment("sweep")
        assert provider._watchers[ui.app] == {"sweep"}
        await ui.experiments()
        assert provider._watchers[ui.app] == frozenset()
    assert ui.app not in provider._watchers
