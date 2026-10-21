"""Events received by the SSH state provider client

The client applies the events to its cache on reception, and notifies its
listeners in batches (every ``_notify_interval`` seconds).
"""

import threading
import time
from dataclasses import asdict

import pytest

from experimaestro.scheduler.remote.client import (
    SSHStateProviderClient,
    coalesce_events,
)
from experimaestro.scheduler.remote.protocol import (
    NotificationMethod,
    RPCNotification,
)
from experimaestro.scheduler.state_status import (
    EventBase,
    ExperimentJobStateEvent,
    ExperimentUpdatedEvent,
    JobProgressEvent,
    JobStateChangedEvent,
    JobSubmittedEvent,
    RunCompletedEvent,
    RunUpdatedEvent,
    ServiceAddedEvent,
    ServiceStateChangedEvent,
    WarningEvent,
)

pytestmark = [pytest.mark.state]


@pytest.fixture
def client():
    return SSHStateProviderClient(host="host", remote_workspace="/workspace")


def notify(client: SSHStateProviderClient, event: EventBase) -> None:
    """Make the client receive an event, as sent by the server"""
    client._handle_notification(
        RPCNotification(
            method=NotificationMethod.STATE_EVENT.value,
            params={"event_type": type(event).__name__, "data": asdict(event)},
        )
    )


def deliver(client: SSHStateProviderClient) -> list[EventBase]:
    """Run the notification loop until the pending events are delivered"""
    received: list[EventBase] = []
    client.add_listener(received.append)
    client._notify_interval = 0.01
    client._running = True
    thread = threading.Thread(target=client._notify_loop, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with client._pending_events_lock:
                if not client._pending_events:
                    break
            time.sleep(0.01)
        time.sleep(0.05)
    finally:
        client._running = False
        thread.join()
        client.remove_listener(received.append)
    return received


def job_state(job_id: str, scheduler_state: str, **kwargs) -> ExperimentJobStateEvent:
    return ExperimentJobStateEvent(
        experiment_id="exp",
        run_id="run",
        job_id=job_id,
        task_id="pkg.Task",
        scheduler_state=scheduler_state,
        **kwargs,
    )


# --- Throttling


def test_events_of_different_jobs_are_all_delivered(client):
    """Two jobs changing state in the same batch: both are notified, with the
    latest state of each"""
    notify(client, JobStateChangedEvent(job_id="j1", state="running"))
    notify(client, JobStateChangedEvent(job_id="j2", state="running"))
    notify(client, JobStateChangedEvent(job_id="j1", state="done"))

    received = deliver(client)
    assert [(e.job_id, e.state) for e in received] == [
        ("j2", "running"),
        ("j1", "done"),
    ]


def test_additions_are_never_collapsed(client):
    """Every submitted job (and added service) is notified, before the later
    state changes of the job"""
    notify(client, job_state("j1", "submitted"))
    notify(client, job_state("j2", "submitted"))
    notify(client, job_state("j1", "waiting"))
    notify(client, job_state("j1", "running"))
    notify(client, JobSubmittedEvent(experiment_id="exp", job_id="j3"))
    notify(client, JobSubmittedEvent(experiment_id="exp", job_id="j4"))
    notify(client, ServiceAddedEvent(experiment_id="exp", service_id="s1"))
    notify(client, ServiceAddedEvent(experiment_id="exp", service_id="s2"))

    received = deliver(client)
    assert [
        (
            type(e).__name__,
            e.service_id if isinstance(e, ServiceAddedEvent) else e.job_id,
        )
        for e in received
    ] == [
        ("ExperimentJobStateEvent", "j1"),
        ("ExperimentJobStateEvent", "j2"),
        ("ExperimentJobStateEvent", "j1"),
        ("JobSubmittedEvent", "j3"),
        ("JobSubmittedEvent", "j4"),
        ("ServiceAddedEvent", "s1"),
        ("ServiceAddedEvent", "s2"),
    ]
    assert [e.scheduler_state for e in received[:3]] == [
        "submitted",
        "submitted",
        "running",
    ]


def test_coalesce_per_entity():
    """Events are collapsed per experiment, run, job or service; warnings and
    completed runs are all kept"""
    events = [
        ExperimentUpdatedEvent(experiment_id="a"),
        ExperimentUpdatedEvent(experiment_id="b"),
        ExperimentUpdatedEvent(experiment_id="a"),
        RunUpdatedEvent(experiment_id="a", run_id="r1"),
        RunUpdatedEvent(experiment_id="a", run_id="r2"),
        RunUpdatedEvent(experiment_id="a", run_id="r1"),
        ServiceStateChangedEvent(experiment_id="a", service_id="s", state="X"),
        ServiceStateChangedEvent(experiment_id="b", service_id="s", state="X"),
        ServiceStateChangedEvent(experiment_id="a", service_id="s", state="Y"),
        JobProgressEvent(job_id="j1", progress=0.1),
        JobProgressEvent(job_id="j1", progress=0.2),
        JobStateChangedEvent(job_id="j1", state="running"),
        job_state("j1", "running"),
        WarningEvent(experiment_id="a", warning_key="w"),
        WarningEvent(experiment_id="a", warning_key="w"),
        RunCompletedEvent(experiment_id="a", run_id="r1"),
        RunCompletedEvent(experiment_id="a", run_id="r1"),
    ]
    kept = coalesce_events(events)
    indices = {id(e): i for i, e in enumerate(events)}
    assert [indices[id(e)] for e in kept] == [
        1,
        2,
        4,
        5,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]
