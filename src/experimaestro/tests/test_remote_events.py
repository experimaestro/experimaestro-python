"""Events received by the SSH state provider client

The client applies the events to its cache on reception, and notifies its
listeners in batches (every ``_notify_interval`` seconds).
"""

import io
import threading
import time
from dataclasses import asdict

import pytest

from experimaestro.scheduler.remote.client import (
    SSHStateProviderClient,
    coalesce_events,
)
from experimaestro.scheduler.interfaces import JobState
from experimaestro.scheduler.remote.protocol import (
    NotificationMethod,
    RPCNotification,
    RPCResponse,
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


# --- Job counts of the experiment list


@pytest.fixture
def server_experiments(client, tmp_path):
    """The client reads one running experiment from the (fake) server"""
    client.local_cache_dir = tmp_path
    calls = []

    def call_sync(method, params, timeout=None, decode=None):
        calls.append(method)
        result = [
            {
                "experiment_id": "exp",
                "run_id": "run",
                "status": "running",
                "job_states": {"j1": "running", "j2": "done", "j3": "error"},
                "total_jobs": 3,
                "finished_jobs": 1,
                "failed_jobs": 1,
            }
        ]
        # Answered at once: no event is received meanwhile
        return decode(result, client._events_received)

    client._call_sync = call_sync
    return calls


def counts(client) -> tuple[int, int, int]:
    (exp,) = client.get_experiments()
    return exp.total_jobs, exp.finished_jobs, exp.failed_jobs


def test_job_events_update_the_experiment_counts(client, server_experiments):
    """The job counts of the cached experiments follow the scheduler events,
    without reading the experiments again"""
    assert counts(client) == (3, 1, 1)

    notify(client, job_state("j1", "done"))
    assert counts(client) == (3, 2, 1)

    # A job is submitted, and fails
    notify(client, job_state("j4", "submitted"))
    assert counts(client) == (4, 2, 1)
    notify(client, job_state("j4", "error", failure_reason="FAILED"))
    assert counts(client) == (4, 2, 2)

    # A failed job is restarted
    notify(client, job_state("j3", "waiting"))
    assert counts(client) == (4, 2, 1)

    # Events of another run are ignored
    notify(
        client,
        ExperimentJobStateEvent(
            experiment_id="exp", run_id="other", job_id="j5", scheduler_state="done"
        ),
    )
    assert counts(client) == (4, 2, 1)

    assert len(server_experiments) == 1


def test_read_experiment_updates_job_states(client, server_experiments):
    """Reading an experiment again updates the job states the events are
    applied to (an event already counted is not counted twice)"""
    assert counts(client) == (3, 1, 1)
    notify(client, job_state("j1", "done"))
    assert counts(client) == (3, 2, 1)

    # The server state is read again... before the event telling j1 is done
    client.get_experiments(refresh=True)
    assert counts(client) == (3, 1, 1)
    server_experiments.clear()

    def call_sync(method, params, timeout=None, decode=None):
        result = [
            {
                "experiment_id": "exp",
                "run_id": "run",
                "status": "running",
                "job_states": {"j1": "done", "j2": "done", "j3": "error"},
                "total_jobs": 3,
            }
        ]
        return decode(result, client._events_received)

    client._call_sync = call_sync
    client.get_experiments(refresh=True)
    assert counts(client) == (3, 2, 1)
    notify(client, job_state("j1", "done"))
    assert counts(client) == (3, 2, 1)


# --- Answers and events about the same jobs


def job_dict(state: str) -> dict:
    return {"job_id": "j1", "task_id": "pkg.Task", "state": state}


@pytest.fixture
def connected(client, tmp_path):
    """The client, sending its requests to a server that the test plays"""
    client.local_cache_dir = tmp_path
    client._stdin = io.BytesIO()
    client._session_open = True
    return client


def asking_jobs(client) -> tuple[threading.Thread, list]:
    """Asks for the jobs of the experiment from another thread, as the UIs
    do: returns once the request is sent"""
    answers: list = []
    sent = len(client._stdin.getvalue())
    thread = threading.Thread(
        target=lambda: answers.append(client.get_jobs("exp", refresh=True))
    )
    thread.start()
    deadline = time.monotonic() + 5
    while len(client._stdin.getvalue()) == sent:
        assert time.monotonic() < deadline, "request not sent"
        time.sleep(0.001)
    return thread, answers


def answer(client, result) -> None:
    """Make the client receive the answer to its last request"""
    client._handle_response(RPCResponse(id=client._request_id, result=result))


def test_event_received_after_the_answer(connected):
    """The server answers, then sends an event: the event is applied after
    the answer, even if the thread that asked reads the answer later"""
    thread, answers = asking_jobs(connected)
    answer(connected, [job_dict("running")])
    notify(connected, JobStateChangedEvent(job_id="j1", state="done"))
    thread.join(5)

    ((job,),) = answers
    assert job.state == JobState.DONE

    # ... also when the job was known before
    thread, answers = asking_jobs(connected)
    answer(connected, [job_dict("running")])
    notify(connected, JobStateChangedEvent(job_id="j1", state="done"))
    thread.join(5)
    assert answers == [[job]]
    assert job.state == JobState.DONE


def test_event_received_before_the_answer(connected):
    """An event sent while the server answers (with the state it read
    before) is applied again to the answer"""
    thread, answers = asking_jobs(connected)
    answer(connected, [job_dict("scheduled")])
    thread.join(5)
    ((job,),) = answers

    thread, answers = asking_jobs(connected)
    notify(connected, JobStateChangedEvent(job_id="j1", state="running"))
    notify(connected, JobProgressEvent(job_id="j1", progress=0.5))
    assert job.state == JobState.RUNNING
    answer(connected, [job_dict("scheduled")])
    thread.join(5)

    assert answers == [[job]]
    assert job.state == JobState.RUNNING
    assert job.progress[0].progress == 0.5

    # The events received before the request are not: the answer is newer
    thread, answers = asking_jobs(connected)
    answer(connected, [job_dict("done")])
    thread.join(5)
    assert job.state == JobState.DONE


def test_experiment_event_received_before_the_answer(connected):
    """Same for the events about the jobs of an experiment"""
    answers: list = []
    thread = threading.Thread(
        target=lambda: answers.append(connected.get_experiment("exp"))
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not connected._stdin.getvalue():
        assert time.monotonic() < deadline, "request not sent"
        time.sleep(0.001)

    notify(connected, job_state("j1", "done"))
    answer(
        connected,
        {
            "experiment_id": "exp",
            "run_id": "run",
            "status": "running",
            "job_states": {"j1": "running", "j2": "done"},
            "total_jobs": 2,
        },
    )
    thread.join(5)

    (exp,) = answers
    assert (exp.total_jobs, exp.finished_jobs, exp.failed_jobs) == (2, 2, 0)
