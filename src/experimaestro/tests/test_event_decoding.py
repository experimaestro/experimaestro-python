"""Validated decoding of events (EventBase.from_dict)"""

import json

import pytest

from experimaestro.scheduler.state_status import (
    EventBase,
    ExperimentJobStateEvent,
    InvalidEventError,
    JobStateChangedEvent,
    JobTag,
    ProgressLevel,
)


def roundtrip(event: EventBase) -> EventBase:
    return EventBase.from_dict(json.loads(event.to_json()))


def test_nested_dataclasses_are_rebuilt():
    event = roundtrip(
        ExperimentJobStateEvent(
            job_id="j", experiment_id="xp", tags=[JobTag("k", "v")], depends_on=["d"]
        )
    )
    assert isinstance(event, ExperimentJobStateEvent)
    assert event.tags == [JobTag("k", "v")]
    assert event.depends_on == ["d"]

    event = roundtrip(
        JobStateChangedEvent(job_id="j", progress=[ProgressLevel(1, 0.5, "x")])
    )
    assert event.progress == [ProgressLevel(1, 0.5, "x")]


def test_unknown_fields_are_ignored():
    """Events written by another version can have more fields"""
    event = EventBase.from_dict(
        {"event_type": "JobStateChangedEvent", "job_id": "j", "added_later": 1}
    )
    assert isinstance(event, JobStateChangedEvent)
    assert event.job_id == "j"


def test_unknown_event_type():
    assert type(EventBase.from_dict({"event_type": "NoSuchEvent"})) is EventBase


@pytest.mark.parametrize(
    "data",
    [
        {"event_type": "JobStateChangedEvent", "retry_count": "many"},
        {"event_type": "ExperimentJobStateEvent", "tags": [{"key": "k"}]},
        {"event_type": "JobStateChangedEvent", "progress": "half"},
    ],
)
def test_invalid_events_are_rejected(data):
    with pytest.raises(InvalidEventError):
        EventBase.from_dict(data)


def test_invalid_event_is_a_value_error():
    """Readers skip bad lines by catching ValueError (JSON or event errors)"""
    assert issubclass(InvalidEventError, ValueError)
