"""Selection of the state events a UI client needs

A client (a web page, or the remote SSH client) only shows some experiments:
it does not need the job events of the others. The filter is applied where
the events are sent — by the SSH server, so that they do not cross the
network, and for each web socket connection.

What a client watching some experiments receives:

- experiment-level events (experiment list, runs, warnings, errors): always,
  the experiment list is shown on every page;
- job events: for the jobs of the watched experiments — those known when the
  watch is declared, and those submitted to them afterwards (job state events
  do not say which experiments the job belongs to);
- action events: for the watched experiments;
- service state events: for the watched experiments, and for the started
  services (shown on every page), up to the event telling they stopped;
- added services: always — they are rare (once per service and run), and keep
  the service lists cached by the clients complete;
- with ``job_counts``, the scheduler events of the other experiments' jobs
  that change the job counts shown by the experiment list (a job submitted,
  a job entering or leaving the done or error state) — but not the job
  events that follow.
"""

import threading
from collections.abc import Iterable

from experimaestro.scheduler.interfaces import STATE_NAME_TO_JOBSTATE, JobState
from experimaestro.scheduler.state_status import (
    ActionAddedEvent,
    EventBase,
    ExperimentJobStateEvent,
    JobEventBase,
    JobSubmittedEvent,
    ServiceStateChangedEvent,
)

#: Service states shown by the started services (on every page)
STARTED_SERVICE_STATES = frozenset({"STARTING", "RUNNING", "STOPPING"})


def count_category(event: ExperimentJobStateEvent) -> str | None:
    """Which job count of its experiment a scheduler state counts in"""
    if event.failure_reason:
        return "failed"
    state = STATE_NAME_TO_JOBSTATE.get(event.scheduler_state)
    if state == JobState.DONE:
        return "finished"
    if state is not None and state.is_error():
        return "failed"
    return None


class EventFilter:
    """Selects the events of the experiments a client watches

    Thread-safe: the watch is declared by the client requests, and events are
    filtered from the thread that produces them.

    Args:
        job_counts: Also accept the scheduler events that change the job
            counts of the experiments that are not watched (for clients that
            keep these counts up to date from the events, like the SSH client)
    """

    def __init__(self, *, job_counts: bool = False):
        #: Watched experiments (None: all of them)
        self.experiments: frozenset[str] | None = None
        self.job_counts = job_counts
        #: Jobs of the watched experiments
        self._jobs: set[str] = set()
        #: Started services (experiment ID, service ID), whatever the experiment
        self._started_services: set[tuple[str, str]] = set()
        #: Count category of the jobs (see count_category), by experiment, run
        #: and job ID, as last told by their scheduler events (job_counts)
        self._job_categories: dict[tuple[str, str, str], str | None] = {}
        self._lock = threading.Lock()

    def watch(self, experiments: Iterable[str] | None, jobs: Iterable[str] = ()):
        """Watch some experiments (None: all of them)

        Args:
            experiments: The watched experiment IDs
            jobs: The identifiers of the jobs of these experiments
        """
        with self._lock:
            self.experiments = None if experiments is None else frozenset(experiments)
            self._jobs = set(jobs)

    def add_jobs(self, jobs: Iterable[str]) -> None:
        """Jobs of the watched experiments, known after the watch was declared"""
        with self._lock:
            self._jobs.update(jobs)

    def accepts(self, event: EventBase) -> bool:
        """Whether the client needs an event"""
        with self._lock:
            experiments = self.experiments

            # Started services are tracked whatever is watched
            if isinstance(event, ServiceStateChangedEvent):
                key = (event.experiment_id, event.service_id)
                was_started = key in self._started_services
                if event.state in STARTED_SERVICE_STATES:
                    self._started_services.add(key)
                else:
                    self._started_services.discard(key)
                return (
                    experiments is None
                    or event.experiment_id in experiments
                    or event.state in STARTED_SERVICE_STATES
                    or was_started
                )

            changes_counts = self._changes_counts(event)
            if experiments is None:
                return True

            match event:
                case JobSubmittedEvent() | ExperimentJobStateEvent():
                    if event.experiment_id not in experiments:
                        return changes_counts
                    self._jobs.add(event.job_id)
                    return True
                case JobEventBase():
                    return event.job_id in self._jobs
                case ActionAddedEvent():
                    return event.experiment_id in experiments
            return True

    def _changes_counts(self, event: EventBase) -> bool:
        """Whether a job event changes the job counts of its experiment

        Records the count category of the job: called (with the lock held)
        for every event, whatever is watched, so that it is up to date when
        the experiment stops being watched.
        """
        if not self.job_counts:
            return False

        match event:
            case JobSubmittedEvent():
                key = (event.experiment_id, event.run_id, event.job_id)
                if key in self._job_categories:
                    return False
                self._job_categories[key] = None
                return True
            case ExperimentJobStateEvent():
                key = (event.experiment_id, event.run_id, event.job_id)
                category = count_category(event)
                if key in self._job_categories:
                    if self._job_categories[key] == category:
                        return False
                self._job_categories[key] = category
                return True
        return False
