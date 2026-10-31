"""Experiment process crashes (see filesystem-protocol.md: I5, I7, I8)"""

import json

import pytest

from experimaestro.scheduler.cleanup import perform_workspace_cleanup
from experimaestro.scheduler.experiment import get_run_status
from experimaestro.tests.utils import is_posix

from .conftest import XP_NAME, JobFiles, wait_for
from .tasks import Controllable, Quick

pytestmark = pytest.mark.skipif(not is_posix(), reason="POSIX signals and flock")


def run_dirs(actors, name: str = XP_NAME):
    base = actors.workdir / "experiments" / name
    return sorted(
        path for path in base.iterdir() if path.is_dir() and not path.is_symlink()
    )


def run_events(run_dir) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((run_dir / "events").glob("event-*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_experiment_killed_job_continues(actors):
    """I7: a job survives the death of its experiment and is re-attached by
    the next run"""
    task = Controllable.C(x=1)
    job = JobFiles.of(actors.workdir, task)

    first = actors.experiment(task)
    wait_for(job.touch_path().exists, "the task to start")
    first.kill()
    first.wait()

    second = actors.experiment(task)
    job.release()
    assert actors.wait_exit(second) == 0, actors.logs()
    assert job.done.exists()
    assert job.runs == 1


@pytest.mark.deviation("D-14")
def test_crashed_run_events_stay_in_their_run(actors):
    """I8: the events of a crashed run never end up in the next run"""
    crashed_task, task = Quick.C(x=1), Quick.C(x=2)
    crashed_id = crashed_task.__xpm__.identifier.all.hex()

    first = actors.experiment(crashed_task, crashpoints="experiment.job_submitted=kill")
    actors.wait_exit(first)

    second = actors.experiment(task)
    assert actors.wait_exit(second) == 0, actors.logs()

    _, last_run = run_dirs(actors)
    leaked = [e for e in run_events(last_run) if e.get("job_id") == crashed_id]
    assert not leaked, f"events of the crashed run found in {last_run.name}"


def test_crashed_run_is_not_completed(actors):
    """I5: a crashed run is not reported as completed (history pruning and
    ``experiments runs prune`` rely on this status to delete runs)"""
    first = actors.experiment(Quick.C(x=1), crashpoints="experiment.job_submitted=kill")
    actors.wait_exit(first)

    (crashed_run,) = run_dirs(actors)
    assert get_run_status(crashed_run) != "completed"


def test_crash_before_event_archive_is_recovered(actors):
    """I7: a crash between the final status and the event archiving is
    recovered by the cleanup without asking the user"""
    first = actors.experiment(
        Quick.C(x=1), crashpoints="experiment.final_status_written=kill"
    )
    actors.wait_exit(first)
    assert list((actors.workdir / ".events" / "experiments").glob("*.jsonl"))

    warnings, _ = perform_workspace_cleanup(actors.workdir)
    assert not warnings, [w.description for w in warnings]
    assert not list((actors.workdir / ".events" / "experiments").glob("*.jsonl"))


def test_crash_after_submission_is_recovered(actors):
    """I7: a run killed after a submission is consolidated by the cleanup
    without asking the user"""
    first = actors.experiment(Quick.C(x=1), crashpoints="experiment.job_submitted=kill")
    actors.wait_exit(first)

    warnings, _ = perform_workspace_cleanup(actors.workdir)
    assert not warnings, [w.description for w in warnings]
    assert not list((actors.workdir / ".events" / "experiments").glob("*.jsonl"))
