"""Killing and deleting jobs and experiments

Both interfaces go through ``scheduler/job_operations.py``, so they accept and
refuse the same operations and report them with the same messages.
"""

import pytest

from .drivers import Notification, refused
from .scenario import (
    EVAL_DONE,
    LEARN_DONE,
    LEARN_ERROR,
    LEARN_RUNNING,
    LEARN_STUCK,
    ORPHAN_DATA_SIZE,
    ORPHAN_DONE,
    STRAY_RUNNING,
)

pytestmark = pytest.mark.anyio


# --- Jobs


async def test_kill_running_job(ui, provider):
    await ui.open_experiment("sweep")
    assert await ui.kill_job(LEARN_RUNNING) == Notification(
        "info", f"Job {LEARN_RUNNING} killed"
    )
    assert provider.killed == [LEARN_RUNNING]


async def test_kill_failure_is_reported(ui, provider):
    await ui.open_experiment("sweep")
    assert await ui.kill_job(LEARN_STUCK) == Notification(
        "error", f"Failed to kill job {LEARN_STUCK}: No process found for job"
    )
    assert provider.killed == []


async def test_kill_refuses_finished_job(ui, provider):
    await ui.open_experiment("baseline")
    assert refused(await ui.kill_job(LEARN_DONE), f"Job {LEARN_DONE} is not running")
    assert provider.killed == []


async def test_delete_finished_job(ui, provider):
    await ui.open_experiment("baseline")
    assert await ui.delete_job(LEARN_ERROR) == Notification(
        "info", f"Deleted job {LEARN_ERROR}"
    )
    assert LEARN_ERROR not in provider.jobs
    assert provider.partials_cleanups == 1
    assert await ui.visible_jobs() == {LEARN_DONE, EVAL_DONE}


async def test_delete_refuses_running_job(ui, provider):
    await ui.open_experiment("sweep")
    assert await ui.delete_job(LEARN_RUNNING) == Notification(
        "warning", f"Cannot delete job {LEARN_RUNNING} while it is running"
    )
    assert LEARN_RUNNING in provider.jobs


# --- Experiments


async def test_kill_experiment(ui, provider):
    # One of the two running jobs cannot be killed
    assert await ui.kill_experiment("sweep") == Notification(
        "warning", "Killed 1 of 2 jobs (1 failed)"
    )
    assert provider.killed == [LEARN_RUNNING]


async def test_kill_experiment_without_running_jobs(ui, provider):
    assert await ui.kill_experiment("baseline") == Notification(
        "warning", "No running jobs in experiment"
    )
    assert provider.killed == []


async def test_delete_experiment(ui, provider):
    assert await ui.delete_experiment("baseline") == Notification(
        "info", "Experiment baseline deleted"
    )
    assert set(await ui.experiments()) == {"sweep", "ablation"}
    # Job directories are kept
    assert LEARN_DONE in provider.jobs


async def test_delete_experiment_refuses_running_one(ui, provider):
    assert refused(
        await ui.delete_experiment("sweep"),
        "Cannot delete: experiment is still running",
    )
    assert set(await ui.experiments()) == {"baseline", "sweep", "ablation"}


# --- Orphans


async def test_orphans_are_listed_with_size(ui):
    orphans = await ui.orphans()
    assert set(orphans) == {ORPHAN_DONE, STRAY_RUNNING}
    assert orphans[STRAY_RUNNING].stray
    assert not orphans[ORPHAN_DONE].stray
    # The data file, plus the two log files
    assert orphans[ORPHAN_DONE].size >= ORPHAN_DATA_SIZE


async def test_kill_stray_job(ui, provider):
    await ui.orphans()
    assert await ui.kill_orphan(STRAY_RUNNING) == Notification(
        "info", f"Job {STRAY_RUNNING} killed"
    )
    assert provider.killed == [STRAY_RUNNING]


async def test_delete_orphan(ui, provider):
    await ui.orphans()
    assert await ui.delete_orphan(ORPHAN_DONE) == Notification(
        "info", f"Deleted job {ORPHAN_DONE}"
    )
    assert provider.partials_cleanups == 1
    assert set(await ui.orphans()) == {STRAY_RUNNING}


async def test_delete_all_orphans(ui, provider):
    await ui.orphans()
    # The running (stray) job is left alone
    assert await ui.delete_all_orphans() == Notification("info", "Deleted 1 of 1 jobs")
    assert provider.partials_cleanups == 1
    assert set(await ui.orphans()) == {STRAY_RUNNING}


async def test_kill_all_strays(ui, provider):
    await ui.orphans()
    assert await ui.kill_all_strays() == Notification("info", "Killed 1 of 1 jobs")
    assert provider.killed == [STRAY_RUNNING]
