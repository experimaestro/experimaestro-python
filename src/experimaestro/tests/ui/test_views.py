"""Sorting and grouping the job list

Both are done by the client in the web UI, so these tests do not apply to the
protocol-level driver.
"""

import pytest

from experimaestro.scheduler.job_status import status_rank

from .scenario import EVAL_DONE, LEARN_DONE, LEARN_ERROR

pytestmark = [pytest.mark.anyio, pytest.mark.ui("tui", "browser")]


async def test_sort_by_task(ui, provider):
    await ui.open_experiment("baseline")
    await ui.sort_by("task")
    order = await ui.job_order()
    assert set(order) == {LEARN_DONE, LEARN_ERROR, EVAL_DONE}
    tasks = [provider.jobs[job_id].task_id for job_id in order]
    assert tasks == sorted(tasks)


@pytest.mark.parametrize("experiment_id", ["baseline", "sweep"])
async def test_sort_by_status(ui, provider, experiment_id):
    """Both interfaces use the shared status order (scheduler/job_status.py)"""
    await ui.open_experiment(experiment_id)
    await ui.sort_by("status")
    ranks = [
        status_rank(provider.jobs[job_id].state) for job_id in await ui.job_order()
    ]
    assert len(set(ranks)) > 1
    assert ranks == sorted(ranks)


async def test_group_by_task(ui):
    await ui.open_experiment("baseline")
    assert await ui.group_by("task") == {
        "pkg.Evaluate": {EVAL_DONE},
        "pkg.Learn": {LEARN_DONE, LEARN_ERROR},
    }


async def test_group_by_status(ui):
    await ui.open_experiment("baseline")
    assert await ui.group_by("status") == {
        "done": {LEARN_DONE, EVAL_DONE},
        "error": {LEARN_ERROR},
    }
