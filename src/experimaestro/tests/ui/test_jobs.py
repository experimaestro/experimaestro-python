"""Browsing experiments and jobs"""

import pytest

from .scenario import (
    EVAL_DONE,
    EVAL_WAITING,
    LEARN_DONE,
    LEARN_ERROR,
    LEARN_RUNNING,
    LEARN_STUCK,
    OLD_LEARN,
    stderr_text,
    stdout_text,
)

pytestmark = pytest.mark.anyio


async def test_experiments_are_listed(ui):
    assert set(await ui.experiments()) == {"baseline", "sweep", "ablation"}


async def test_experiment_status(ui):
    """Same status in both interfaces (BaseExperiment.status_category)"""
    assert await ui.experiments() == {
        "baseline": "failed",
        "sweep": "running",
        # Stopped before all its jobs finished
        "ablation": "stopped",
    }


@pytest.mark.ui("tui", "browser")
async def test_experiment_sort(ui):
    """Newest first by default; the key of the current sort reverses it"""
    assert await ui.experiment_order() == ["ablation", "sweep", "baseline"]

    await ui.sort_experiments("started")
    assert await ui.experiment_order() == ["baseline", "sweep", "ablation"]

    # Running first, then failed, then stopped
    await ui.sort_experiments("status")
    assert await ui.experiment_order() == ["sweep", "baseline", "ablation"]

    await ui.sort_experiments("id")
    assert await ui.experiment_order() == ["ablation", "baseline", "sweep"]
    await ui.sort_experiments("id")
    assert await ui.experiment_order() == ["sweep", "baseline", "ablation"]


@pytest.mark.ui("tui", "browser")
async def test_experiment_search(ui):
    """The search keeps the experiments whose ID contains the text (ignoring
    case)"""
    await ui.search_experiments("LINE")
    assert await ui.experiment_order() == ["baseline"]

    await ui.search_experiments("e")
    assert await ui.experiment_order() == ["sweep", "baseline"]

    await ui.search_experiments("")
    assert set(await ui.experiment_order()) == {"baseline", "sweep", "ablation"}


async def test_jobs_of_current_run(ui):
    await ui.open_experiment("baseline")
    assert await ui.visible_jobs() == {LEARN_DONE, LEARN_ERROR, EVAL_DONE}

    await ui.open_experiment("sweep")
    assert await ui.visible_jobs() == {LEARN_RUNNING, EVAL_WAITING, LEARN_STUCK}


async def test_run_history(ui):
    assert await ui.runs("baseline") == ["20260101_100000", "20260102_090000"]

    await ui.open_experiment("baseline", "20260101_100000")
    assert await ui.visible_jobs() == {OLD_LEARN}


async def test_filter_by_tag_and_state(ui):
    await ui.open_experiment("baseline")
    assert await ui.set_filter("model = 'bm25'") is None
    assert await ui.visible_jobs() == {LEARN_DONE, EVAL_DONE}

    assert await ui.set_filter("@state = 'error'") is None
    assert await ui.visible_jobs() == {LEARN_ERROR}


async def test_filter_regex_and_in(ui):
    await ui.open_experiment("baseline")
    # Anchored at the start of the task identifier
    assert await ui.set_filter("@name ~ 'Eval'") is None
    assert await ui.visible_jobs() == set()
    assert await ui.set_filter("@name ~ 'pkg\\.Eval'") is None
    assert await ui.visible_jobs() == {EVAL_DONE}

    assert await ui.set_filter("model in ['splade', 'dense']") is None
    assert await ui.visible_jobs() == {LEARN_ERROR}


async def test_filter_uses_tags_of_displayed_run(ui):
    await ui.open_experiment("sweep")
    assert await ui.set_filter("model = 'dense'") is None
    assert await ui.visible_jobs() == {LEARN_STUCK}


async def test_invalid_filter_reports_error(ui):
    await ui.open_experiment("baseline")
    error = await ui.set_filter("@state = ")
    assert error is not None and error.startswith("Invalid query")


async def test_job_detail(ui):
    await ui.open_experiment("baseline")
    detail = await ui.job_detail(EVAL_DONE)
    assert detail.task_id == "pkg.Evaluate"
    assert detail.state == "done"
    assert detail.tags == {"model": "bm25"}
    assert detail.dependencies == {LEARN_DONE: "done"}


async def test_job_detail_dependency_state(ui):
    await ui.open_experiment("sweep")
    detail = await ui.job_detail(EVAL_WAITING)
    assert detail.state == "waiting"
    assert detail.dependencies == {LEARN_RUNNING: "running"}


async def test_job_logs(ui):
    await ui.open_experiment("baseline")
    assert await ui.job_log(LEARN_ERROR, "stdout") == stdout_text(LEARN_ERROR)
    assert await ui.job_log(LEARN_ERROR, "stderr") == stderr_text(LEARN_ERROR)


@pytest.mark.ui("browser")
async def test_jobs_requested_on_demand(ui):
    """The page reads the jobs of the experiment it shows, from the cache: it
    does not ask the server to re-read every experiment (slow over SSH)"""
    await ui.open_experiment("sweep")
    assert ui.requests("refresh") == []
    requested = {r.get("experimentId") for r in ui.requests("jobs")}
    assert "sweep" in requested
    assert None not in requested
