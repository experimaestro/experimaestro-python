"""Browsing experiments and jobs"""

import pytest

from .scenario import (
    EVAL_DEP_FAILED,
    EVAL_DONE,
    EVAL_WAITING,
    LEARN_DONE,
    LEARN_ERROR,
    LEARN_FAILED,
    LEARN_FAILED_MARKER,
    LEARN_TIMEOUT,
    LEARN_TIMEOUT_MARKER,
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


@pytest.mark.ui("tui", "browser")
async def test_experiment_search_tags(ui):
    """The search also keeps the experiments with a tag containing the text"""
    await ui.search_experiments("BM2")
    assert await ui.experiment_order() == ["baseline"]

    await ui.search_experiments("pap")
    assert await ui.experiment_order() == ["sweep", "baseline"]


async def test_experiment_tags(ui):
    assert await ui.experiment_tags() == {
        "baseline": {"paper", "bm25"},
        "sweep": {"paper"},
        "ablation": set(),
    }


@pytest.mark.ui("tui", "browser")
async def test_hide_tags(ui):
    """Hidden tag keys are not shown for any job, until shown again"""
    await ui.open_experiment("baseline")
    assert await ui.hide_tags({"model"}) == {
        LEARN_DONE: set(),
        LEARN_ERROR: set(),
        EVAL_DONE: set(),
    }
    assert await ui.hide_tags(set()) == {
        LEARN_DONE: {"model"},
        LEARN_ERROR: {"model"},
        EVAL_DONE: {"model"},
    }


@pytest.mark.ui("tui", "browser")
async def test_experiment_grouping(ui):
    """An experiment is in the group of each of its tags, or in the untagged
    group"""
    assert await ui.group_experiments("tag") == {
        "bm25": {"baseline"},
        "paper": {"baseline", "sweep"},
        "(untagged)": {"ablation"},
    }
    # No separator in the IDs: each experiment is its own prefix
    assert await ui.group_experiments("prefix") == {
        "ablation": {"ablation"},
        "baseline": {"baseline"},
        "sweep": {"sweep"},
    }
    assert await ui.group_experiments("none") == {"": {"baseline", "sweep", "ablation"}}


async def test_group_carbon(ui):
    """The header of a group shows the CO2 of all the runs of its experiments"""
    assert await ui.group_carbon() == {
        "bm25": 3.5,
        "paper": 7.5,
        "(untagged)": 8.0,
    }


async def test_jobs_of_current_run(ui):
    await ui.open_experiment("baseline")
    assert await ui.visible_jobs() == {LEARN_DONE, LEARN_ERROR, EVAL_DONE}

    await ui.open_experiment("sweep")
    assert await ui.visible_jobs() == {LEARN_RUNNING, EVAL_WAITING, LEARN_STUCK}


async def test_run_history(ui):
    assert await ui.runs("baseline") == ["20260101_100000", "20260102_090000"]

    await ui.open_experiment("baseline", "20260101_100000")
    assert await ui.visible_jobs() == {
        OLD_LEARN,
        LEARN_FAILED,
        EVAL_DEP_FAILED,
        LEARN_TIMEOUT,
    }


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


@pytest.mark.ui("web")
async def test_filter_completion(ui):
    """Fields, operators and values (from the jobs displayed) are suggested"""
    await ui.open_experiment("sweep")
    assert await ui.filter_completions("") == ["@state", "@name", "model"]
    assert await ui.filter_completions("@st") == ["@state"]
    assert await ui.filter_completions("model ") == ["=", "~", "in", "not in"]
    assert "'running'" in await ui.filter_completions("@state = ")
    # Tag values of the displayed run only
    assert await ui.filter_completions("model = ") == ["'bm25'", "'dense'"]
    assert await ui.filter_completions("@name = 'pkg.E") == ["'pkg.Evaluate'"]

    await ui.open_experiment("baseline")
    assert await ui.filter_completions("model = ") == ["'bm25'", "'splade'"]


@pytest.mark.ui("web")
async def test_filter_completion_accept(ui):
    await ui.open_experiment("baseline")
    assert await ui.accept_filter_completion("mo", "model") == "model "
    assert await ui.accept_filter_completion("model ", "in") == "model in ["
    query = await ui.accept_filter_completion("model = 's", "'splade'")
    assert query == "model = 'splade' "
    assert await ui.set_filter(query) is None
    assert await ui.visible_jobs() == {LEARN_ERROR}


@pytest.mark.ui("browser")
async def test_filter_completion_while_typing(ui):
    """Suggestions open while typing; Enter accepts, Escape closes them (and
    then clears the filter)"""
    await ui.open_experiment("baseline")
    search = ui.page.locator("#searchjobs")
    options = ui.page.locator("#searchjobs-completions [role=option]")
    await search.press_sequentially("@sta")
    await options.first.wait_for()
    assert await options.all_inner_texts() == ["@state\njob state"]
    await search.press("Enter")
    # Then the operators are suggested
    await ui.settle()
    assert await search.input_value() == "@state "
    assert await options.count() == 4
    await search.press("Escape")
    await ui.settle()
    assert await options.count() == 0
    assert await search.input_value() == "@state "
    await search.press("Escape")
    assert await search.input_value() == ""


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


async def test_job_detail_failure_reason(ui):
    await ui.open_experiment("baseline", "20260101_100000")
    detail = await ui.job_detail(LEARN_FAILED)
    assert (detail.state, detail.failure) == ("error", "FAILED")

    detail = await ui.job_detail(EVAL_DEP_FAILED)
    assert (detail.state, detail.failure) == ("error", "DEPENDENCY")
    assert detail.dependencies == {LEARN_FAILED: "error"}
    assert detail.dependency_failures == {LEARN_FAILED: "FAILED"}
    # No marker: nothing more than the reason
    assert detail.failure_message is None


async def test_job_detail_failure_details(ui):
    # What happened, as recorded in the .failed marker
    await ui.open_experiment("baseline", "20260101_100000")
    detail = await ui.job_detail(LEARN_FAILED)
    assert detail.failure_message == LEARN_FAILED_MARKER["message"]
    assert detail.failure_location == LEARN_FAILED_MARKER["location"]
    assert detail.failure_job_message is None


async def test_job_detail_launcher_failure(ui):
    # The reason given by the launcher, and what the job itself reported
    await ui.open_experiment("baseline", "20260101_100000")
    detail = await ui.job_detail(LEARN_TIMEOUT)
    assert (detail.state, detail.failure) == ("error", "TIMEOUT")
    assert detail.failure_message == LEARN_TIMEOUT_MARKER["message"]
    assert detail.failure_job_message == LEARN_TIMEOUT_MARKER["job"]["message"]


async def test_job_detail_follow_dependency(ui):
    # From a job failed because of a dependency to the source of the failure
    await ui.open_experiment("baseline", "20260101_100000")
    detail = await ui.follow_dependency(EVAL_DEP_FAILED, LEARN_FAILED)
    assert detail.task_id == "pkg.Learn"
    assert (detail.state, detail.failure) == ("error", "FAILED")


@pytest.mark.ui("tui", "browser")
async def test_job_detail_back_from_dependency(ui):
    """Going back from the detail of a followed dependency shows the job it
    was followed from, then closes the detail view"""
    await ui.open_experiment("baseline", "20260101_100000")
    detail = await ui.back_from_dependency(EVAL_DEP_FAILED, LEARN_FAILED)
    assert detail is not None and detail.job_id == EVAL_DEP_FAILED


async def test_no_logs_before_start(ui):
    """A job that has not started has no logs to show, even if log files of an
    earlier run are there"""
    await ui.open_experiment("sweep")
    assert not await ui.logs_offered(EVAL_WAITING)
    assert await ui.logs_offered(LEARN_RUNNING)


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
