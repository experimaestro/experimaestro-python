"""Experiment dependency graph (not in the TUI)"""

import pytest

from .scenario import EVAL_DONE, LEARN_DONE, LEARN_ERROR, JobSpec

pytestmark = [pytest.mark.anyio, pytest.mark.ui("web", "browser")]


async def test_graph(ui):
    assert await ui.graph("baseline") == {
        frozenset({LEARN_DONE}),
        frozenset({LEARN_ERROR}),
        frozenset({EVAL_DONE}),
    }


@pytest.mark.ui("browser")
async def test_graph_requests_while_jobs_arrive(ui, provider):
    """Jobs streaming in do not each re-request the graph

    Only the jobs of the displayed experiment change its graph, and a burst of
    them is answered with a single request.
    """
    await ui.graph("baseline")
    start = len(ui.requests("experiment.graph"))
    for ix in range(5):
        for experiment_id in ("sweep", "baseline"):
            provider.submit(
                experiment_id,
                JobSpec(f"e{ix}{experiment_id}", "pkg.Evaluate", "waiting"),
            )
    await ui.settle(quiet=2000)
    requests = ui.requests("experiment.graph")[start:]
    assert [r["experimentId"] for r in requests] == ["baseline"]
