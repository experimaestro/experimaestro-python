"""Experiment dependency graph (not in the TUI)"""

import pytest

from experimaestro.notifications import LevelInformation
from experimaestro.scheduler.state_status import JobProgressEvent

from .scenario import EVAL_DONE, LEARN_DONE, LEARN_ERROR, LEARN_RUNNING, JobSpec

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


#: Identifiers of the init task and of the configuration added to the graph
INIT_TASK = "i1loader"
CONFIG = "k1dataset"


@pytest.fixture
def with_init_task(provider, monkeypatch):
    """The evaluation of the baseline has an init task and a configuration
    (which are not jobs)"""
    fetch = provider._fetch_run_objects

    def fetch_run_objects(experiment_id, run_id):
        objects = fetch(experiment_id, run_id)
        if experiment_id == "baseline":
            for o in objects:
                if o["identifier"] == EVAL_DONE:
                    o["init-tasks"] = [100]
                    o["fields"]["dataset"] = {"type": "python", "value": 101}
            objects.append(
                {"id": 100, "module": "pkg", "type": "Loader", "identifier": INIT_TASK}
            )
            objects.append(
                {"id": 101, "module": "pkg", "type": "Dataset", "identifier": CONFIG}
            )
        return objects

    monkeypatch.setattr(provider, "_fetch_run_objects", fetch_run_objects)


@pytest.mark.ui("browser")
async def test_graph_init_task_style(ui, with_init_task):
    """Init tasks are not drawn as configurations (dashed border)"""
    await ui.graph("baseline", tasks_only=False, stack=False)

    async def style(node_id: str) -> list[str]:
        node = ui.page.locator(f".xpm-dag-node[data-members~='{node_id}']")
        return await node.evaluate(
            """el => {
                const s = getComputedStyle(el);
                return [s.borderTopStyle, s.backgroundColor];
            }"""
        )

    assert await style(INIT_TASK) != await style(CONFIG)


@pytest.mark.ui("browser")
async def test_graph_progress(ui, provider):
    from playwright.async_api import expect

    """The node of a running job shows its progress (in place of the status
    bar)"""
    provider.jobs[LEARN_RUNNING].progress = [LevelInformation(0, None, 0.4)]
    await ui.graph("sweep")
    node = ui.page.locator(f".xpm-dag-node[data-members~='{LEARN_RUNNING}']")
    bar = node.get_by_role("progressbar")
    assert await bar.count() == 1
    assert float(await bar.get_attribute("aria-valuenow")) == pytest.approx(40)

    # Progress updates reach the graph without requesting it again
    start = len(ui.requests("experiment.graph"))
    job = provider.jobs[LEARN_RUNNING]
    job.progress = [LevelInformation(0, None, 0.7)]
    provider._notify_state_listeners(
        JobProgressEvent(job_id=job.identifier, task_id=job.task_id, progress=0.7)
    )
    await expect(bar).to_have_attribute("aria-valuenow", "70")
    assert ui.requests("experiment.graph")[start:] == []
