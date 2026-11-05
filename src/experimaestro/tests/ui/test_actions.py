"""Actions declared by the experiment runs

Each run declares its own actions: the interfaces list those of the run shown
(the current run by default). The action of a former run can be executed, once
the warning that it comes from a former run is confirmed.
"""

import pytest

from experimaestro.scheduler.action_operations import PAST_RUN_WARNING

pytestmark = pytest.mark.anyio

FORMER = "20260101_100000"
CURRENT = "20260102_090000"


async def test_actions_of_the_run_shown(ui):
    assert await ui.actions("baseline") == {"e2export": "Export"}
    assert await ui.actions("baseline", CURRENT) == {"e2export": "Export"}
    assert await ui.actions("baseline", FORMER) == {"e1oldexport": "Export (first run)"}
    assert await ui.actions("sweep") == {}


async def test_run_action_of_current_run(ui, provider):
    run = await ui.run_action("baseline", "e2export")
    assert (run.warning, run.severity) == (None, "info")
    assert provider.executed_actions == [("baseline", CURRENT, "e2export")]


async def test_run_action_of_former_run_is_confirmed(ui, provider):
    warning = PAST_RUN_WARNING.format(run_id=FORMER)

    declined = await ui.run_action("baseline", "e1oldexport", FORMER, confirm=False)
    assert (declined.warning, declined.severity) == (warning, None)
    assert provider.executed_actions == []

    run = await ui.run_action("baseline", "e1oldexport", FORMER)
    assert (run.warning, run.severity) == (warning, "info")
    assert provider.executed_actions == [("baseline", FORMER, "e1oldexport")]
