"""Where notifications are shown"""

import pytest

from .scenario import LEARN_DONE

pytestmark = [pytest.mark.anyio, pytest.mark.ui("browser")]


@pytest.mark.ui_gap("browser", reason="notifications are shown in the middle")
async def test_notifications_top_right(ui):
    """Notifications are shown at the top right, out of the way of the lists"""
    await ui.open_experiment("baseline")
    # Copying notifies
    await ui.copy_job_id(LEARN_DONE)
    toast = ui.page.locator(".toast").last
    await toast.wait_for()
    box = await toast.bounding_box()
    viewport = ui.page.viewport_size
    assert box["y"] < viewport["height"] / 4
    assert box["x"] + box["width"] > viewport["width"] * 3 / 4
