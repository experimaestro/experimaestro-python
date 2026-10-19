"""Services of an experiment"""

import pytest

pytestmark = pytest.mark.anyio


@pytest.mark.ui_gap(
    "web", reason="the web UI lists services only when a scheduler is running"
)
async def test_services_are_listed(ui):
    assert await ui.services("sweep") == {"tensorboard": "stopped"}
