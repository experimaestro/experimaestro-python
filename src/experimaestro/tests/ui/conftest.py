"""Fixtures running each UI feature test on every interface

Tests take the ``ui`` fixture (a :class:`~.drivers.UIDriver`) and the
``provider`` it reads (a :class:`~.scenario.ScenarioProvider`). They run once
per driver: ``tui``, ``web`` (WebSocket protocol) and ``browser`` (the web UI
in Chromium, enabled by ``XPM_TEST_BROWSER=1``). Test modules are marked
``pytest.mark.anyio`` (Textual and the drivers are asynchronous).

Markers name interfaces (``tui``, ``web`` — which covers both web drivers) or
drivers (``browser`` alone):

- ``@pytest.mark.ui("tui")``: the feature only exists in these interfaces
- ``@pytest.mark.ui_gap("web", reason=...)``: known defect of an interface; a
  strict xfail, which fails once the gap is closed so that the marker is removed
"""

import os

import pytest

from .scenario import ScenarioProvider
from .tui_driver import TuiDriver
from .web_driver import WebDriver

DRIVERS = {"tui": TuiDriver, "web": WebDriver}

#: Interface tested by each driver
INTERFACES = {"tui": "tui", "web": "web", "browser": "web"}

#: Set to run the tests in a real browser too (see ``browser_driver.py``)
BROWSER = bool(os.environ.get("XPM_TEST_BROWSER"))


def _browser_driver():
    if not BROWSER:
        pytest.skip("browser tests are enabled by XPM_TEST_BROWSER=1")
    from .browser_driver import BrowserDriver

    return BrowserDriver


@pytest.fixture
def anyio_backend():
    # Textual runs on asyncio
    return "asyncio"


@pytest.fixture
def provider(tmp_path):
    return ScenarioProvider(tmp_path / "workspace")


@pytest.fixture(params=[*sorted(DRIVERS), "browser"])
async def ui(request, provider):
    names = {request.param, INTERFACES[request.param]}
    marker = request.node.get_closest_marker("ui")
    if marker is not None and not names & set(marker.args):
        pytest.skip(f"not available through the {request.param} driver")
    for gap in request.node.iter_markers("ui_gap"):
        if names & set(gap.args):
            request.node.add_marker(
                pytest.mark.xfail(reason=gap.kwargs["reason"], strict=True)
            )
    driver_class = (
        _browser_driver() if request.param == "browser" else DRIVERS[request.param]
    )
    async with driver_class(provider) as driver:
        yield driver
