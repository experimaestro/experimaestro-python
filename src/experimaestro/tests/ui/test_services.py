"""Services of an experiment

Services are read through the state provider by both interfaces, so they are
listed and can be started without a running experiment. The same service id is
used by two experiments in the scenario.
"""

import pytest

from .drivers import refused

pytestmark = pytest.mark.anyio


async def test_services_are_listed(ui):
    assert await ui.services("sweep") == {"tensorboard": "stopped"}
    assert await ui.services("baseline") == {"tensorboard": "stopped"}


async def test_start_and_stop_service(ui):
    started = await ui.start_service("sweep", "tensorboard")
    assert started.severity == "info"
    assert started.message.startswith("Service started: ")
    assert await ui.services("sweep") == {"tensorboard": "running"}
    # The service with the same id in another experiment is another service
    assert await ui.services("baseline") == {"tensorboard": "stopped"}

    assert refused(
        await ui.start_service("sweep", "tensorboard"),
        "Service 'tensorboard' is already running",
    )

    stopped = await ui.stop_service("sweep", "tensorboard")
    assert (stopped.severity, stopped.message) == (
        "info",
        "Service stopped: tensorboard",
    )
    assert await ui.services("sweep") == {"tensorboard": "stopped"}


async def test_stop_refuses_stopped_service(ui):
    assert refused(
        await ui.stop_service("sweep", "tensorboard"),
        "Service 'tensorboard' is not running",
    )


@pytest.mark.ui("browser")
async def test_active_services_shown_everywhere(ui):
    """Running services are shown with their experiment, whatever the page"""
    assert await ui.active_services() == {}
    await ui.start_service("sweep", "tensorboard")
    await ui.open_experiment("baseline")
    assert await ui.active_services() == {"sweep/tensorboard": "running"}

    await ui.stop_service("sweep", "tensorboard")
    assert await ui.active_services() == {}
