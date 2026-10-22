"""Copying job identifiers and paths from the job list"""

import pytest

from .scenario import LEARN_DONE

pytestmark = [pytest.mark.anyio, pytest.mark.ui("browser")]


async def test_copy_job_id(ui):
    """A click on the job hash copies ``task id/job id``"""
    await ui.open_experiment("baseline")
    assert await ui.copy_job_id(LEARN_DONE) == f"pkg.Learn/{LEARN_DONE}"


async def test_copy_job_path(ui, provider):
    """A shift-click on the job hash copies the job folder"""
    await ui.open_experiment("baseline")
    path = str(provider.jobs[LEARN_DONE].path)
    assert await ui.copy_job_id(LEARN_DONE, shift=True) == path


async def test_copy_job_id_help(ui):
    """Hovering the job hash tells what a click and a shift-click copy"""
    await ui.open_experiment("baseline")
    title = await ui._job_row(LEARN_DONE).locator(".job-id").get_attribute("title")
    assert "shift" in (title or "").lower()
