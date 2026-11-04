"""Copying job identifiers and paths from the job list"""

import pytest

from .scenario import LEARN_DONE

pytestmark = [pytest.mark.anyio, pytest.mark.ui("browser")]


async def test_copy_job_id(ui):
    """The copy icon of a job copies ``task id/job id``"""
    await ui.open_experiment("baseline")
    assert await ui.copy_job_id(LEARN_DONE) == f"pkg.Learn/{LEARN_DONE}"


async def test_copy_job_path(ui, provider):
    """The folder icon of a job copies the job folder"""
    await ui.open_experiment("baseline")
    path = str(provider.jobs[LEARN_DONE].path)
    assert await ui.copy_job_id(LEARN_DONE, shift=True) == path


async def test_copy_job_id_help(ui):
    """Hovering the copy icons tells their keyboard shortcut"""
    await ui.open_experiment("baseline")
    for action, key in (("copy-id", "c"), ("copy-folder", "f")):
        icon = await ui._job_action_icon(LEARN_DONE, action)
        assert f"({key})" in (await icon.get_attribute("title") or "")
