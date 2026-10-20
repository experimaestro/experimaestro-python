"""Experiment dependency graph (not in the TUI)"""

import pytest

from .scenario import EVAL_DONE, LEARN_DONE, LEARN_ERROR

pytestmark = [pytest.mark.anyio, pytest.mark.ui("web", "browser")]


async def test_graph(ui):
    assert await ui.graph("baseline") == {
        frozenset({LEARN_DONE}),
        frozenset({LEARN_ERROR}),
        frozenset({EVAL_DONE}),
    }
