"""Job logs: progress bars and the order of the streams"""

import pytest

from .scenario import LEARN_DONE, LEARN_RUNNING

pytestmark = pytest.mark.anyio


async def test_progress_folded_while_logging(ui):
    """A progress bar (tqdm) rewriting its line with ``\\r`` shows its last
    state only, also when it is written while the log is displayed"""
    await ui.open_experiment("sweep")
    lines = await ui.follow_log(
        LEARN_RUNNING,
        "stderr",
        [
            "\r 10%|█         |",
            "\r 20%|██        |\r 30%|███       |",
            "\r100%\ndone\n",
        ],
    )
    assert lines == [f"stderr of {LEARN_RUNNING}", "100%", "done"]


@pytest.mark.ui("tui", "browser")
async def test_stderr_first(ui):
    """stderr (where errors and progress bars go) is the first log shown"""
    await ui.open_experiment("baseline")
    assert await ui.log_streams(LEARN_DONE) == (["stderr", "stdout"], "stderr")


async def test_nested_progress_folded(ui):
    """A nested progress bar (tqdm ``position=1``) goes back to its line with
    ``ESC[A`` after each update: it shows its last state only, as in a
    terminal"""
    await ui.open_experiment("sweep")
    lines = await ui.follow_log(
        LEARN_RUNNING,
        "stderr",
        [
            "\nEpoch 1:  10%|█         |\x1b[A",
            "\nEpoch 1:  20%|██        |\x1b[A",
            "\nEpoch 1: 100%|██████████|\x1b[A\n\ndone\n",
        ],
    )
    assert lines == [
        f"stderr of {LEARN_RUNNING}",
        "",
        "Epoch 1: 100%|██████████|",
        "done",
    ]
