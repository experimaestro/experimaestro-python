import json
from pathlib import Path

import pytest

from experimaestro import Task, Param
from experimaestro.tests.utils import TemporaryDirectory, TemporaryExperiment
from experimaestro.scheduler.experiment import cleanup_experiment_history
from experimaestro.settings import HistorySettings


class TaskA(Task):
    def execute(self):
        pass


class TaskB(Task):
    task_a: Param[TaskA]
    x: Param[int]

    def execute(self):
        pass


class FlagHandler:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


def test_experiment_events():
    """Test handlers"""

    flag = FlagHandler()
    with TemporaryExperiment("experiment"):
        task_a = TaskA.C()
        task_a.submit()
        task_a.on_completed(flag.set)

    assert flag.is_set()


# === Tests for cleanup_experiment_history ===


def _create_run_dir(experiment_base: Path, run_id: str, status: str) -> Path:
    """Helper to create a fake run directory with a given status."""
    run_dir = experiment_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env_data = {"run": {"status": status}}
    (run_dir / "environment.json").write_text(json.dumps(env_data))
    return run_dir


# fmt: off
# Cleanup uses `while count > max` logic, so max_done=N keeps N runs
CLEANUP_TEST_CASES = [
    # runs [(run_id, status, should_remain)], max_done, max_failed, current_run_id, current_status
    pytest.param(
        [
            ("20260101_000000", "completed", False),
            ("20260101_010000", "completed", False),
            ("20260101_020000", "completed", True),   # kept (3rd newest)
            ("20260101_030000", "completed", True),   # kept (2nd newest)
            ("20260101_040000", "completed", True),   # kept (newest)
        ],
        3, 1, None, None,
        id="oldest_deleted_first__max_done_3_keeps_3",
    ),
    pytest.param(
        [
            ("20260101_000000", "completed", False),  # removed (oldest completed)
            ("20260101_010000", "completed", True),   # kept (2nd newest completed)
            ("20260101_020000", "failed",    False),   # remove (failed after success)
            ("20260101_030000", "completed", True),   # kept (newest completed)
        ],
        2, 1, None, None,
        id="max_done_2_keeps_2_and_failed_1",
    ),
    pytest.param(
        [
            ("20260101_000000", "completed", False),
            ("20260101_010000", "completed", False),
            ("20260101_020000", "completed", True),   # kept (2nd newest)
            ("20260101_030000", "completed", True),   # kept (newest)
        ],
        2, 1, None, None,
        id="max_done_2_keeps_2",
    ),
    pytest.param(
        [
            ("20260101_000000", "completed", True),
            ("20260101_010000", "completed", True),
            ("20260101_020000", "completed", True),
            ("20260101_030000", "completed", True),
        ],
        5, 1, None, None,
        id="max_done_5_keeps_all_4",
    ),
    pytest.param(
        [
            ("20260101_000000", "failed",    False),  # removed (oldest failed)
            ("20260101_010000", "completed", True),   # kept (only completed)
            ("20260101_020000", "failed",    True),   # kept (2nd newest failed)
            ("20260101_030000", "failed",    True),   # kept (newest failed)
        ],
        5, 2, None, None,
        id="max_failed_2_keeps_2",
    ),
    pytest.param(
        [
            ("20260101_000000", "completed", True),   # current, preserved
            ("20260101_010000", "completed", True),   # kept (2 non-current <= max_done=2)
            ("20260101_020000", "completed", True),   # kept (newest)
        ],
        2, 1, "20260101_000000", None,
        id="excludes_current_run__all_kept",
    ),
    pytest.param(
        [
            ("20260101_000000", "failed",    False),  # removed (success clears failed)
            ("20260101_010000", "completed", True),
            ("20260101_020000", "failed",    False),  # removed (success clears failed)
            ("20260101_030000", "completed", True),
        ],
        5, 5, "current_run", "completed",
        id="success_removes_all_failed",
    ),
    pytest.param(
        [
            ("20260101_010000", "completed", False),  # removed (oldest completed)
            ("20260101_020000", "failed",    False),  # removed (before newest success)
            ("20260101_030000", "completed", False),  # removed (oldest completed)
            ("20260101_040000", "failed",    False),  # removed (before newest success)
            ("20260101_050000", "completed", True),   # kept (2nd newest completed)
            ("20260101_060000", "failed",    False),  # removed (before newest success)
            ("20260101_070000", "completed", True),   # kept (newest completed)
        ],
        2, 1, None, None,
        id="mixed_runs__max_done_2_max_failed_1",
    ),
    pytest.param(
        [
            ("20260101_120000",   "completed", False),
            ("20260101_120000.1", "completed", False),
            ("20260101_120000.2", "completed", True),  # kept (2nd newest)
            ("20260101_130000",   "completed", True),  # kept (newest)
        ],
        2, 1, None, None,
        id="handles_modifiers_in_order",
    ),
]
# fmt: on


@pytest.mark.parametrize(
    "runs,max_done,max_failed,current_run_id,current_status",
    CLEANUP_TEST_CASES,
)
def test_cleanup_experiment_history(
    runs: list[tuple[str, str, bool]],
    max_done: int,
    max_failed: int,
    current_run_id: str | None,
    current_status: str | None,
):
    """Test cleanup_experiment_history with various configurations.

    Args:
        runs: List of (run_id, status, should_remain) tuples
        max_done: HistorySettings.max_done (removes while count >= max_done)
        max_failed: HistorySettings.max_failed (removes while count >= max_failed)
        current_run_id: Run to exclude from cleanup
        current_status: If "completed", removes ALL past failed runs
    """
    with TemporaryDirectory() as workdir:
        experiment_base = workdir / "experiments" / "test-exp"
        experiment_base.mkdir(parents=True)

        # Create all run directories
        for run_id, status, _ in runs:
            _create_run_dir(experiment_base, run_id, status)

        # Run cleanup
        history = HistorySettings(max_done=max_done, max_failed=max_failed)
        removed = cleanup_experiment_history(
            experiment_base,
            current_run_id=current_run_id,
            current_status=current_status,
            history=history,
        )

        # Verify results
        remaining = {d.name for d in experiment_base.iterdir()}
        expected_remaining = {run_id for run_id, _, keep in runs if keep}
        expected_removed = {run_id for run_id, _, keep in runs if not keep}

        assert remaining == expected_remaining, (
            f"Remaining mismatch: got {remaining}, expected {expected_remaining}"
        )
        assert len(removed) == len(expected_removed), (
            f"Removed count: got {len(removed)}, expected {len(expected_removed)}"
        )
