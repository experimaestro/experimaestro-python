"""Tests for parameter inspection integration in the TUI monitor app."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label, Static

from experimaestro.scheduler.interfaces import BaseJob, JobState
from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tools.inspect_params import ParamsInspectorScreen, ParamsTree
from experimaestro.tui.messages import InspectJobParams, InspectJobParamsRequest
from experimaestro.tui.widgets.jobs import (
    JobDetailView,
    JobsTable,
    JobsTreeView,
    TreeJobData,
)


class DummyJob(BaseJob):
    """Minimal dummy job for state provider and widget testing"""

    def __init__(self, identifier: str, task_id: str, path: Path):
        self.identifier = identifier
        self.task_id = task_id
        self.path = path
        self._state = JobState.RUNNING
        self.starttime = None
        self.endtime = None
        self.progress = []
        self._process = None
        self.carbon_metrics = None


class DummyLocalStateProvider(StateProvider):
    """Minimal local state provider for testing"""

    @property
    def is_remote(self) -> bool:
        return False

    def get_display_path(self, job):
        return str(job.path)

    def get_process_info(self, job):
        return None

    def get_experiments(self, *args, **kwargs):
        return []

    def get_experiment(self, *args, **kwargs):
        return None

    def get_experiment_runs(self, *args, **kwargs):
        return []

    def get_jobs(self, *args, **kwargs):
        return []

    def get_job(self, task_id: str, job_id: str, *args, **kwargs):
        return None

    def get_all_jobs(self, *args, **kwargs):
        return []

    def get_services(self, *args, **kwargs):
        return []

    def get_tags_map(self, *args, **kwargs):
        return {}

    def get_dependencies_map(self, *args, **kwargs):
        return {}

    def get_current_run(self, *args, **kwargs):
        return None

    def get_experiment_job_info(self, *args, **kwargs):
        return {}

    def kill_job(self, *args, **kwargs):
        return True

    def clean_job(self, *args, **kwargs):
        return True

    def close(self):
        pass


class DummyRemoteStateProvider(DummyLocalStateProvider):
    """Minimal remote state provider for testing"""

    @property
    def is_remote(self) -> bool:
        return True


def test_state_provider_job_params_local(tmp_path: Path):
    """Verify local StateProvider parameter detection and retrieval."""
    sp = DummyLocalStateProvider()
    job_dir = tmp_path / "job_a"
    job_dir.mkdir()
    job = DummyJob("job_a", "task_pkg.task_a", job_dir)

    # Neither exists
    assert sp.has_job_params(job) is False
    assert sp.get_job_params_file(job) is None

    # params.json exists
    params_file = job_dir / "params.json"
    params_file.write_text('{"version": 1}')
    assert sp.has_job_params(job) is True
    assert sp.get_job_params_file(job) == params_file

    # experimaestro.json fallback
    params_file.unlink()
    xpm_file = job_dir / "experimaestro.json"
    xpm_file.write_text('[{"version": 1}]')
    assert sp.has_job_params(job) is True
    assert sp.get_job_params_file(job) == xpm_file


def test_state_provider_job_params_remote(tmp_path: Path):
    """Verify remote StateProvider synchronizes params via sync_path."""
    sp = DummyRemoteStateProvider()
    remote_job_dir = Path("/remote/workspace/jobs/task_a/12345678")
    job = DummyJob("12345678", "task_a", remote_job_dir)

    # Remote job always has_job_params = True if path is set
    assert sp.has_job_params(job) is True

    # Mock sync_path
    cache_dir = tmp_path / "cache_job"
    cache_dir.mkdir()
    cached_params = cache_dir / "params.json"
    cached_params.write_text('{"version": 1}')

    sp.sync_path = MagicMock(return_value=cache_dir)

    result_path = sp.get_job_params_file(job)
    sp.sync_path.assert_called_once_with(
        str(remote_job_dir), include=["params.json", "experimaestro.json"]
    )
    assert result_path == cached_params


def test_job_detail_view_hints_and_actions(tmp_path: Path):
    """Verify JobDetailView updates the hint and posts InspectJobParams."""
    sp = DummyLocalStateProvider()
    view = JobDetailView(sp)

    labels: dict[str, str] = {}

    def mock_query_one(selector, expected_type=None):
        if not isinstance(selector, str):
            # A widget queried by type (e.g. the dependency list)
            return MagicMock(spec=selector)
        mock_label = MagicMock(spec=Label)
        mock_label.update = lambda text: labels.__setitem__(selector, str(text))
        return mock_label

    view.query_one = mock_query_one

    # Setup dummy job with both logs and params
    job_dir = tmp_path / "job_both"
    job_dir.mkdir()
    (job_dir / "params.json").write_text("{}")
    (job_dir / "my_task.out").write_text("log")
    job = DummyJob("job_both", "pkg.my_task", job_dir)
    view.job_data = job

    view._update_job_display(job)
    assert "Press 'l' to view logs, 'i' to inspect params" in labels["#job-logs-hint"]

    # Action posts InspectJobParams
    messages = []
    view.post_message = lambda msg: messages.append(msg)
    view.action_inspect_params()
    assert len(messages) == 1
    assert isinstance(messages[0], InspectJobParams)
    assert messages[0].job is job

    # Setup dummy job with only params
    (job_dir / "my_task.out").unlink()
    view._update_job_display(job)
    assert "Press 'i' to inspect params" in labels["#job-logs-hint"]
    assert "no logs yet" in labels["#job-logs-hint"]

    # Setup dummy job with only logs
    (job_dir / "params.json").unlink()
    (job_dir / "my_task.out").write_text("log")
    view._update_job_display(job)
    assert labels["#job-logs-hint"] == "[bold cyan]Press 'l' to view logs[/bold cyan]"

    # Setup dummy job with neither
    (job_dir / "my_task.out").unlink()
    view._update_job_display(job)
    assert labels["#job-logs-hint"] == "(no log or param files found)"


def test_jobs_table_inspect_action():
    """Verify JobsTable action_inspect_params posts InspectJobParamsRequest."""
    table = JobsTable.__new__(JobsTable)
    table.current_experiment = "mnist_train"
    table._tree_mode = False
    table._get_selected_job_info = lambda: ("job_123", "pkg.my_task")

    messages = []
    table.post_message = lambda msg: messages.append(msg)
    table.action_inspect_params()

    assert len(messages) == 1
    assert isinstance(messages[0], InspectJobParamsRequest)
    assert messages[0].job_id == "job_123"
    assert messages[0].task_id == "pkg.my_task"
    assert messages[0].experiment_id == "mnist_train"


def test_jobs_tree_inspect_action():
    """Verify JobsTreeView action_inspect_params posts InspectJobParamsRequest."""
    tree_view = JobsTreeView.__new__(JobsTreeView)
    tree_view.current_experiment = "mnist_train"
    tree_view._get_selected_job_data = lambda: TreeJobData(
        job_id="job_456", task_id="pkg.eval_task", status_text="RUNNING", tags={}
    )

    messages = []
    tree_view.post_message = lambda msg: messages.append(msg)
    tree_view.action_inspect_params()

    assert len(messages) == 1
    assert isinstance(messages[0], InspectJobParamsRequest)
    assert messages[0].job_id == "job_456"
    assert messages[0].task_id == "pkg.eval_task"
    assert messages[0].experiment_id == "mnist_train"


@pytest.mark.anyio
async def test_params_inspector_screen_pushed_and_dismissed(tmp_path: Path):
    """Verify ParamsInspectorScreen can be pushed onto an App and dismissed with escape."""
    params_file = tmp_path / "params.json"
    params_file.write_text(
        '{"version": 1, "task": "test_task", "params": {"lr": 0.01}}'
    )

    class ParentApp(App):
        def compose(self) -> ComposeResult:
            yield Static("Parent Screen", id="parent-text")

    app = ParentApp()
    async with app.run_test() as pilot:
        assert len(app.screen_stack) == 1
        assert app.query_one("#parent-text", Static) is not None

        # Push the screen as ExperimaestroUI would
        screen = ParamsInspectorScreen(params_file, title="Test Job")
        app.push_screen(screen)
        await pilot.pause()

        assert len(app.screen_stack) == 2
        # Tree is active on the pushed screen
        tree = app.screen.query_one(ParamsTree)
        assert tree is not None

        # Press escape to dismiss back to parent screen
        await pilot.press("escape")
        await pilot.pause()

        assert len(app.screen_stack) == 1
        assert app.query_one("#parent-text", Static) is not None
