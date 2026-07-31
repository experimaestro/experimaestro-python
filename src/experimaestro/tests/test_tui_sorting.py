"""Tests for TUI sorting keybindings and logic"""

from dataclasses import dataclass
from datetime import datetime

from experimaestro.tui.widgets.experiments import ExperimentsList
from experimaestro.tui.widgets.jobs import JobsTable


@dataclass
class DummyExperiment:
    experiment_id: str
    started_at: datetime | None = None


@dataclass
class DummyJob:
    identifier: str
    task_id: str = "task_a"


def make_experiments_list(sort_column: str | None, reverse: bool) -> ExperimentsList:
    """Build an ExperimentsList without mounting it in a Textual app"""
    widget = ExperimentsList.__new__(ExperimentsList)
    widget._sort_column = sort_column
    widget._sort_reverse = reverse
    return widget


def make_jobs_table(sort_column: str | None, reverse: bool) -> JobsTable:
    """Build a JobsTable without mounting it in a Textual app"""
    widget = JobsTable.__new__(JobsTable)
    widget._sort_column = sort_column
    widget._sort_reverse = reverse
    widget._needs_rebuild = False
    widget.experiment_job_info = {}
    return widget


def stub_ui(widget, refreshes: list) -> None:
    """Replace the Textual-dependent parts of a sort action"""
    widget._update_column_headers = lambda: None
    widget.notify = lambda *args, **kwargs: None
    widget.refresh_experiments = lambda: refreshes.append("refresh")
    widget.refresh_jobs = lambda: refreshes.append("refresh")


def test_experiments_sort_alphabetical():
    experiments = [
        DummyExperiment("zebra_exp"),
        DummyExperiment("Alpha_exp"),
        DummyExperiment("beta_exp"),
    ]

    widget = make_experiments_list("id", False)
    assert [e.experiment_id for e in widget._sort_experiments(experiments)] == [
        "Alpha_exp",
        "beta_exp",
        "zebra_exp",
    ]

    widget._sort_reverse = True
    assert [e.experiment_id for e in widget._sort_experiments(experiments)] == [
        "zebra_exp",
        "beta_exp",
        "Alpha_exp",
    ]

    # The provider order is kept when no sort column is selected
    widget = make_experiments_list(None, False)
    assert widget._sort_experiments(experiments) == experiments


def test_jobs_sort_alphabetical():
    jobs = [DummyJob("job_z99"), DummyJob("job_a01"), DummyJob("job_M50")]

    widget = make_jobs_table("job_id", False)
    widget._sort_jobs(jobs)
    assert [j.identifier for j in jobs] == ["job_a01", "job_M50", "job_z99"]

    widget._sort_reverse = True
    widget._sort_jobs(jobs)
    assert [j.identifier for j in jobs] == ["job_z99", "job_M50", "job_a01"]


def test_experiments_sort_action_toggles():
    refreshes: list = []
    widget = make_experiments_list(None, False)
    stub_ui(widget, refreshes)

    widget.action_sort_alphabetical()
    assert (widget._sort_column, widget._sort_reverse) == ("id", False)

    # Pressing again toggles the direction
    widget.action_sort_alphabetical()
    assert (widget._sort_column, widget._sort_reverse) == ("id", True)

    # Switching to another column restarts from ascending
    widget.action_sort_by_status()
    assert (widget._sort_column, widget._sort_reverse) == ("status", False)

    assert len(refreshes) == 3


def test_jobs_sort_action_toggles():
    refreshes: list = []
    widget = make_jobs_table(None, False)
    stub_ui(widget, refreshes)

    widget.action_sort_alphabetical()
    assert (widget._sort_column, widget._sort_reverse) == ("job_id", False)
    assert widget._needs_rebuild

    widget.action_sort_alphabetical()
    assert (widget._sort_column, widget._sort_reverse) == ("job_id", True)

    widget.action_sort_by_task()
    assert (widget._sort_column, widget._sort_reverse) == ("task", False)

    assert len(refreshes) == 3


def test_sort_column_mappings_and_bindings():
    # Sortable columns must match actual table columns
    assert ExperimentsList.SORTABLE_COLUMNS["id"] == "id"
    assert JobsTable.SORTABLE_COLUMNS["job_id"] == "job_id"
    assert set(ExperimentsList.SORTABLE_COLUMNS) <= set(ExperimentsList.COLUMN_LABELS)
    assert set(JobsTable.SORTABLE_COLUMNS) <= set(JobsTable.COLUMN_LABELS)

    for widget_class in (ExperimentsList, JobsTable):
        bindings = {b.key: b.action for b in widget_class.BINDINGS}
        assert bindings["O"] == "sort_alphabetical"
