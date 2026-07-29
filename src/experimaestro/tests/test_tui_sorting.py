"""Tests for TUI sorting keybindings and logic."""

from dataclasses import dataclass
from typing import Optional
from experimaestro.tui.widgets.experiments import ExperimentsList
from experimaestro.tui.widgets.jobs import JobsTable


@dataclass
class DummyExperiment:
    experiment_id: str
    started_at: Optional[object] = None
    failed_jobs: int = 0
    total_jobs: int = 0
    finished_jobs: int = 0


@dataclass
class DummyJob:
    identifier: str
    task_id: str = "task_a"


def test_experiments_alphabetical_sorting():
    exps = [
        DummyExperiment("zebra_exp"),
        DummyExperiment("Alpha_exp"),
        DummyExperiment("beta_exp"),
    ]

    # Standard sorting by ID ascending (case-insensitive)
    sorted_asc = sorted(exps, key=lambda e: (e.experiment_id or "").lower())
    assert [e.experiment_id for e in sorted_asc] == [
        "Alpha_exp",
        "beta_exp",
        "zebra_exp",
    ]

    # Sorting by ID descending
    sorted_desc = sorted(
        exps, key=lambda e: (e.experiment_id or "").lower(), reverse=True
    )
    assert [e.experiment_id for e in sorted_desc] == [
        "zebra_exp",
        "beta_exp",
        "Alpha_exp",
    ]


def test_jobs_alphabetical_sorting():
    jobs = [
        DummyJob("job_z99"),
        DummyJob("job_a01"),
        DummyJob("job_M50"),
    ]

    sorted_asc = sorted(jobs, key=lambda j: (j.identifier or "").lower())
    assert [j.identifier for j in sorted_asc] == ["job_a01", "job_M50", "job_z99"]

    sorted_desc = sorted(
        jobs, key=lambda j: (j.identifier or "").lower(), reverse=True
    )
    assert [j.identifier for j in sorted_desc] == ["job_z99", "job_M50", "job_a01"]


def test_sort_column_mappings_and_bindings():
    assert ExperimentsList.SORTABLE_COLUMNS.get("id") == "id"
    assert JobsTable.SORTABLE_COLUMNS.get("job_id") == "job_id"

    exp_binding_keys = [b.key for b in ExperimentsList.BINDINGS]
    assert "O" in exp_binding_keys
    assert "S" in exp_binding_keys

    job_binding_keys = [b.key for b in JobsTable.BINDINGS]
    assert "O" in job_binding_keys
    assert "S" in job_binding_keys
