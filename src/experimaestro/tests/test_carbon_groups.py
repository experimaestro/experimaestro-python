"""Carbon totals of groups of experiments (experiment tags) and per task"""

import json
import subprocess
import sys

import pytest

from experimaestro import experiment
from experimaestro.carbon.storage import (
    CarbonRecord,
    CarbonStorage,
    aggregate_experiment_groups,
)
from experimaestro.scheduler.state_status import CarbonMetricsEvent

from .ui.scenario import LEARN_DONE, ScenarioProvider

#: Shared by the two experiments of the group
SHARED = "jobshared"
TRAIN_A = "jobtraina"
EVAL_B = "jobevalb"
OTHER = "jobother"


def _record(job_id: str, co2_kg: float, task_id: str, ended_at: str = "01:00:00"):
    return CarbonRecord(
        job_id=job_id,
        task_id=task_id,
        started_at="2026-01-01T00:00:00",
        ended_at=f"2026-01-01T{ended_at}",
        co2_kg=co2_kg,
        energy_kwh=co2_kg * 10,
        cpu_power_w=0.0,
        gpu_power_w=0.0,
        ram_power_w=0.0,
        duration_s=3600.0,
        region="FRA",
    )


@pytest.fixture
def workspace(tmp_path):
    """Three experiments: two in the group (sharing a job), one outside"""
    workspace = tmp_path / "workspace"
    jobs = {
        ("xp-a", ("group-x",)): [SHARED, TRAIN_A],
        ("xp-b", ("group-x", "other-tag")): [SHARED, EVAL_B],
        ("xp-c", ()): [OTHER],
    }
    for (name, tags), job_ids in jobs.items():
        with experiment(workspace, name, tags=list(tags), no_environmental_impact=True):
            pass
        run_dir = workspace / "experiments" / name / "current"
        with (run_dir / "jobs.jsonl").open("a") as out:
            for job_id in job_ids:
                out.write(json.dumps({"job_id": job_id, "task_id": "pkg.task"}) + "\n")

    storage = CarbonStorage(workspace)
    # The shared job failed and was retried
    storage.write_record(_record(SHARED, 1.0, "pkg.Train", "01:00:00"))
    storage.write_record(_record(SHARED, 2.0, "pkg.Train", "02:00:00"))
    storage.write_record(_record(TRAIN_A, 4.0, "pkg.Train"))
    storage.write_record(_record(EVAL_B, 8.0, "pkg.Eval"))
    storage.write_record(_record(OTHER, 16.0, "pkg.Train"))
    return workspace


def _xpm(workspace, *args: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", "experimaestro", "--workdir", str(workspace), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _summary(workspace, *args: str) -> dict:
    return json.loads(
        _xpm(workspace, "experiments", "carbon", "summary", "--json", *args)
    )


def test_group_counts_shared_jobs_once(workspace):
    totals = aggregate_experiment_groups(
        workspace, {"both": ["xp-a", "xp-b"], "one": ["xp-b"], "none": []}
    )
    assert totals["both"].co2_kg == 15.0
    assert totals["both"].energy_kwh == 150.0
    assert totals["both"].job_count == 3
    assert totals["one"].co2_kg == 11.0
    assert totals["none"].co2_kg == 0.0


def test_summary_group(workspace):
    result = _summary(workspace, "--group", "group-x", "-v")
    assert result["totals"]["co2_kg"] == 15.0
    assert result["unique_jobs"] == 3
    # Each experiment counts the shared job: the details add up to more
    assert result["experiments"]["xp-a"]["co2_kg"] == 7.0
    assert result["experiments"]["xp-b"]["co2_kg"] == 11.0
    assert "xp-c" not in result["experiments"]


def test_summary_several_groups(workspace):
    result = _summary(workspace, "-g", "other-tag", "-g", "unknown")
    assert result["totals"]["co2_kg"] == 11.0


def test_summary_unknown_group(workspace):
    output = _xpm(workspace, "experiments", "carbon", "summary", "-g", "unknown")
    assert "No jobs found" in output


def test_summary_by_task(workspace):
    result = _summary(workspace, "--group", "group-x", "--by-task")
    assert result["tasks"]["pkg.Train"]["co2_kg"] == 7.0
    assert result["tasks"]["pkg.Train"]["job_count"] == 2
    assert result["tasks"]["pkg.Eval"]["co2_kg"] == 8.0

    result = _summary(workspace, "--group", "group-x", "--by-task", "--latest-only")
    assert result["tasks"]["pkg.Train"]["co2_kg"] == 6.0
    assert result["totals"]["co2_kg"] == 14.0


def test_summary_by_task_text(workspace):
    output = _xpm(
        workspace, "experiments", "carbon", "summary", "-g", "group-x", "--by-task"
    )
    assert "Per-Task Breakdown" in output
    # Largest first
    assert output.index("pkg.Eval") < output.index("pkg.Train")


def test_list_group(workspace):
    output = _xpm(workspace, "experiments", "list", "--group", "group-x")
    assert "xp-a #group-x" in output
    assert "xp-b #group-x #other-tag" in output
    assert "xp-c" not in output


def test_provider_totals_follow_final_carbon_events(tmp_path):
    provider = ScenarioProvider(tmp_path)
    groups = {"group": ["baseline"]}
    assert provider.get_carbon_totals(groups)["group"].co2_kg == 3.5

    CarbonStorage(tmp_path).write_record(_record(LEARN_DONE, 1.0, "pkg.Learn"))

    def event(is_final: bool) -> CarbonMetricsEvent:
        return CarbonMetricsEvent(job_id=LEARN_DONE, co2_kg=1.0, is_final=is_final)

    # Cached: periodic events only update the job
    provider.apply_event(event(False))
    assert provider.get_carbon_totals(groups)["group"].co2_kg == 3.5

    # The final event comes with a new record
    provider.apply_event(event(True))
    assert provider.get_carbon_totals(groups)["group"].co2_kg == 4.5


def test_scheduler_totals(tmp_path):
    """A running experiment answers from the records of its workspace"""
    workspace = tmp_path / "workspace"
    with experiment(workspace, "live", no_environmental_impact=True) as xp:
        with (xp.workdir / "jobs.jsonl").open("a") as out:
            out.write(json.dumps({"job_id": TRAIN_A, "task_id": "pkg.task"}) + "\n")
        CarbonStorage(workspace).write_record(_record(TRAIN_A, 2.0, "pkg.Train"))
        totals = xp.scheduler.get_carbon_totals({"group": ["live", "unknown"]})
    assert totals["group"].co2_kg == 2.0
    assert totals["group"].job_count == 1
