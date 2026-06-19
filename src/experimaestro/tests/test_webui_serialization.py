"""Tests for the WebUI serialization layer and the shared log reader."""

from experimaestro.scheduler.logs import read_log_slice
from experimaestro.webui.websocket import (
    _job_dict_to_frontend,
    serialize_warning,
)


def _sample_state_dict():
    return {
        "job_id": "abc123",
        "task_id": "pkg.MyTask",
        "path": "/wd/jobs/x",
        "state": "RUNNING",
        "scheduler_state": "RUNNING",
        "failure_reason": None,
        "started_time": "2026-01-01T00:00:00",
        "ended_time": None,
        "exit_code": None,
        "retry_count": 2,
        "progress": [{"level": 0, "progress": 0.5, "desc": "half"}],
        "process": {
            "pid": 123,
            "type": "local",
            "running": True,
            "cpu_percent": 12.5,
            "memory_mb": 256.0,
            "num_threads": 4,
        },
        "carbon_metrics": {
            "co2_kg": 0.1,
            "energy_kwh": 0.2,
            "cpu_power_w": 5,
            "gpu_power_w": 0,
            "ram_power_w": 1,
            "duration_s": 60,
            "region": "FR",
            "is_final": False,
        },
        "tags": [("lr", "0.1")],
        "depends_on": ["dep1"],
        "experiment_ids": ["exp1"],
        "submitted_time": "2026-01-01T00:00:00",
    }


def test_job_dict_to_frontend_maps_canonical_schema():
    out = _job_dict_to_frontend(_sample_state_dict())
    assert out["jobId"] == "abc123"
    assert out["taskId"] == "pkg.MyTask"
    assert out["locator"] == "/wd/jobs/x"
    assert out["status"] == "running"
    assert out["schedulerState"] == "running"
    assert out["start"] == "2026-01-01T00:00:00"
    assert out["retryCount"] == 2
    assert out["dependsOn"] == ["dep1"]
    assert out["experimentIds"] == ["exp1"]
    # carbon + process surfaced (camelCase)
    assert out["carbon"]["co2kg"] == 0.1
    assert out["carbon"]["region"] == "FR"
    assert out["process"]["pid"] == 123
    assert out["process"]["cpuPercent"] == 12.5


def test_job_dict_to_frontend_without_carbon_or_process():
    d = _sample_state_dict()
    d.pop("carbon_metrics")
    d.pop("process")
    out = _job_dict_to_frontend(d)
    assert out["carbon"] is None
    assert out["process"] is None


def test_serialize_warning():
    class W:
        warning_key = "k1"
        experiment_id = "exp1"
        run_id = "r1"
        description = "something happened"
        severity = "error"
        actions = {"retry": "Retry"}
        context = {"job": "j"}

    out = serialize_warning(W())
    assert out["warningKey"] == "k1"
    assert out["severity"] == "error"
    assert out["actions"] == {"retry": "Retry"}


def test_read_log_slice_tail_and_offset(tmp_path):
    log = tmp_path / "task.out"
    log.write_text("line1\nline2\n")

    first = read_log_slice(log)  # tail
    assert "line1" in first["content"]
    assert "line2" in first["content"]

    # No new content from the returned offset
    again = read_log_slice(log, first["offset"])
    assert again["content"] == ""

    # Append and read incrementally
    with open(log, "a") as f:
        f.write("line3\n")
    nxt = read_log_slice(log, first["offset"])
    assert "line3" in nxt["content"]


def test_read_log_slice_collapses_carriage_returns(tmp_path):
    log = tmp_path / "task.out"
    # tqdm-style progress rewriting the same line
    log.write_text("10%\r50%\r100%\ndone\n")
    out = read_log_slice(log)
    assert "100%" in out["content"]
    assert "10%" not in out["content"]
    assert "done" in out["content"]


def test_read_log_slice_missing_file(tmp_path):
    out = read_log_slice(tmp_path / "nope.out")
    assert out == {"content": "", "offset": 0, "size": 0}
