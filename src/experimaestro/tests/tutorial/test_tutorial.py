"""Runs the walkthrough example (docs/source/tutorial.md) as the page does"""

import subprocess
import sys
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).parent

#: Output of results.py, shown in the walkthrough
EXPECTED_RESULTS = """\
{'lr': 0.001} mse=0.0098
{'lr': 0.01} mse=0.0088
{'lr': 0.1} mse=0.0092
"""


def experimaestro(
    *args: str, config_dir: Path, check: bool = True
) -> subprocess.CompletedProcess:
    """Runs the experimaestro CLI (with an empty configuration directory: no
    settings.yaml nor launchers.py, as for a new user)"""
    return subprocess.run(
        [sys.executable, "-m", "experimaestro", "run"]
        + ["--xpm-config-dir", str(config_dir), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
        timeout=120,
    )


@pytest.mark.timeout(300)
def test_tutorial(tmp_path: Path):
    workdir = tmp_path / "workspace"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    params = str(TUTORIAL / "params.yaml")

    # Dry run: lists the tasks, runs nothing
    run = experimaestro(
        "--workdir",
        str(workdir),
        "--run-mode",
        "dry_run",
        params,
        config_dir=config_dir,
    )
    assert run.stdout.count("[not run]") == 6
    assert not (workdir / "jobs").exists()

    # Run: three Learn tasks and their evaluations
    experimaestro("--workdir", str(workdir), params, config_dir=config_dir)
    for task_id in ("tasks.learn", "tasks.evaluate"):
        jobs = list((workdir / "jobs" / task_id).iterdir())
        assert len(jobs) == 3
        assert all((job / f"{task_id.split('.')[-1]}.done").is_file() for job in jobs)

    # Second run: nothing to do (no job process is started)
    run = experimaestro("--workdir", str(workdir), params, config_dir=config_dir)
    assert "Process started" not in run.stdout

    # Results, as shown in the walkthrough
    results = subprocess.run(
        [sys.executable, str(TUTORIAL / "results.py"), str(workdir)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert results.stdout == EXPECTED_RESULTS

    # A learning rate too large: Learn fails, and its evaluation is not run
    diverging = tmp_path / "diverging.yaml"
    diverging.write_text("learning_rates: [0.1, 10]\n")
    args = ("--workdir", str(workdir), "--post-yaml", str(diverging), params)
    run = experimaestro(*args, config_dir=config_dir, check=False)
    assert run.returncode == 1, run.stdout
    # Reported once, the evaluation being not run (rather than failed)
    summary = (
        "Experiment failed: 1 failed job"
        " (and 1 job not run because of a failed dependency)"
    )
    assert run.stdout.count(summary) == 1, run.stdout
    assert "evaluate.err" not in run.stdout
    failed = [
        job
        for job in (workdir / "jobs" / "tasks.learn").iterdir()
        if (job / "learn.failed").is_file()
    ]
    assert len(failed) == 1
    assert "Training diverged" in (failed[0] / "learn.err").read_text()
    assert len(list((workdir / "jobs" / "tasks.evaluate").glob("*/*.done"))) == 3

    # Running again only runs the failed job again
    run = experimaestro(*args, config_dir=config_dir, check=False)
    assert run.returncode == 1
    assert run.stdout.count("Process started") == 1
