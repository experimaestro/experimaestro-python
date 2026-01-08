"""Tests for pre_experiment feature in run-experiment CLI"""

import pytest
from pathlib import Path
from click.testing import CliRunner

from experimaestro.cli import cli


# --- Test fixture files as separate modules ---

# Pre-experiment scripts
PRE_SETUP_ENV = (
    Path(__file__).parent / "fixtures" / "pre_experiment" / "pre_setup_env.py"
)
PRE_SETUP_MOCK = (
    Path(__file__).parent / "fixtures" / "pre_experiment" / "pre_setup_mock.py"
)
PRE_SETUP_ERROR = (
    Path(__file__).parent / "fixtures" / "pre_experiment" / "pre_setup_error.py"
)

# Experiment files
EXP_CHECK_ENV = (
    Path(__file__).parent / "fixtures" / "pre_experiment" / "experiment_check_env.py"
)
EXP_CHECK_MOCK = (
    Path(__file__).parent / "fixtures" / "pre_experiment" / "experiment_check_mock.py"
)
EXP_SIMPLE = (
    Path(__file__).parent / "fixtures" / "pre_experiment" / "experiment_simple.py"
)


@pytest.fixture
def experiment_dir(tmp_path):
    """Create a directory with experiment files"""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    return exp_dir, workdir


def _create_yaml(exp_dir, experiment_id, file_path, pre_experiment_path=None):
    """Helper to create a YAML config file"""
    yaml_file = exp_dir / "config.yaml"
    content = f"id: {experiment_id}\nfile: {file_path}\n"
    if pre_experiment_path:
        content += f"pre_experiment: {pre_experiment_path}\n"
    yaml_file.write_text(content)
    return yaml_file


def test_pre_experiment_sets_env_var(experiment_dir):
    """Test that pre_experiment script can set environment variables"""
    exp_dir, workdir = experiment_dir

    # Copy fixture files
    import shutil

    shutil.copy(PRE_SETUP_ENV, exp_dir / "pre_setup.py")
    shutil.copy(EXP_CHECK_ENV, exp_dir / "experiment.py")

    yaml_file = _create_yaml(
        exp_dir, "test-pre-experiment", "experiment", "pre_setup.py"
    )

    runner = CliRunner(env={"XPM_TEST_PRE_EXPERIMENT": ""})
    result = runner.invoke(
        cli,
        [
            "run-experiment",
            "--workdir",
            str(workdir),
            "--run-mode",
            "DRY_RUN",
            str(yaml_file),
        ],
    )

    assert "PRE_EXPERIMENT_TEST_PASSED" in result.output, (
        f"Pre-experiment did not execute correctly. Output: {result.output}"
    )
    assert result.exit_code == 0, f"CLI failed with: {result.output}"


def test_pre_experiment_file_not_found(experiment_dir):
    """Test error handling when pre_experiment file doesn't exist"""
    exp_dir, workdir = experiment_dir

    import shutil

    shutil.copy(EXP_SIMPLE, exp_dir / "experiment.py")

    yaml_file = _create_yaml(
        exp_dir, "test-pre-experiment-missing", "experiment", "nonexistent.py"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-experiment",
            "--workdir",
            str(workdir),
            "--run-mode",
            "DRY_RUN",
            str(yaml_file),
        ],
    )

    assert result.exit_code != 0, "Should fail when pre_experiment file doesn't exist"
    assert "not found" in result.output.lower(), (
        f"Should mention file not found. Output: {result.output}"
    )


def test_pre_experiment_execution_error(experiment_dir):
    """Test error handling when pre_experiment script has an error"""
    exp_dir, workdir = experiment_dir

    import shutil

    shutil.copy(PRE_SETUP_ERROR, exp_dir / "pre_setup.py")
    shutil.copy(EXP_SIMPLE, exp_dir / "experiment.py")

    yaml_file = _create_yaml(
        exp_dir, "test-pre-experiment-error", "experiment", "pre_setup.py"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-experiment",
            "--workdir",
            str(workdir),
            "--run-mode",
            "DRY_RUN",
            str(yaml_file),
        ],
    )

    assert result.exit_code != 0, "Should fail when pre_experiment has an error"
    assert (
        "failed to execute" in result.output.lower()
        or "intentional error" in result.output.lower()
    ), f"Should show execution error. Output: {result.output}"


def test_pre_experiment_relative_path(experiment_dir):
    """Test that pre_experiment relative paths are resolved from YAML file location"""
    exp_dir, workdir = experiment_dir

    # Create a subdirectory for the pre_experiment script
    setup_dir = exp_dir / "setup"
    setup_dir.mkdir()

    import shutil

    shutil.copy(PRE_SETUP_ENV, setup_dir / "init.py")
    shutil.copy(EXP_CHECK_ENV, exp_dir / "experiment.py")

    yaml_file = _create_yaml(
        exp_dir, "test-pre-experiment-relative", "experiment", "setup/init.py"
    )

    runner = CliRunner(env={"XPM_TEST_PRE_EXPERIMENT": ""})
    result = runner.invoke(
        cli,
        [
            "run-experiment",
            "--workdir",
            str(workdir),
            "--run-mode",
            "DRY_RUN",
            str(yaml_file),
        ],
    )

    assert "PRE_EXPERIMENT_TEST_PASSED" in result.output, (
        f"Relative path resolution failed. Output: {result.output}"
    )
    assert result.exit_code == 0, f"CLI failed with: {result.output}"


def test_pre_experiment_mocking_modules(experiment_dir):
    """Test that pre_experiment can mock modules before experiment import"""
    exp_dir, workdir = experiment_dir

    import shutil

    shutil.copy(PRE_SETUP_MOCK, exp_dir / "pre_setup.py")
    shutil.copy(EXP_CHECK_MOCK, exp_dir / "experiment.py")

    yaml_file = _create_yaml(
        exp_dir, "test-pre-experiment-mock", "experiment", "pre_setup.py"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-experiment",
            "--workdir",
            str(workdir),
            "--run-mode",
            "DRY_RUN",
            str(yaml_file),
        ],
    )

    assert "MOCK_MODULE_TEST_PASSED" in result.output, (
        f"Module mocking failed. Output: {result.output}"
    )
    assert result.exit_code == 0, f"CLI failed with: {result.output}"
