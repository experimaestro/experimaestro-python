"""Tests for v1 to v2 workspace migration

Tests cover:
1. Basic migration of experiments from xp/ to experiments/
2. Run ID generation from directory modification time
3. Collision handling for same-timestamp directories
4. Dry-run mode (no changes made)
5. Keep-old mode (xp directory renamed)
6. Empty xp directory cleanup
7. Broken symlink creation to prevent v1 usage
8. Edge cases (no experiments, no xp directory)
"""

import os
from datetime import datetime

import pytest
from click.testing import CliRunner

from experimaestro.cli import cli
from experimaestro.tests.test_workspace_state_provider import (
    create_v1_experiment,
    create_v2_experiment,
)


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory with required marker file"""
    ws = tmp_path / "workspace"
    ws.mkdir()
    # Create the marker file that identifies this as an experimaestro workspace
    (ws / ".__experimaestro__").touch()
    return ws


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


class TestMigrateV1ToV2Basic:
    """Tests for basic v1 to v2 migration functionality"""

    def test_migrate_single_experiment(self, workspace, cli_runner):
        """Single experiment should be migrated correctly"""
        # Create a v1 experiment
        create_v1_experiment(
            workspace,
            "my-experiment",
            jobs=[
                ("pkg.TaskA", "job-1", "done"),
                ("pkg.TaskB", "job-2", "done"),
            ],
        )

        # Run migration
        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0, f"Migration failed: {result.output}"
        assert "Migrated: my-experiment" in result.output

        # Verify old xp directory is gone
        assert not (workspace / "xp").exists() or (workspace / "xp").is_symlink()

        # Verify new experiment directory exists
        new_exp_dir = workspace / "experiments" / "my-experiment"
        assert new_exp_dir.exists()

        # Verify a run directory was created
        runs = list(new_exp_dir.iterdir())
        assert len(runs) == 1
        run_dir = runs[0]

        # Verify jobs symlinks were moved
        jobs_dir = run_dir / "jobs"
        assert jobs_dir.exists()

    def test_migrate_multiple_experiments(self, workspace, cli_runner):
        """Multiple experiments should all be migrated"""
        # Create multiple v1 experiments
        for i in range(3):
            create_v1_experiment(
                workspace,
                f"experiment-{i}",
                jobs=[("pkg.Task", f"job-{i}", "done")],
            )

        # Run migration
        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0
        assert "Migrated 3/3 experiment(s)" in result.output

        # Verify all experiments were migrated
        for i in range(3):
            exp_dir = workspace / "experiments" / f"experiment-{i}"
            assert exp_dir.exists(), f"experiment-{i} not migrated"
            runs = list(exp_dir.iterdir())
            assert len(runs) == 1

    def test_migrated_experiment_preserves_jobs_dir(self, workspace, cli_runner):
        """Migration should preserve the jobs directory contents"""
        create_v1_experiment(
            workspace,
            "test-exp",
            jobs=[
                ("pkg.TaskA", "job-a1", "done"),
                ("pkg.TaskA", "job-a2", "error"),
                ("pkg.TaskB", "job-b1", "running"),
            ],
        )

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Get the migrated run directory
        exp_dir = workspace / "experiments" / "test-exp"
        run_dir = list(exp_dir.iterdir())[0]
        jobs_dir = run_dir / "jobs"

        # Verify job symlinks exist
        assert (jobs_dir / "pkg.TaskA" / "job-a1").exists()
        assert (jobs_dir / "pkg.TaskA" / "job-a2").exists()
        assert (jobs_dir / "pkg.TaskB" / "job-b1").exists()


class TestMigrateRunIdGeneration:
    """Tests for run ID generation from modification time"""

    def test_run_id_format(self, workspace, cli_runner):
        """Run ID should use YYYYMMDD_HHMMSS format"""
        create_v1_experiment(workspace, "test-exp", jobs=[])

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Get the run ID
        exp_dir = workspace / "experiments" / "test-exp"
        run_dir = list(exp_dir.iterdir())[0]
        run_id = run_dir.name

        # Verify format (YYYYMMDD_HHMMSS)
        assert len(run_id) == 15, f"Unexpected run_id format: {run_id}"
        assert run_id[8] == "_", f"Missing underscore in run_id: {run_id}"
        # Verify it's a valid datetime
        datetime.strptime(run_id, "%Y%m%d_%H%M%S")

    def test_run_id_based_on_mtime(self, workspace, cli_runner):
        """Run ID should be based on directory modification time"""
        # Create experiment with specific mtime
        exp_dir = create_v1_experiment(workspace, "test-exp", jobs=[])

        # Set a specific modification time
        target_time = datetime(2025, 6, 15, 14, 30, 45)
        target_timestamp = target_time.timestamp()
        os.utime(exp_dir, (target_timestamp, target_timestamp))

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Verify run ID matches the mtime
        run_dir = list((workspace / "experiments" / "test-exp").iterdir())[0]
        assert run_dir.name == "20250615_143045"


class TestMigrateCollisionHandling:
    """Tests for collision handling when run IDs collide"""

    def test_collision_adds_suffix(self, workspace, cli_runner):
        """Colliding run IDs should get suffixes"""
        # Create v1 experiment
        exp_dir = create_v1_experiment(workspace, "test-exp", jobs=[])

        # Pre-create a v2 run with the same timestamp
        target_time = datetime(2025, 1, 1, 12, 0, 0)
        target_timestamp = target_time.timestamp()
        os.utime(exp_dir, (target_timestamp, target_timestamp))

        # Pre-create existing run directory
        existing_run = workspace / "experiments" / "test-exp" / "20250101_120000"
        existing_run.mkdir(parents=True)
        (existing_run / "status.json").write_text('{"version": 1}')

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Verify collision was handled with suffix
        runs = sorted((workspace / "experiments" / "test-exp").iterdir())
        assert len(runs) == 2
        run_names = [r.name for r in runs]
        assert "20250101_120000" in run_names
        assert "20250101_120000.1" in run_names

    def test_multiple_collisions(self, workspace, cli_runner):
        """Multiple collisions should increment suffix"""
        # Create v1 experiment
        exp_dir = create_v1_experiment(workspace, "test-exp", jobs=[])

        target_time = datetime(2025, 3, 15, 10, 0, 0)
        target_timestamp = target_time.timestamp()
        os.utime(exp_dir, (target_timestamp, target_timestamp))

        # Pre-create multiple existing run directories
        base_dir = workspace / "experiments" / "test-exp"
        (base_dir / "20250315_100000").mkdir(parents=True)
        (base_dir / "20250315_100000.1").mkdir(parents=True)
        (base_dir / "20250315_100000.2").mkdir(parents=True)

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Should have suffix .3
        assert (base_dir / "20250315_100000.3").exists()


class TestMigrateDryRun:
    """Tests for dry-run mode"""

    def test_dry_run_shows_changes(self, workspace, cli_runner):
        """Dry run should show what would be migrated"""
        create_v1_experiment(workspace, "my-exp", jobs=[("pkg.Task", "job-1", "done")])

        result = cli_runner.invoke(
            cli, ["migrate", "v1-to-v2", str(workspace), "--dry-run"]
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "my-exp" in result.output

    def test_dry_run_makes_no_changes(self, workspace, cli_runner):
        """Dry run should not modify the filesystem"""
        create_v1_experiment(workspace, "my-exp", jobs=[("pkg.Task", "job-1", "done")])

        result = cli_runner.invoke(
            cli, ["migrate", "v1-to-v2", str(workspace), "--dry-run"]
        )

        assert result.exit_code == 0

        # Verify v1 structure still exists
        assert (workspace / "xp" / "my-exp").exists()

        # Verify v2 structure was NOT created
        assert not (workspace / "experiments").exists()


class TestMigrateLeftoverFiles:
    """Tests for handling leftover files in xp/ directory"""

    def test_leftover_files_renamed(self, workspace, cli_runner):
        """Leftover files in xp/ should be preserved in xp_MIGRATED_TO_V2"""
        # Create v1 experiment
        create_v1_experiment(workspace, "my-exp", jobs=[])

        # Add extra file that won't be migrated
        (workspace / "xp" / "some-file.txt").write_text("extra content")

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0
        assert "xp_MIGRATED_TO_V2" in result.output
        assert "leftover item(s)" in result.output

        # Verify renamed directory exists with extra file
        renamed_dir = workspace / "xp_MIGRATED_TO_V2"
        assert renamed_dir.exists()
        assert (renamed_dir / "some-file.txt").exists()

        # Verify broken symlink was created
        xp_link = workspace / "xp"
        assert xp_link.is_symlink()

    def test_empty_xp_directory_removed(self, workspace, cli_runner):
        """Empty xp directory should be removed after migration"""
        create_v1_experiment(workspace, "my-exp", jobs=[])

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0
        assert "Removed empty 'xp' directory" in result.output

        # Verify xp directory is replaced by symlink
        xp_dir = workspace / "xp"
        assert xp_dir.is_symlink()
        assert not xp_dir.exists()  # Broken symlink


class TestMigrateEdgeCases:
    """Tests for edge cases in migration"""

    def test_no_xp_directory(self, workspace, cli_runner):
        """Should handle missing xp directory gracefully"""
        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0
        assert "No old 'xp' directory found" in result.output

    def test_empty_xp_directory(self, workspace, cli_runner):
        """Should handle empty xp directory gracefully"""
        (workspace / "xp").mkdir()

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0
        assert "No experiments found" in result.output

    def test_existing_v2_experiments_preserved(self, workspace, cli_runner):
        """Pre-existing v2 experiments should not be affected"""
        # Create v1 experiment
        create_v1_experiment(workspace, "v1-exp", jobs=[("pkg.Task", "job-1", "done")])

        # Create pre-existing v2 experiment
        create_v2_experiment(
            workspace,
            "v2-exp",
            runs=[("20260101_100000", "completed", [("pkg.Task", "job-2", "done")])],
            current_run="20260101_100000",
        )

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0

        # Verify v2 experiment is still there
        assert (workspace / "experiments" / "v2-exp" / "20260101_100000").exists()

        # Verify v1 experiment was migrated
        v1_migrated = workspace / "experiments" / "v1-exp"
        assert v1_migrated.exists()

    def test_migrate_preserves_jobs_bak(self, workspace, cli_runner):
        """Migration should preserve jobs.bak directory if present"""
        # Create v1 experiment with jobs.bak
        exp_dir = create_v1_experiment(
            workspace, "my-exp", jobs=[("pkg.Task", "job-1", "done")]
        )

        # Add jobs.bak directory
        jobs_bak = exp_dir / "jobs.bak"
        jobs_bak.mkdir()
        (jobs_bak / "old-task").mkdir()

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Verify jobs.bak was migrated
        run_dir = list((workspace / "experiments" / "my-exp").iterdir())[0]
        assert (run_dir / "jobs.bak" / "old-task").exists()

    def test_migration_is_idempotent(self, workspace, cli_runner):
        """Running migration twice should be safe (idempotent)"""
        # Create v1 experiment
        create_v1_experiment(workspace, "my-exp", jobs=[("pkg.Task", "job-1", "done")])

        # First migration
        result1 = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result1.exit_code == 0
        assert "Migrated: my-exp" in result1.output

        # Get the migrated experiment path
        exp_dir = workspace / "experiments" / "my-exp"
        runs_after_first = list(exp_dir.iterdir())
        assert len(runs_after_first) == 1

        # Second migration - should be a no-op
        result2 = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result2.exit_code == 0
        assert "No old 'xp' directory found" in result2.output

        # Verify experiment is unchanged
        runs_after_second = list(exp_dir.iterdir())
        assert runs_after_first == runs_after_second

    def test_migration_twice_with_leftover_files(self, workspace, cli_runner):
        """Running migration twice with leftover files should be safe"""
        # Create v1 experiment with extra file
        create_v1_experiment(workspace, "my-exp", jobs=[])
        (workspace / "xp" / "leftover.txt").write_text("content")

        # First migration - leftover files get renamed to xp_MIGRATED_TO_V2
        result1 = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result1.exit_code == 0
        assert "xp_MIGRATED_TO_V2" in result1.output

        # Second migration - xp is now a broken symlink
        result2 = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result2.exit_code == 0
        assert "No old 'xp' directory found" in result2.output


class TestMigrateBrokenSymlink:
    """Tests for broken symlink creation after migration"""

    def test_creates_broken_symlink(self, workspace, cli_runner):
        """Should create broken symlink to prevent v1 usage"""
        create_v1_experiment(workspace, "my-exp", jobs=[])

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0
        assert "Created broken 'xp' symlink" in result.output

        # Verify symlink exists and is broken
        xp_link = workspace / "xp"
        assert xp_link.is_symlink()
        assert not xp_link.exists()  # Broken symlink - target doesn't exist

        # Verify symlink target
        target = os.readlink(xp_link)
        assert "experimaestro_v2_migrated_workspace_do_not_use_v1" in target

    def test_symlink_created_even_with_leftover_files(self, workspace, cli_runner):
        """Should create broken symlink even when leftover files exist"""
        create_v1_experiment(workspace, "my-exp", jobs=[])
        (workspace / "xp" / "leftover.txt").write_text("content")

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])

        assert result.exit_code == 0

        # xp directory should be a broken symlink (leftover files moved to xp_MIGRATED_TO_V2)
        xp_link = workspace / "xp"
        assert xp_link.is_symlink()
        assert not xp_link.exists()  # Broken symlink

        # Leftover files should be in renamed directory
        assert (workspace / "xp_MIGRATED_TO_V2" / "leftover.txt").exists()


class TestMigrateWithWorkspaceStateProvider:
    """Tests verifying migrated workspaces work with WorkspaceStateProvider"""

    def test_migrated_experiments_are_readable(self, workspace, cli_runner):
        """Migrated experiments should be readable by WorkspaceStateProvider"""
        from experimaestro.scheduler.workspace_state_provider import (
            WorkspaceStateProvider,
        )

        # Create and migrate v1 experiment
        create_v1_experiment(
            workspace,
            "migrated-exp",
            jobs=[
                ("pkg.TaskA", "job-a", "done"),
                ("pkg.TaskB", "job-b", "error"),
            ],
        )

        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Use WorkspaceStateProvider to read the migrated experiment
        provider = WorkspaceStateProvider(workspace)
        experiments = provider.get_experiments()

        exp_ids = {e.experiment_id for e in experiments}
        assert "migrated-exp" in exp_ids

    def test_mixed_v1_v2_after_partial_migration(self, workspace, cli_runner):
        """Should handle workspace with both v1 and migrated v2 experiments"""
        from experimaestro.scheduler.workspace_state_provider import (
            WorkspaceStateProvider,
        )

        # Create v1 experiment
        create_v1_experiment(workspace, "v1-only", jobs=[("pkg.Task", "job-1", "done")])

        # Create v2 experiment directly (simulating already migrated)
        create_v2_experiment(
            workspace,
            "already-v2",
            runs=[("20260101_100000", "completed", [("pkg.Task", "job-2", "done")])],
            current_run="20260101_100000",
        )

        # Migrate only the v1 experiment
        result = cli_runner.invoke(cli, ["migrate", "v1-to-v2", str(workspace)])
        assert result.exit_code == 0

        # Both should be accessible
        provider = WorkspaceStateProvider(workspace)
        experiments = provider.get_experiments()

        exp_ids = {e.experiment_id for e in experiments}
        assert "v1-only" in exp_ids
        assert "already-v2" in exp_ids
