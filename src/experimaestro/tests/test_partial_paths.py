"""Integration tests for partial paths and cleanup"""

from pathlib import Path
from experimaestro import (
    Task,
    Param,
    Meta,
    field,
    PathGenerator,
    subparameters,
    param_group,
)
from experimaestro.scheduler import JobState

from .utils import TemporaryExperiment, TemporaryDirectory


# Define parameter groups
iter_group = param_group("iter")


class TaskWithPartial(Task):
    """Task that uses subparameters for partial paths"""

    # Define a subparameters set
    checkpoints = subparameters(exclude_groups=[iter_group])

    # Parameter in iter_group - excluded from partial identifier
    max_iter: Param[int] = field(groups=[iter_group])

    # Parameter not in any group - included in partial identifier
    learning_rate: Param[float]

    # Path generated using the partial identifier
    checkpoint_path: Meta[Path] = field(
        default_factory=PathGenerator("checkpoint", partial=checkpoints)
    )

    def execute(self):
        # Create the checkpoint directory and a marker file
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_path / "model.pt").write_text("checkpoint data")


def test_partial_path_created():
    """Test that partial paths are correctly created during task execution"""
    with TemporaryDirectory(prefix="xpm", suffix="partial") as workdir:
        with TemporaryExperiment("partial_test", workdir=workdir, maxwait=30):
            task = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()

        assert task.__xpm__.job.state == JobState.DONE

        # Verify the partial path was created
        assert task.checkpoint_path.exists()
        assert (task.checkpoint_path / "model.pt").exists()

        # Verify the path is in the partials directory
        partials_path = workdir / "partials"
        assert partials_path.exists()

        # The checkpoint_path should be under partials/TASK_ID/checkpoints/PARTIAL_ID/
        # Use resolve() to handle symlinks like /var -> /private/var on macOS
        assert task.checkpoint_path.resolve().is_relative_to(partials_path.resolve())


def test_partial_path_shared_across_tasks():
    """Test that tasks with same non-excluded params share partial paths"""
    with TemporaryDirectory(prefix="xpm", suffix="partial_shared") as workdir:
        with TemporaryExperiment("partial_shared", workdir=workdir, maxwait=30):
            # Submit two tasks with different max_iter but same learning_rate
            task1 = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()
            task2 = TaskWithPartial.C(max_iter=200, learning_rate=0.1).submit()

        assert task1.__xpm__.job.state == JobState.DONE
        assert task2.__xpm__.job.state == JobState.DONE

        # They should share the same partial path
        assert task1.checkpoint_path == task2.checkpoint_path


def test_partial_path_different_for_different_params():
    """Test that tasks with different non-excluded params have different partial paths"""
    with TemporaryDirectory(prefix="xpm", suffix="partial_diff") as workdir:
        with TemporaryExperiment("partial_diff", workdir=workdir, maxwait=30):
            # Submit two tasks with different learning_rate
            task1 = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()
            task2 = TaskWithPartial.C(max_iter=100, learning_rate=0.2).submit()

        assert task1.__xpm__.job.state == JobState.DONE
        assert task2.__xpm__.job.state == JobState.DONE

        # They should have different partial paths
        assert task1.checkpoint_path != task2.checkpoint_path


def test_partial_registered_in_database():
    """Test that partials are registered in the database when jobs are submitted"""
    from experimaestro.scheduler.state_provider import WorkspaceStateProvider
    from experimaestro.scheduler.state_db import PartialModel, JobPartialModel

    with TemporaryDirectory(prefix="xpm", suffix="partial_db") as workdir:
        with TemporaryExperiment("partial_db", workdir=workdir, maxwait=30) as xp:
            task = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()

        assert task.__xpm__.job.state == JobState.DONE

        # Get the state provider and check database
        # Note: Must use read_only=False since the experiment left a singleton
        # with read_only=False that hasn't been closed yet
        provider = WorkspaceStateProvider.get_instance(workdir, read_only=False)

        try:
            with provider.workspace_db.bind_ctx([PartialModel, JobPartialModel]):
                # Check that partial is registered
                partials = list(PartialModel.select())
                assert len(partials) == 1
                assert partials[0].subparameters_name == "checkpoints"

                # Check that job is linked to partial
                job_partials = list(JobPartialModel.select())
                assert len(job_partials) == 1
                assert job_partials[0].partial_id == partials[0].partial_id
                assert job_partials[0].experiment_id == xp.workdir.name
        finally:
            provider.close()


def test_orphan_partial_cleanup():
    """Test that orphan partials are cleaned up when jobs are deleted"""
    from experimaestro.scheduler.state_provider import WorkspaceStateProvider
    from experimaestro.scheduler.state_db import PartialModel, JobPartialModel

    with TemporaryDirectory(prefix="xpm", suffix="partial_cleanup") as workdir:
        with TemporaryExperiment("partial_cleanup", workdir=workdir, maxwait=30) as xp:
            task = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()

        assert task.__xpm__.job.state == JobState.DONE
        checkpoint_path = task.checkpoint_path

        # Verify partial path exists
        assert checkpoint_path.exists()

        # Get the state provider
        provider = WorkspaceStateProvider.get_instance(workdir, read_only=False)

        try:
            # Delete the job
            with provider.workspace_db.bind_ctx([PartialModel, JobPartialModel]):
                job_partials = list(JobPartialModel.select())
                assert len(job_partials) == 1

            # Delete job (this also removes job-partial link)
            provider.delete_job(
                task.__xpm__.job.identifier,
                xp.workdir.name,
                xp.run_id,
            )

            # Now the partial should be orphaned
            orphans = provider.get_orphan_partials()
            assert len(orphans) == 1

            # Cleanup orphan partials
            deleted = provider.cleanup_orphan_partials(perform=True)
            assert len(deleted) == 1

            # Verify partial directory is deleted
            assert not checkpoint_path.exists()

            # Verify partial is removed from database
            with provider.workspace_db.bind_ctx([PartialModel]):
                partials = list(PartialModel.select())
                assert len(partials) == 0
        finally:
            provider.close()


def test_shared_partial_not_orphaned():
    """Test that partials shared by multiple jobs are not orphaned until all jobs deleted"""
    from experimaestro.scheduler.state_provider import WorkspaceStateProvider

    with TemporaryDirectory(prefix="xpm", suffix="partial_shared_cleanup") as workdir:
        with TemporaryExperiment(
            "partial_shared_cleanup", workdir=workdir, maxwait=30
        ) as xp:
            # Submit two tasks with same learning_rate (same partial)
            task1 = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()
            task2 = TaskWithPartial.C(max_iter=200, learning_rate=0.1).submit()

        assert task1.__xpm__.job.state == JobState.DONE
        assert task2.__xpm__.job.state == JobState.DONE

        # They share the same partial path
        checkpoint_path = task1.checkpoint_path
        assert checkpoint_path == task2.checkpoint_path
        assert checkpoint_path.exists()

        provider = WorkspaceStateProvider.get_instance(workdir, read_only=False)

        try:
            # Delete first job
            provider.delete_job(
                task1.__xpm__.job.identifier,
                xp.workdir.name,
                xp.run_id,
            )

            # Partial should NOT be orphaned (still used by task2)
            orphans = provider.get_orphan_partials()
            assert len(orphans) == 0

            # Partial directory should still exist
            assert checkpoint_path.exists()

            # Delete second job
            provider.delete_job(
                task2.__xpm__.job.identifier,
                xp.workdir.name,
                xp.run_id,
            )

            # Now partial should be orphaned
            orphans = provider.get_orphan_partials()
            assert len(orphans) == 1

            # Cleanup
            deleted = provider.cleanup_orphan_partials(perform=True)
            assert len(deleted) == 1
            assert not checkpoint_path.exists()
        finally:
            provider.close()
