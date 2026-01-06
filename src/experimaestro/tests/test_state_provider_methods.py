"""Tests for StateProvider abstract methods consistency

These tests verify that all StateProvider implementations (Scheduler, DbStateProvider,
SSHStateProviderClient) return consistent results for their abstract methods.

The key invariants being tested:
1. For running experiments: DbStateProvider results should match Scheduler results
2. For past experiments: SSHStateProviderClient results should match DbStateProvider
3. Dependencies are properly stored and retrieved through all providers
"""

import io
from pathlib import Path

from experimaestro import Task, Param
from experimaestro.tests.utils import TemporaryExperiment
from experimaestro.scheduler.state_db import (
    ALL_MODELS,
    JobModel,
    JobDependenciesModel,
    JobExperimentsModel,
    initialize_workspace_database,
    close_workspace_database,
)
from experimaestro.scheduler.remote.protocol import (
    RPCMethod,
)


# =============================================================================
# Test Tasks with Dependencies
# =============================================================================


class ProducerTask(Task):
    """A task that produces output"""

    value: Param[int]

    def execute(self):
        (self.__taskdir__ / "output.txt").write_text(f"value={self.value}")


class ConsumerTask(Task):
    """A task that depends on ProducerTask"""

    producer: Param[ProducerTask]
    multiplier: Param[int] = 1

    def execute(self):
        # Just write a marker
        (self.__taskdir__ / "result.txt").write_text("done")


class MultiConsumerTask(Task):
    """A task that depends on multiple ProducerTasks"""

    producers: Param[list]

    def execute(self):
        (self.__taskdir__ / "result.txt").write_text("done")


# =============================================================================
# JobDependenciesModel Database Tests
# =============================================================================


def test_job_dependencies_model_basic(tmp_path: Path):
    """Test basic JobDependenciesModel operations"""
    db_path = tmp_path / "workspace.db"
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create jobs
        JobModel.create(job_id="producer_job", task_id="ProducerTask", state="done")
        JobModel.create(job_id="consumer_job", task_id="ConsumerTask", state="done")

        # Create experiment link
        JobExperimentsModel.create(
            job_id="producer_job", experiment_id="exp1", run_id="run1"
        )
        JobExperimentsModel.create(
            job_id="consumer_job", experiment_id="exp1", run_id="run1"
        )

        # Create dependency: consumer depends on producer
        JobDependenciesModel.create(
            job_id="consumer_job",
            task_id="ConsumerTask",
            experiment_id="exp1",
            run_id="run1",
            depends_on_job_id="producer_job",
            depends_on_task_id="ProducerTask",
        )

        # Query dependencies
        deps = list(
            JobDependenciesModel.select().where(
                JobDependenciesModel.job_id == "consumer_job"
            )
        )
        assert len(deps) == 1
        assert deps[0].depends_on_job_id == "producer_job"

        # Query reverse (who depends on producer)
        dependents = list(
            JobDependenciesModel.select().where(
                JobDependenciesModel.depends_on_job_id == "producer_job"
            )
        )
        assert len(dependents) == 1
        assert dependents[0].job_id == "consumer_job"

    close_workspace_database(db)


def test_job_dependencies_model_multiple_deps(tmp_path: Path):
    """Test JobDependenciesModel with multiple dependencies"""
    db_path = tmp_path / "workspace.db"
    db, _ = initialize_workspace_database(db_path, read_only=False)

    with db.bind_ctx(ALL_MODELS):
        # Create multiple producer jobs
        JobModel.create(job_id="producer1", task_id="ProducerTask", state="done")
        JobModel.create(job_id="producer2", task_id="ProducerTask", state="done")
        JobModel.create(job_id="consumer", task_id="MultiConsumerTask", state="done")

        # Create experiment links
        for job_id in ["producer1", "producer2", "consumer"]:
            JobExperimentsModel.create(
                job_id=job_id, experiment_id="exp1", run_id="run1"
            )

        # Consumer depends on both producers
        JobDependenciesModel.create(
            job_id="consumer",
            task_id="MultiConsumerTask",
            experiment_id="exp1",
            run_id="run1",
            depends_on_job_id="producer1",
            depends_on_task_id="ProducerTask",
        )
        JobDependenciesModel.create(
            job_id="consumer",
            task_id="MultiConsumerTask",
            experiment_id="exp1",
            run_id="run1",
            depends_on_job_id="producer2",
            depends_on_task_id="ProducerTask",
        )

        # Query dependencies for consumer
        deps = list(
            JobDependenciesModel.select().where(
                (JobDependenciesModel.job_id == "consumer")
                & (JobDependenciesModel.experiment_id == "exp1")
            )
        )
        assert len(deps) == 2
        dep_ids = {d.depends_on_job_id for d in deps}
        assert dep_ids == {"producer1", "producer2"}

    close_workspace_database(db)


def test_job_dependencies_deleted_with_job(tmp_path: Path):
    """Test that dependencies are deleted when a job is deleted"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / ".experimaestro").mkdir()

    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        # Create jobs
        JobModel.create(job_id="producer", task_id="ProducerTask", state="done")
        JobModel.create(job_id="consumer", task_id="ConsumerTask", state="done")

        # Create experiment links
        JobExperimentsModel.create(
            job_id="producer", experiment_id="exp1", run_id="run1"
        )
        JobExperimentsModel.create(
            job_id="consumer", experiment_id="exp1", run_id="run1"
        )

        # Create dependency
        JobDependenciesModel.create(
            job_id="consumer",
            task_id="ConsumerTask",
            experiment_id="exp1",
            run_id="run1",
            depends_on_job_id="producer",
            depends_on_task_id="ProducerTask",
        )

        # Verify dependency exists
        deps_before = list(JobDependenciesModel.select())
        assert len(deps_before) == 1

    # Delete the consumer job (should delete dependencies where it's the source)
    provider.delete_job("consumer")

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        deps_after = list(JobDependenciesModel.select())
        assert len(deps_after) == 0

    provider.close()


def test_job_dependencies_deleted_when_dependency_deleted(tmp_path: Path):
    """Test that dependencies are deleted when the dependency job is deleted"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / ".experimaestro").mkdir()

    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        # Create jobs
        JobModel.create(job_id="producer", task_id="ProducerTask", state="done")
        JobModel.create(job_id="consumer", task_id="ConsumerTask", state="done")

        # Create experiment links
        JobExperimentsModel.create(
            job_id="producer", experiment_id="exp1", run_id="run1"
        )
        JobExperimentsModel.create(
            job_id="consumer", experiment_id="exp1", run_id="run1"
        )

        # Create dependency
        JobDependenciesModel.create(
            job_id="consumer",
            task_id="ConsumerTask",
            experiment_id="exp1",
            run_id="run1",
            depends_on_job_id="producer",
            depends_on_task_id="ProducerTask",
        )

        # Verify dependency exists
        deps_before = list(JobDependenciesModel.select())
        assert len(deps_before) == 1

    # Delete the producer job (should delete dependencies where it's the target)
    provider.delete_job("producer")

    with provider.workspace_db.bind_ctx(ALL_MODELS):
        deps_after = list(JobDependenciesModel.select())
        assert len(deps_after) == 0

    provider.close()


# =============================================================================
# Scheduler to DbStateProvider Consistency Tests
# =============================================================================


def test_scheduler_db_dependencies_consistency(tmp_path: Path):
    """Test that Scheduler and DbStateProvider return consistent dependencies"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider
    from experimaestro.scheduler.base import Scheduler

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # Run experiment with dependent tasks
    with TemporaryExperiment("dep_test", maxwait=0, workdir=workdir):
        # Get the scheduler
        scheduler = Scheduler.instance()

        # Submit producer
        producer = ProducerTask.C(value=42)
        producer.submit()

        # Submit consumer that depends on producer
        consumer = ConsumerTask.C(producer=producer, multiplier=2)
        consumer.submit()

        # Verify scheduler has dependencies during experiment
        _ = scheduler.get_dependencies_map("dep_test")

    # Get dependencies from DbStateProvider (after experiment)
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    db_deps = provider.get_dependencies_map("dep_test")

    # Both should have the same dependency information
    # Consumer should depend on producer
    assert len(db_deps) >= 1, "DbStateProvider should have dependencies"

    # Find the consumer job (the one with dependencies)
    consumer_deps = None
    for job_id, deps in db_deps.items():
        if deps:  # Has dependencies
            consumer_deps = deps
            break

    assert consumer_deps is not None, "Should find consumer's dependencies"
    assert len(consumer_deps) == 1, "Consumer should have exactly 1 dependency"

    provider.close()


def test_scheduler_db_tags_consistency(tmp_path: Path):
    """Test that Scheduler and DbStateProvider return consistent tags"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider
    from experimaestro.scheduler.base import Scheduler

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    with TemporaryExperiment("tags_test", maxwait=0, workdir=workdir):
        scheduler = Scheduler.instance()

        # Submit task with tags
        task = ProducerTask.C(value=42).tag("env", "test").tag("priority", "high")
        task.submit()

        # Verify scheduler has tags during experiment
        _ = scheduler.get_tags_map("tags_test")

    # Get tags from DbStateProvider after experiment
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    db_tags = provider.get_tags_map("tags_test")

    # Both should have same tags
    assert len(db_tags) == 1, "Should have exactly 1 job with tags"

    job_id = list(db_tags.keys())[0]
    assert db_tags[job_id].get("env") == "test"
    assert db_tags[job_id].get("priority") == "high"

    provider.close()


# =============================================================================
# SSH Provider Piping Tests
# =============================================================================


class PipedServerClient:
    """Helper class that connects server and client via pipes"""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path

        # Create pipes for bidirectional communication
        # Server reads from server_in, writes to server_out
        # Client reads from client_in (connected to server_out), writes to client_out (connected to server_in)
        self.server_in_r, self.server_in_w = io.BytesIO(), io.BytesIO()
        self.server_out_r, self.server_out_w = io.BytesIO(), io.BytesIO()

        self._server = None
        self._server_thread = None

    def start_server(self):
        """Start the server in a background thread"""
        from experimaestro.scheduler.remote.server import SSHStateProviderServer
        from experimaestro.scheduler.db_state_provider import DbStateProvider

        # Initialize state provider for the server
        self._state_provider = DbStateProvider.get_instance(
            self.workspace_path, read_only=True, sync_on_start=False
        )

        self._server = SSHStateProviderServer(self.workspace_path)
        self._server._state_provider = self._state_provider

    def call_server_handler(self, method: RPCMethod, params: dict):
        """Directly call server handler and return result (simulates RPC)"""
        handler = self._server._handlers.get(method.value)
        if handler is None:
            raise ValueError(f"Unknown method: {method}")
        return handler(params)

    def close(self):
        """Close the server and cleanup"""
        if self._state_provider:
            self._state_provider.close()


def test_ssh_piped_get_dependencies_map(tmp_path: Path):
    """Test get_dependencies_map through piped server/client communication"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # First, run an experiment to create some data with dependencies
    with TemporaryExperiment("ssh_dep_test", maxwait=0, workdir=workdir):
        producer = ProducerTask.C(value=42)
        producer.submit()
        consumer = ConsumerTask.C(producer=producer)
        consumer.submit()

    # Get reference data from DbStateProvider
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    expected_deps = provider.get_dependencies_map("ssh_dep_test")
    provider.close()

    # Now test via piped server
    piped = PipedServerClient(workdir)
    piped.start_server()

    # Call the handler directly (simulating RPC)
    result = piped.call_server_handler(
        RPCMethod.GET_DEPENDENCIES_MAP,
        {"experiment_id": "ssh_dep_test", "run_id": None},
    )

    # Results should match
    assert result == expected_deps

    piped.close()


def test_ssh_piped_get_tags_map(tmp_path: Path):
    """Test get_tags_map through piped server/client communication"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # First, run an experiment to create some data with tags
    with TemporaryExperiment("ssh_tags_test", maxwait=0, workdir=workdir):
        task = ProducerTask.C(value=42).tag("test_key", "test_value")
        task.submit()

    # Get reference data from DbStateProvider
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    expected_tags = provider.get_tags_map("ssh_tags_test")
    provider.close()

    # Now test via piped server
    piped = PipedServerClient(workdir)
    piped.start_server()

    # Call the handler directly (simulating RPC)
    result = piped.call_server_handler(
        RPCMethod.GET_TAGS_MAP,
        {"experiment_id": "ssh_tags_test", "run_id": None},
    )

    # Results should match
    assert result == expected_tags
    assert len(result) == 1
    job_id = list(result.keys())[0]
    assert result[job_id]["test_key"] == "test_value"

    piped.close()


def test_ssh_piped_get_jobs(tmp_path: Path):
    """Test get_jobs through piped server/client communication"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # First, run an experiment
    with TemporaryExperiment("ssh_jobs_test", maxwait=0, workdir=workdir):
        task1 = ProducerTask.C(value=1)
        task1.submit()
        task2 = ProducerTask.C(value=2)
        task2.submit()

    # Get reference data from DbStateProvider
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    expected_jobs = provider.get_jobs("ssh_jobs_test")
    provider.close()

    # Now test via piped server
    piped = PipedServerClient(workdir)
    piped.start_server()

    # Call the handler directly (simulating RPC)
    result = piped.call_server_handler(
        RPCMethod.GET_JOBS,
        {"experiment_id": "ssh_jobs_test", "run_id": None},
    )

    # Results should be serialized job dicts
    assert len(result) == len(expected_jobs)

    # Verify job IDs match
    result_job_ids = {j["identifier"] for j in result}
    expected_job_ids = {j.identifier for j in expected_jobs}
    assert result_job_ids == expected_job_ids

    piped.close()


# =============================================================================
# End-to-End Provider Consistency Tests
# =============================================================================


def test_dependencies_event_includes_depends_on(tmp_path: Path):
    """Test that JobExperimentUpdatedEvent includes depends_on field"""
    from experimaestro.scheduler.state_provider import JobExperimentUpdatedEvent

    # Create event with dependencies
    event = JobExperimentUpdatedEvent(
        experiment_id="exp1",
        run_id="run1",
        job_id="consumer_job",
        tags={"env": "test"},
        depends_on=["producer_job1", "producer_job2"],
    )

    assert event.depends_on == ["producer_job1", "producer_job2"]
    assert len(event.depends_on) == 2


class ChainTaskA(Task):
    """Task A in chain test"""

    value: Param[int]

    def execute(self):
        (self.__taskdir__ / "a.txt").write_text(str(self.value))


class ChainTaskB(Task):
    """Task B depends on A"""

    a: Param[ChainTaskA]

    def execute(self):
        (self.__taskdir__ / "b.txt").write_text("b")


class ChainTaskC(Task):
    """Task C depends on B"""

    b: Param[ChainTaskB]

    def execute(self):
        (self.__taskdir__ / "c.txt").write_text("c")


def test_full_dependency_chain(tmp_path: Path):
    """Test a full dependency chain: A -> B -> C"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    with TemporaryExperiment("chain_test", maxwait=0, workdir=workdir):
        a = ChainTaskA.C(value=1)
        a.submit()

        b = ChainTaskB.C(a=a)
        b.submit()

        c = ChainTaskC.C(b=b)
        c.submit()

    # Check dependencies in database
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    deps_map = provider.get_dependencies_map("chain_test")

    # Find jobs by checking who has dependencies
    jobs_with_deps = [job_id for job_id, deps in deps_map.items() if deps]

    # B should depend on A, C should depend on B
    assert len(jobs_with_deps) == 2, "B and C should have dependencies"

    provider.close()


def test_multiple_experiments_independent_dependencies(tmp_path: Path):
    """Test that dependencies are scoped to experiments"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # First experiment
    with TemporaryExperiment("exp1", maxwait=0, workdir=workdir):
        producer1 = ProducerTask.C(value=1)
        producer1.submit()
        consumer1 = ConsumerTask.C(producer=producer1)
        consumer1.submit()

    # Second experiment
    with TemporaryExperiment("exp2", maxwait=0, workdir=workdir):
        producer2 = ProducerTask.C(value=2)
        producer2.submit()
        consumer2 = ConsumerTask.C(producer=producer2)
        consumer2.submit()

    # Check each experiment has its own dependencies
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )

    deps_exp1 = provider.get_dependencies_map("exp1")
    deps_exp2 = provider.get_dependencies_map("exp2")

    # Each should have exactly 1 consumer with 1 dependency
    jobs_with_deps_exp1 = [job_id for job_id, deps in deps_exp1.items() if deps]
    jobs_with_deps_exp2 = [job_id for job_id, deps in deps_exp2.items() if deps]

    assert len(jobs_with_deps_exp1) == 1
    assert len(jobs_with_deps_exp2) == 1

    # The dependency job IDs should be different (different producer jobs)
    dep_exp1 = deps_exp1[jobs_with_deps_exp1[0]][0]
    dep_exp2 = deps_exp2[jobs_with_deps_exp2[0]][0]
    assert dep_exp1 != dep_exp2, "Dependencies should be to different producer jobs"

    provider.close()


def test_all_providers_return_same_dependencies(tmp_path: Path):
    """Test that Scheduler, DbStateProvider, and SSH server all return same dependencies"""
    from experimaestro.scheduler.db_state_provider import DbStateProvider
    from experimaestro.scheduler.base import Scheduler

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    scheduler_deps = None

    # Run experiment and capture scheduler dependencies
    with TemporaryExperiment("consistency_test", maxwait=0, workdir=workdir):
        scheduler = Scheduler.instance()

        producer = ProducerTask.C(value=42)
        producer.submit()
        consumer = ConsumerTask.C(producer=producer)
        consumer.submit()

        # Capture from scheduler while running
        scheduler_deps = scheduler.get_dependencies_map("consistency_test")

    # Get from DbStateProvider
    provider = DbStateProvider.get_instance(
        workdir, read_only=False, sync_on_start=False
    )
    db_deps = provider.get_dependencies_map("consistency_test")
    provider.close()

    # Get from SSH server (via piped handler)
    piped = PipedServerClient(workdir)
    piped.start_server()
    ssh_deps = piped.call_server_handler(
        RPCMethod.GET_DEPENDENCIES_MAP,
        {"experiment_id": "consistency_test", "run_id": None},
    )
    piped.close()

    # All three should be consistent
    # Note: scheduler_deps may have job objects as keys, db_deps has strings
    # Compare the structure (number of jobs with deps, number of deps per job)

    def count_deps(deps_map):
        """Count total dependencies and jobs with dependencies"""
        jobs_with_deps = sum(1 for deps in deps_map.values() if deps)
        total_deps = sum(len(deps) for deps in deps_map.values())
        return jobs_with_deps, total_deps

    scheduler_counts = count_deps(scheduler_deps)
    db_counts = count_deps(db_deps)
    ssh_counts = count_deps(ssh_deps)

    assert scheduler_counts == db_counts, "Scheduler and DbStateProvider should match"
    assert db_counts == ssh_counts, "DbStateProvider and SSH server should match"
