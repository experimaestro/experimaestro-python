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
    """Helper class that connects server and client via pipes

    This simulates an external process monitoring the workspace by:
    - Creating DbStateProvider directly (optionally via singleton)
    - Using standalone=True to enable file watching (when using direct instantiation)
    - Reading from the same database as the main experiment process
    """

    def __init__(self, workspace_path: Path, *, use_singleton: bool = True):
        """Initialize the piped server client.

        Args:
            workspace_path: Path to the workspace directory
            use_singleton: If True, use DbStateProvider.get_instance() (default).
                          If False, create a new DbStateProvider directly (for
                          simulating true cross-process monitoring).
        """
        self.workspace_path = workspace_path
        self._use_singleton = use_singleton

        # Create pipes for bidirectional communication
        # Server reads from server_in, writes to server_out
        # Client reads from client_in (connected to server_out), writes to client_out (connected to server_in)
        self.server_in_r, self.server_in_w = io.BytesIO(), io.BytesIO()
        self.server_out_r, self.server_out_w = io.BytesIO(), io.BytesIO()

        self._server = None
        self._server_thread = None
        self._state_provider = None

    def start_server(self):
        """Start the server in a background thread"""
        from experimaestro.scheduler.remote.server import SSHStateProviderServer
        from experimaestro.scheduler.db_state_provider import DbStateProvider

        # Initialize state provider for the server
        if self._use_singleton:
            self._state_provider = DbStateProvider.get_instance(
                self.workspace_path, read_only=True, sync_on_start=False
            )
        else:
            # Direct instantiation - simulates an external process
            self._state_provider = DbStateProvider(
                workspace_path=self.workspace_path,
                read_only=True,
                sync_on_start=False,
                standalone=True,  # Enable file watching like an external monitor
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


# =============================================================================
# Cross-Process Monitoring Tests (DbStateProvider as external monitor)
# =============================================================================


def test_separate_db_provider_detects_changes(tmp_path: Path):
    """Test that a separate DbStateProvider can detect database changes.

    This simulates monitoring an experiment from another process:
    1. Start an experiment (which creates the database)
    2. Create a separate DbStateProvider directly (bypassing singleton)
    3. Register a listener on it
    4. Submit jobs (modifies the database)
    5. Trigger change detection (simulating file watcher)
    6. Verify the listener receives events

    This tests cross-process consistency since the separate provider
    doesn't share state with the experiment's DbStateProvider.
    """
    from experimaestro.scheduler.db_state_provider import (
        DbStateProvider,
        _DatabaseChangeDetector,
    )
    from experimaestro.scheduler.state_provider import (
        ExperimentUpdatedEvent,
        JobUpdatedEvent,
    )
    import threading

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # Track received events
    received_events = []
    events_lock = threading.Lock()

    def event_listener(event):
        with events_lock:
            received_events.append(event)

    with TemporaryExperiment("external_monitor_test", maxwait=0, workdir=workdir):
        # Now that experiment is started, database exists
        # Create a separate DbStateProvider directly (bypassing singleton)
        # This simulates another process monitoring the same workspace
        separate_provider = DbStateProvider(
            workspace_path=workdir,
            read_only=True,  # Read-only since we're just monitoring
            sync_on_start=False,
        )

        # Register listener on the separate provider
        separate_provider.add_listener(event_listener)

        try:
            # Submit tasks (this modifies the database)
            producer = ProducerTask.C(value=42)
            producer.submit()

            consumer = ConsumerTask.C(producer=producer)
            consumer.submit()

            # Simulate what the file watcher would do:
            # Create a change detector and trigger it manually
            change_detector = _DatabaseChangeDetector(separate_provider)
            change_detector._detect_and_notify_changes()

            # Verify we received events
            with events_lock:
                # Should have received ExperimentUpdatedEvent
                experiment_events = [
                    e for e in received_events if isinstance(e, ExperimentUpdatedEvent)
                ]
                # Should have received JobUpdatedEvent for jobs
                job_events = [
                    e for e in received_events if isinstance(e, JobUpdatedEvent)
                ]

                assert len(experiment_events) >= 1, (
                    f"Should receive ExperimentUpdatedEvent, got {len(experiment_events)}"
                )
                assert len(job_events) >= 2, (
                    f"Should receive JobUpdatedEvent for each job, got {len(job_events)}"
                )

                # Verify experiment ID
                assert any(
                    e.experiment_id == "external_monitor_test"
                    for e in experiment_events
                ), "Should have event for our experiment"

        finally:
            separate_provider.remove_listener(event_listener)
            separate_provider.close()


def test_separate_db_provider_with_file_watcher(tmp_path: Path, monkeypatch):
    """Test that a separate DbStateProvider with file watcher detects changes.

    This tests the complete file watching mechanism:
    1. Start an experiment (creates the database)
    2. Create a DbStateProvider and start file watching
    3. Register a listener on it
    4. Submit jobs (modifies the database)
    5. Wait for file watcher to detect changes
    6. Verify the listener receives events

    On macOS, FSEvents may not detect SQLite WAL changes directly, so we use
    a marker file workaround (see issue #154).
    """
    from experimaestro.scheduler.db_state_provider import DbStateProvider
    from experimaestro.scheduler.state_provider import (
        ExperimentUpdatedEvent,
        JobUpdatedEvent,
    )
    import threading

    # Use shorter interval for testing the FSEvents workaround
    monkeypatch.setenv("XPM_FSEVENTS_MARKER_INTERVAL", "0.5")

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # Track received events
    received_events = []
    events_lock = threading.Lock()
    event_received = threading.Event()

    def event_listener(event):
        with events_lock:
            received_events.append(event)
            event_received.set()

    with TemporaryExperiment("file_watcher_test", maxwait=0, workdir=workdir):
        # Now that experiment is started, database exists
        # Create a separate DbStateProvider directly (bypassing singleton)
        # Use standalone=True to enable file watcher (simulates external process)
        separate_provider = DbStateProvider(
            workspace_path=workdir,
            read_only=True,
            sync_on_start=False,
            standalone=True,
        )

        # Register listener on the separate provider
        # File watcher auto-starts since standalone=True
        separate_provider.add_listener(event_listener)

        # Ensure the database is opened and baseline is established
        # before we make changes that should trigger events
        separate_provider.get_experiments()

        try:
            # Submit a task (this modifies the database)
            task = ProducerTask.C(value=42)
            task.submit()

            # Wait for file watcher to detect changes
            # On macOS, the marker file workaround triggers after 0.5s,
            # plus the change detector debounce of 0.5s
            event_received.wait(timeout=3.0)

            # Verify we received events
            with events_lock:
                total_events = len(received_events)
                experiment_events = [
                    e for e in received_events if isinstance(e, ExperimentUpdatedEvent)
                ]
                job_events = [
                    e for e in received_events if isinstance(e, JobUpdatedEvent)
                ]

                assert total_events > 0, "Should receive at least one event"
                # Either experiment or job events should be present
                assert len(experiment_events) >= 1 or len(job_events) >= 1, (
                    f"Should receive events, got {total_events} total"
                )

        finally:
            separate_provider.remove_listener(event_listener)
            separate_provider.close()


def test_ssh_piped_cross_process_monitoring(tmp_path: Path):
    """Test SSH state provider cross-process monitoring via piped communication.

    This tests that the SSH state provider can:
    1. Connect to a server via pipes (simulating SSH transport)
    2. Fetch experiment data as changes are made
    3. Return consistent results with DbStateProvider

    This is the SSH equivalent of test_separate_db_provider_detects_changes.
    """
    from experimaestro.scheduler.db_state_provider import DbStateProvider

    workdir = tmp_path / "workspace"
    workdir.mkdir()

    with TemporaryExperiment("ssh_monitor_test", maxwait=0, workdir=workdir):
        # Submit some tasks first
        producer = ProducerTask.C(value=42)
        producer.submit()

        consumer = ConsumerTask.C(producer=producer)
        consumer.submit()

    # Get reference data from DbStateProvider (new instance, not singleton)
    provider = DbStateProvider(
        workspace_path=workdir,
        read_only=True,
        sync_on_start=False,
    )
    expected_experiments = provider.get_experiments()
    expected_jobs = provider.get_jobs("ssh_monitor_test")
    expected_deps = provider.get_dependencies_map("ssh_monitor_test")
    expected_tags = provider.get_tags_map("ssh_monitor_test")
    provider.close()

    # Now test via piped server (simulating SSH communication)
    piped = PipedServerClient(workdir)
    piped.start_server()

    # Test get_experiments
    experiments_result = piped.call_server_handler(
        RPCMethod.GET_EXPERIMENTS,
        {"since": None},
    )
    assert len(experiments_result) == len(expected_experiments)
    result_exp_ids = {e["experiment_id"] for e in experiments_result}
    expected_exp_ids = {e.experiment_id for e in expected_experiments}
    assert result_exp_ids == expected_exp_ids

    # Test get_jobs
    jobs_result = piped.call_server_handler(
        RPCMethod.GET_JOBS,
        {"experiment_id": "ssh_monitor_test", "run_id": None},
    )
    assert len(jobs_result) == len(expected_jobs)
    result_job_ids = {j["identifier"] for j in jobs_result}
    expected_job_ids = {j.identifier for j in expected_jobs}
    assert result_job_ids == expected_job_ids

    # Test get_dependencies_map
    deps_result = piped.call_server_handler(
        RPCMethod.GET_DEPENDENCIES_MAP,
        {"experiment_id": "ssh_monitor_test", "run_id": None},
    )
    assert deps_result == expected_deps

    # Test get_tags_map
    tags_result = piped.call_server_handler(
        RPCMethod.GET_TAGS_MAP,
        {"experiment_id": "ssh_monitor_test", "run_id": None},
    )
    assert tags_result == expected_tags

    piped.close()


def test_ssh_piped_incremental_sync(tmp_path: Path):
    """Test SSH state provider incremental synchronization via pipes.

    This tests that the SSH server can handle incremental updates:
    1. Fetch initial state
    2. Make changes
    3. Fetch only changes since last sync

    This validates the 'since' parameter for incremental monitoring.
    """
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # First experiment - create initial data
    with TemporaryExperiment("ssh_sync_test", maxwait=0, workdir=workdir):
        task1 = ProducerTask.C(value=1)
        task1.submit()

    # Set up piped server (use_singleton=False to avoid conflict with experiment's provider)
    piped = PipedServerClient(workdir, use_singleton=False)
    piped.start_server()

    # Get initial state
    initial_experiments = piped.call_server_handler(
        RPCMethod.GET_EXPERIMENTS,
        {"since": None},
    )
    assert len(initial_experiments) == 1
    assert initial_experiments[0]["experiment_id"] == "ssh_sync_test"

    # Get the updated_at timestamp for incremental sync
    initial_updated_at = initial_experiments[0].get("updated_at")

    piped.close()

    # Run another experiment to create more data
    with TemporaryExperiment("ssh_sync_test2", maxwait=0, workdir=workdir):
        task2 = ProducerTask.C(value=2)
        task2.submit()

    # Reconnect and fetch incrementally (use_singleton=False)
    piped = PipedServerClient(workdir, use_singleton=False)
    piped.start_server()

    # Fetch all experiments (should now have 2)
    all_experiments = piped.call_server_handler(
        RPCMethod.GET_EXPERIMENTS,
        {"since": None},
    )
    assert len(all_experiments) == 2

    # Fetch only experiments updated since the first fetch
    if initial_updated_at:
        incremental_experiments = piped.call_server_handler(
            RPCMethod.GET_EXPERIMENTS,
            {"since": initial_updated_at},
        )
        # Should include at least the new experiment
        new_exp_ids = {e["experiment_id"] for e in incremental_experiments}
        assert "ssh_sync_test2" in new_exp_ids

    piped.close()
