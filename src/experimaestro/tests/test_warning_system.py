"""Tests for the generic warning/action system and stale lock detection

This module tests:
- WarningEvent and ErrorEvent
- StaleLockError exception
- StateProvider warning action management
- Stale lock detection in token acquisition
- End-to-end warning flow from detection to action execution
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

from experimaestro.locking import StaleLockError
from experimaestro.scheduler.state_status import (
    WarningEvent,
    ErrorEvent,
)
from experimaestro.scheduler.state_provider import StateProvider

# Mark all tests as anyio for async support
pytestmark = [pytest.mark.anyio, pytest.mark.tokens]


@pytest.fixture(autouse=True)
def reset_async_bridge():
    """Reset AsyncEventBridge singleton before and after each test."""
    from experimaestro.ipc import AsyncEventBridge

    AsyncEventBridge.reset()
    yield
    AsyncEventBridge.reset()


# =============================================================================
# Test Fixtures
# =============================================================================


class MockStateProvider(StateProvider):
    """Mock state provider for testing warning actions"""

    def __init__(self):
        super().__init__()
        self.events_emitted = []

    def _notify_state_listeners(self, event):
        """Capture emitted events for testing"""
        self.events_emitted.append(event)
        super()._notify_state_listeners(event)

    # Abstract method stubs (not used in these tests)
    def get_experiments(self, since=None):
        return []

    def get_experiment(self, experiment_id):
        return None

    def get_experiment_runs(self, experiment_id):
        return []

    def get_current_run(self, experiment_id):
        return None

    def get_jobs(self, experiment_id, run_id=None, since=None):
        return []

    def get_job(self, task_id, identifier):
        return None

    def get_all_jobs(self, since=None):
        return []

    def get_tags_map(self, experiment_id, run_id=None):
        return {}

    def get_dependencies_map(self, experiment_id, run_id=None):
        return {}

    def get_services(self, experiment_id=None, run_id=None):
        return []

    def kill_job(self, job, perform=False):
        return True

    def clean_job(self, job, perform=False):
        return True

    def close(self):
        pass


class MockProcess:
    """Mock process for testing - doesn't validate PID exists"""

    def __init__(self, pid: int, running: bool = False):
        self.pid = pid
        self._running = running

    def isrunning(self):
        return self._running

    async def aio_isrunning(self):
        return self._running


def create_mock_job(name: str, tmpdir: str):
    """Create a mock job with all required attributes."""
    import hashlib

    class MockXPM:
        def __init__(self, name):
            self.identifier = type(
                "Identifier",
                (),
                {
                    "main": type(
                        "Main",
                        (),
                        {"hex": lambda: hashlib.sha256(name.encode()).hexdigest()},
                    )()
                },
            )()

    class MockJob:
        task_id = "mock-task"
        config = type("Config", (), {"__xpm__": MockXPM(name)})()

        @property
        def identifier(self):
            return f"mock-job-{name}"

        @property
        def basepath(self):
            return Path(tmpdir) / name

    job = MockJob()

    # Create PID file
    pid_path = job.basepath.with_suffix(".pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(json.dumps({"type": "mock"}))

    return job


# =============================================================================
# Tests: WarningEvent and ErrorEvent
# =============================================================================


class TestWarningEvents:
    """Tests for WarningEvent and ErrorEvent dataclasses"""

    def test_warning_event_creation(self):
        """WarningEvent should be created with all required fields"""
        event = WarningEvent(
            experiment_id="test-exp",
            run_id="run-1",
            warning_key="test_warning",
            description="Test warning description",
            actions={"clean": "Remove", "dismiss": "Dismiss"},
            context={"title": "Test Warning"},
            severity="warning",
        )

        assert event.warning_key == "test_warning"
        assert event.description == "Test warning description"
        assert event.actions == {"clean": "Remove", "dismiss": "Dismiss"}
        assert event.context["title"] == "Test Warning"
        assert event.severity == "warning"

    def test_warning_event_serialization(self):
        """WarningEvent should serialize to JSON correctly"""
        event = WarningEvent(
            experiment_id="test-exp",
            run_id="run-1",
            warning_key="test_warning",
            description="Test description",
            actions={"clean": "Remove"},
            context={"title": "Test"},
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_type"] == "WarningEvent"
        assert data["warning_key"] == "test_warning"
        assert data["actions"] == {"clean": "Remove"}

    def test_error_event_creation(self):
        """ErrorEvent should be created with all required fields"""
        event = ErrorEvent(
            experiment_id="test-exp",
            run_id="run-1",
            warning_key="test_warning",
            action_key="clean",
            error_message="Action failed",
        )

        assert event.warning_key == "test_warning"
        assert event.action_key == "clean"
        assert event.error_message == "Action failed"


# =============================================================================
# Tests: StateProvider Warning Action Management
# =============================================================================


class TestStateProviderWarningActions:
    """Tests for StateProvider warning action cache and execution"""

    def test_register_warning_actions(self):
        """Should register action callbacks in cache"""
        provider = MockStateProvider()

        callback1 = MagicMock()
        callback2 = MagicMock()

        provider.register_warning_actions(
            "warning1",
            {"clean": callback1, "dismiss": callback2},
        )

        assert "warning1" in provider._warning_actions
        assert provider._warning_actions["warning1"]["clean"] == callback1
        assert provider._warning_actions["warning1"]["dismiss"] == callback2

    def test_execute_warning_action_success(self):
        """Should execute registered callback successfully"""
        provider = MockStateProvider()

        callback = MagicMock()
        provider.register_warning_actions("warning1", {"clean": callback})

        provider.execute_warning_action("warning1", "clean")

        callback.assert_called_once()

    def test_execute_warning_action_missing_warning(self):
        """Should emit ErrorEvent when warning_key not found"""
        provider = MockStateProvider()

        with pytest.raises(KeyError):
            provider.execute_warning_action("nonexistent", "clean")

        # Check that ErrorEvent was emitted
        assert len(provider.events_emitted) == 1
        event = provider.events_emitted[0]
        assert isinstance(event, ErrorEvent)
        assert event.warning_key == "nonexistent"
        assert "not found" in event.error_message

    def test_execute_warning_action_missing_action(self):
        """Should emit ErrorEvent when action_key not found"""
        provider = MockStateProvider()

        callback = MagicMock()
        provider.register_warning_actions("warning1", {"clean": callback})

        with pytest.raises(KeyError):
            provider.execute_warning_action("warning1", "nonexistent")

        # Check that ErrorEvent was emitted
        assert len(provider.events_emitted) == 1
        event = provider.events_emitted[0]
        assert isinstance(event, ErrorEvent)
        assert event.action_key == "nonexistent"

    def test_execute_warning_action_callback_failure(self):
        """Should emit ErrorEvent when callback raises exception"""
        provider = MockStateProvider()

        def failing_callback():
            raise RuntimeError("Callback failed")

        provider.register_warning_actions("warning1", {"clean": failing_callback})

        with pytest.raises(RuntimeError):
            provider.execute_warning_action("warning1", "clean")

        # Check that ErrorEvent was emitted
        assert len(provider.events_emitted) == 1
        event = provider.events_emitted[0]
        assert isinstance(event, ErrorEvent)
        assert "failed" in event.error_message.lower()


# =============================================================================
# Tests: StaleLockError Exception
# =============================================================================


class TestStaleLockError:
    """Tests for StaleLockError exception"""

    def test_stale_lock_error_creation(self):
        """StaleLockError should contain all required fields"""
        callback = MagicMock()

        error = StaleLockError(
            warning_key="stale_locks_gpu_1",
            description="Found 2 stale locks",
            actions={"clean": "Remove Locks", "dismiss": "Dismiss"},
            context={"token_name": "gpu", "stale_lock_count": 2},
            callbacks={"clean": callback},
        )

        assert error.warning_key == "stale_locks_gpu_1"
        assert error.description == "Found 2 stale locks"
        assert error.actions == {"clean": "Remove Locks", "dismiss": "Dismiss"}
        assert error.context["token_name"] == "gpu"
        assert error.callbacks["clean"] == callback
        assert str(error) == "Found 2 stale locks"


# =============================================================================
# Tests: Stale Lock Detection
# =============================================================================


class TestStaleLockDetection:
    """Tests for stale lock detection in token acquisition"""

    async def test_is_stale_detects_dead_process(self):
        """DynamicLockFile.is_stale() should return True for dead processes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple lock file
            from experimaestro.locking import DynamicLockFile

            lock_path = Path(tmpdir) / "test.lock"
            lock_path.write_text(json.dumps({"job_uri": "/tmp/job1"}))

            # Create a minimal mock resource
            class MockResource:
                def _account_lock_file(self, lf):
                    pass

                def _unaccount_lock_file(self, lf):
                    pass

            lock_file = DynamicLockFile(lock_path, MockResource())

            # Verify it's not stale without a process
            assert not lock_file.is_stale()

            # Set a mock dead process
            lock_file.process = MockProcess(99999, running=False)
            lock_file.timestamp = time.time() - 7200  # 2 hours ago

            # Should be stale now
            assert lock_file.is_stale(min_age_seconds=3600)

            # Should not be stale if recent
            lock_file.timestamp = time.time() - 60  # 1 minute ago
            assert not lock_file.is_stale(min_age_seconds=3600)


# =============================================================================
# Tests: Stale Lock Cleanup Callback
# =============================================================================


class TestStaleLockCleanup:
    """Tests for stale lock cleanup callbacks"""

    def test_stale_lock_error_callback_cleans_locks(self):
        """StaleLockError cleanup callback should remove lock files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake lock file
            lock_path = Path(tmpdir) / "test.lock"
            lock_path.write_text(json.dumps({"job_uri": "/tmp/job1"}))
            assert lock_path.exists()

            # Create a callback that removes the file
            files_to_remove = [lock_path]

            def cleanup_callback():
                for f in files_to_remove:
                    f.unlink()

            # Create a StaleLockError with the callback
            error = StaleLockError(
                warning_key="test_stale",
                description="Test stale lock",
                actions={"clean": "Remove", "dismiss": "Dismiss"},
                context={"count": 1},
                callbacks={"clean": cleanup_callback, "dismiss": lambda: None},
            )

            # Execute the cleanup callback
            error.callbacks["clean"]()

            # File should be removed
            assert not lock_path.exists()


# =============================================================================
# Tests: BaseExperiment.finalize_status()
# =============================================================================


class TestExperimentFinalizeStatus:
    """Tests for BaseExperiment.finalize_status() method"""

    async def test_finalize_status_with_cleanup(self):
        """finalize_status should consolidate events and clean up when cleanup_events=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            exp_id = "test-exp"
            run_id = "run-1"

            # Set up directory structure
            exp_base = workspace / "experiments" / exp_id
            run_dir = exp_base / run_id
            run_dir.mkdir(parents=True)

            # Create status.json with events_count
            status_path = run_dir / "status.json"
            status_data = {
                "experiment_id": exp_id,
                "run_id": run_id,
                "status": "running",
                "events_count": 0,
            }
            status_path.write_text(json.dumps(status_data))

            # Create event files in .events directory
            events_dir = workspace / ".events" / "experiments" / exp_id
            events_dir.mkdir(parents=True)
            event_file = events_dir / "events-0.jsonl"
            event_file.write_text(
                '{"event_type": "JobSubmittedEvent", "job_id": "test-job"}\n'
            )

            # Create mock experiment
            from experimaestro.scheduler.state_provider import MockExperiment

            from experimaestro.scheduler.interfaces import ExperimentStatus

            exp = MockExperiment(
                workdir=run_dir,
                run_id=run_id,
                status=ExperimentStatus.RUNNING,
            )
            exp._events_count = 0

            # Call finalize_status with cleanup
            await exp.finalize_status(cleanup_events=True)

            # Events should be archived
            archived_events = run_dir / "events"
            assert archived_events.exists()

            # Temp events should be removed
            assert not event_file.exists()

            # Status should not have events_count field
            new_status = json.loads(status_path.read_text())
            assert "events_count" not in new_status

    async def test_finalize_status_callback(self):
        """finalize_status should call callback to modify experiment state"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            exp_id = "test-exp"
            run_id = "run-1"

            run_dir = workspace / "experiments" / exp_id / run_id
            run_dir.mkdir(parents=True)

            from experimaestro.scheduler.state_provider import MockExperiment
            from experimaestro.scheduler.interfaces import ExperimentStatus

            exp = MockExperiment(
                workdir=run_dir,
                run_id=run_id,
                status=ExperimentStatus.RUNNING,
            )

            # Callback to mark experiment as done
            def mark_done(exp):
                exp._status = ExperimentStatus.DONE

            # Call finalize_status with callback
            await exp.finalize_status(callback=mark_done, cleanup_events=False)

            # Status should be updated
            assert exp.status == ExperimentStatus.DONE

            # Status file should reflect the change
            status_path = run_dir / "status.json"
            status_data = json.loads(status_path.read_text())
            assert status_data["status"] == "done"
