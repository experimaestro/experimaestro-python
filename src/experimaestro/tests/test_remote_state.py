"""Tests for SSH-based remote state provider

Tests cover:
- Protocol serialization/deserialization
- Server request handling
- Client-server communication (using pipes instead of SSH)
- File synchronization logic
"""

import io
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experimaestro.scheduler.remote.protocol import (
    JSONRPC_VERSION,
    RPCMethod,
    NotificationMethod,
    RPCRequest,
    RPCResponse,
    RPCNotification,
    RPCError,
    parse_message,
    create_request,
    create_success_response,
    create_error_response,
    create_notification,
    serialize_datetime,
    deserialize_datetime,
)
from experimaestro.scheduler.state_provider import (
    MockJob,
    MockExperiment,
    MockService,
)
from experimaestro.notifications import LevelInformation

# Mark all tests in this module as state tests
pytestmark = [pytest.mark.state]


# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocolMessages:
    """Test JSON-RPC message creation and parsing"""

    def test_request_creation(self):
        """Test creating a JSON-RPC request"""
        req_json = create_request(RPCMethod.GET_EXPERIMENTS, {"since": None}, 1)
        data = json.loads(req_json)

        assert data["jsonrpc"] == JSONRPC_VERSION
        assert data["method"] == "get_experiments"
        assert data["params"] == {"since": None}
        assert data["id"] == 1

    def test_request_parsing(self):
        """Test parsing a JSON-RPC request"""
        req_json = '{"jsonrpc": "2.0", "method": "get_jobs", "params": {"experiment_id": "exp1"}, "id": 42}'
        msg = parse_message(req_json)

        assert isinstance(msg, RPCRequest)
        assert msg.method == "get_jobs"
        assert msg.params == {"experiment_id": "exp1"}
        assert msg.id == 42

    def test_response_creation(self):
        """Test creating a JSON-RPC response"""
        resp_json = create_success_response(1, [{"id": "test"}])
        data = json.loads(resp_json)

        assert data["jsonrpc"] == JSONRPC_VERSION
        assert data["id"] == 1
        assert data["result"] == [{"id": "test"}]
        assert "error" not in data

    def test_error_response_creation(self):
        """Test creating a JSON-RPC error response"""
        resp_json = create_error_response(
            1, -32600, "Invalid request", {"detail": "missing method"}
        )
        data = json.loads(resp_json)

        assert data["jsonrpc"] == JSONRPC_VERSION
        assert data["id"] == 1
        assert data["error"]["code"] == -32600
        assert data["error"]["message"] == "Invalid request"
        assert data["error"]["data"] == {"detail": "missing method"}

    def test_response_parsing(self):
        """Test parsing a JSON-RPC response"""
        resp_json = '{"jsonrpc": "2.0", "id": 1, "result": {"success": true}}'
        msg = parse_message(resp_json)

        assert isinstance(msg, RPCResponse)
        assert msg.id == 1
        assert msg.result == {"success": True}
        assert msg.error is None

    def test_error_response_parsing(self):
        """Test parsing a JSON-RPC error response"""
        resp_json = '{"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}'
        msg = parse_message(resp_json)

        assert isinstance(msg, RPCResponse)
        assert msg.id == 1
        assert msg.result is None
        assert msg.error.code == -32601
        assert msg.error.message == "Method not found"

    def test_notification_creation(self):
        """Test creating a JSON-RPC notification"""
        notif_json = create_notification(
            NotificationMethod.STATE_EVENT,
            {"event_type": "JobStateChangedEvent", "data": {"job_id": "job1"}},
        )
        data = json.loads(notif_json)

        assert data["jsonrpc"] == JSONRPC_VERSION
        assert data["method"] == "notification.state_event"
        assert data["params"] == {
            "event_type": "JobStateChangedEvent",
            "data": {"job_id": "job1"},
        }
        assert "id" not in data

    def test_notification_parsing(self):
        """Test parsing a JSON-RPC notification"""
        notif_json = '{"jsonrpc": "2.0", "method": "notification.shutdown", "params": {"reason": "test"}}'
        msg = parse_message(notif_json)

        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.shutdown"
        assert msg.params == {"reason": "test"}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON raises ValueError"""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_message("not valid json")

    def test_parse_missing_version(self):
        """Test parsing message without jsonrpc version raises ValueError"""
        with pytest.raises(ValueError, match="Invalid or missing jsonrpc version"):
            parse_message('{"method": "test", "id": 1}')

    def test_parse_wrong_version(self):
        """Test parsing message with wrong jsonrpc version raises ValueError"""
        with pytest.raises(ValueError, match="Invalid or missing jsonrpc version"):
            parse_message('{"jsonrpc": "1.0", "method": "test", "id": 1}')


class TestDatetimeSerialization:
    """Test datetime serialization helpers"""

    def test_serialize_none(self):
        """Test serializing None"""
        assert serialize_datetime(None) is None

    def test_serialize_datetime(self):
        """Test serializing datetime object"""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = serialize_datetime(dt)
        assert result == "2024-01-15T10:30:00"

    def test_serialize_timestamp(self):
        """Test serializing Unix timestamp"""
        # 2024-01-01 00:00:00 UTC (adjusted for local timezone)
        result = serialize_datetime(1704067200.0)
        assert "2024-01-01" in result

    def test_serialize_string_passthrough(self):
        """Test that strings pass through unchanged"""
        result = serialize_datetime("2024-01-15T10:30:00")
        assert result == "2024-01-15T10:30:00"

    def test_deserialize_none(self):
        """Test deserializing None"""
        assert deserialize_datetime(None) is None

    def test_deserialize_iso_string(self):
        """Test deserializing ISO format string"""
        result = deserialize_datetime("2024-01-15T10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_roundtrip(self):
        """Test datetime serialization roundtrip"""
        original = datetime(2024, 6, 15, 14, 30, 45)
        serialized = serialize_datetime(original)
        deserialized = deserialize_datetime(serialized)
        assert deserialized == original


class TestJobSerialization:
    """Test job serialization using state_dict"""

    def test_serialize_mock_job(self):
        """Test serializing a MockJob using state_dict()"""
        # Note: tags, experiment_id, run_id are not part of MockJob
        # as they are experiment-specific
        job = MockJob(
            identifier="job123",
            task_id="task.MyTask",
            path=Path("/tmp/jobs/job123"),
            state="running",
            submittime=1704067200.0,
            starttime=1704067300.0,
            endtime=None,
            progress=[],
            updated_at="2024-01-01T00:00:00",
        )

        result = job.state_dict()

        assert result["job_id"] == "job123"
        assert result["task_id"] == "task.MyTask"
        assert result["path"] == "/tmp/jobs/job123"
        # State is serialized from JobState enum - case may vary
        assert result["state"].upper() == "RUNNING"


class TestExperimentSerialization:
    """Test experiment serialization using state_dict"""

    def test_serialize_mock_experiment(self):
        """Test serializing a MockExperiment using state_dict()"""
        from experimaestro.scheduler.interfaces import ExperimentStatus

        # New layout: experiments/{experiment_id}/{run_id}
        exp = MockExperiment(
            workdir=Path("/tmp/experiments/myexp/run_20240101"),
            run_id="run_20240101",
            status=ExperimentStatus.RUNNING,
            started_at=1704067200.0,
            ended_at=None,
            hostname="server1",
        )

        result = exp.state_dict()

        assert result["experiment_id"] == "myexp"
        assert result["run_id"] == "run_20240101"
        assert result["status"] == "running"


class TestServiceSerialization:
    """Test service serialization using state_dict"""

    def test_serialize_mock_service(self):
        """Test serializing a MockService using full_state_dict()

        MockService.full_state_dict() preserves the original service class name
        (not MockService's class name) to enable proper round-trip serialization.
        """
        service = MockService(
            service_id="svc123",
            description_text="Test service",
            state_dict_data={"port": 8080},
            service_class="mymodule.MyService",
            experiment_id="exp1",
            run_id="run1",
            url="http://localhost:8080",
        )

        result = service.full_state_dict()

        assert result["service_id"] == "svc123"
        assert result["description"] == "Test service"
        # Preserves original service class, not MockService's class name
        assert result["class"] == "mymodule.MyService"
        assert result["state_dict"] == {"port": 8080}

    def test_serialize_mock_service_no_class(self):
        """Test serializing a MockService with service_class=None"""
        service = MockService(
            service_id="svc123",
            description_text="Test service",
            state_dict_data={"port": 8080},
            service_class=None,
        )

        result = service.full_state_dict()

        # class is always present, even when None
        assert result["class"] is None
        assert result["service_id"] == "svc123"
        assert result["description"] == "Test service"
        assert result["state_dict"] == {"port": 8080}


# =============================================================================
# SSH Round-Trip Tests (Mock → full_state_dict → client → Mock)
# =============================================================================


class MockSSHClient:
    """Minimal mock of SSHStateProviderClient for testing deserialization"""

    def __init__(self, remote_workspace: str, local_cache_dir: Path):
        import threading as _threading

        self.remote_workspace = remote_workspace
        self.local_cache_dir = local_cache_dir
        # Add cache attributes to match SSHStateProviderClient (from OfflineStateProvider)
        self._job_cache: dict[str, MockJob] = {}
        self._job_cache_lock = _threading.Lock()
        self._experiment_cache: dict[tuple[str, str], MockExperiment] = {}
        self._experiment_cache_lock = _threading.Lock()

    def _parse_datetime_to_timestamp(self, value) -> float | None:
        """Convert datetime value to Unix timestamp"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt.timestamp()
            except ValueError:
                return None
        if isinstance(value, datetime):
            return value.timestamp()
        return None

    def _get_cached_job(self, full_id: str) -> MockJob | None:
        """Mock implementation of inherited method"""
        with self._job_cache_lock:
            return self._job_cache.get(full_id)

    def _cache_job(self, full_id: str, job: MockJob) -> None:
        """Mock implementation of inherited method"""
        with self._job_cache_lock:
            self._job_cache[full_id] = job

    def _get_cached_experiment(
        self, experiment_id: str, run_id: str
    ) -> MockExperiment | None:
        """Mock implementation of inherited method"""
        with self._experiment_cache_lock:
            return self._experiment_cache.get((experiment_id, run_id))

    def _cache_experiment(
        self, experiment_id: str, run_id: str, exp: MockExperiment
    ) -> None:
        """Mock implementation of inherited method"""
        with self._experiment_cache_lock:
            self._experiment_cache[(experiment_id, run_id)] = exp

    def _update_job_from_dict(self, job: MockJob, d: dict) -> None:
        """Mock implementation - does nothing for tests"""
        pass

    def _update_experiment_from_dict(self, exp: MockExperiment, d: dict) -> None:
        """Mock implementation - does nothing for tests"""
        pass


class TestSSHRoundTrip:
    """Test SSH round-trip serialization: Mock → state_dict → from_state_dict"""

    def test_mockjob_ssh_roundtrip(self, tmp_path: Path):
        """Test MockJob round-trip through SSH serialization path"""
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        job_path = workspace_path / "jobs" / "my.Task" / "abc123"

        # Create original MockJob
        # Note: tags, experiment_id, run_id are not part of MockJob
        original = MockJob(
            identifier="abc123",
            task_id="my.Task",
            path=job_path,
            state="running",
            submittime=1234567890.0,
            starttime=1234567891.0,
            endtime=None,
            progress=[LevelInformation(level=0, progress=0.5, desc="halfway")],
            updated_at="2024-01-01T00:00:00",
            exit_code=None,
            retry_count=2,
        )

        # Server-side: serialize using state_dict
        serialized = original.state_dict()

        # Client-side: deserialize using _dict_to_job (which uses from_state_dict)
        mock_client = MockSSHClient(
            remote_workspace=str(workspace_path),
            local_cache_dir=workspace_path,
        )
        restored = SSHStateProviderClient._dict_to_job(mock_client, serialized)

        # Verify equality by comparing state_dict outputs
        assert original.state_dict() == restored.state_dict()

    def test_mockexperiment_ssh_roundtrip(self, tmp_path: Path):
        """Test MockExperiment round-trip through SSH serialization path"""
        from experimaestro.scheduler.interfaces import ExperimentStatus
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        # New layout: experiments/{exp-id}/{run-id}/
        (workspace_path / "experiments" / "test_exp" / "run_001").mkdir(parents=True)

        # Create original MockExperiment
        original = MockExperiment(
            workdir=workspace_path / "experiments" / "test_exp" / "run_001",
            run_id="run_001",
            status=ExperimentStatus.RUNNING,
            started_at=1234567890.0,
            ended_at=None,
            hostname="testhost",
        )

        # Server-side: serialize using state_dict
        serialized = original.state_dict()

        # Client-side: deserialize using _dict_to_experiment
        mock_client = MockSSHClient(
            remote_workspace=str(workspace_path),
            local_cache_dir=workspace_path,
        )
        restored = SSHStateProviderClient._dict_to_experiment(mock_client, serialized)

        # Verify equality by comparing state_dict outputs
        assert original.state_dict() == restored.state_dict()

    def test_mockservice_ssh_roundtrip(self, tmp_path: Path):
        """Test MockService round-trip through SSH serialization path"""
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        # Create original MockService (no service_class to avoid Service recreation)
        original = MockService(
            service_id="svc_123",
            description_text="Test service description",
            state_dict_data={"port": 8080, "host": "localhost"},
            service_class=None,
            experiment_id="exp1",
            run_id="run1",
            url="http://localhost:8080",
        )

        # Server-side: serialize using full_state_dict
        serialized = original.full_state_dict()

        # Client-side: deserialize using _dict_to_service
        mock_client = MockSSHClient(
            remote_workspace="/tmp/workspace",
            local_cache_dir=tmp_path,
        )
        restored = SSHStateProviderClient._dict_to_service(mock_client, serialized)

        # Verify equality by comparing full_state_dict outputs
        assert original.full_state_dict() == restored.full_state_dict()


# =============================================================================
# Server Tests
# =============================================================================


class TestServerRequestHandling:
    """Test server request handling with mocked state provider"""

    @pytest.fixture
    def mock_state_provider(self):
        """Create a mock state provider"""
        provider = MagicMock()
        provider.workspace_path = Path("/tmp/workspace")
        provider.get_experiments.return_value = []
        provider.get_experiment.return_value = None
        provider.get_experiment_runs.return_value = []
        provider.get_jobs.return_value = []
        provider.get_job.return_value = None
        provider.get_all_jobs.return_value = []
        provider.get_services.return_value = []
        provider.get_last_sync_time.return_value = None
        return provider

    @pytest.fixture
    def server_with_mock(self, mock_state_provider, tmp_path):
        """Create a server with mocked state provider"""
        from experimaestro.scheduler.remote.server import SSHStateProviderServer

        # Create workspace directory
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".experimaestro").mkdir()

        server = SSHStateProviderServer(workspace)
        server._state_provider = mock_state_provider
        return server

    def test_handle_get_experiments(self, server_with_mock, mock_state_provider):
        """Test handling get_experiments request"""
        from experimaestro.scheduler.interfaces import ExperimentStatus

        # New layout: experiments/{exp-id}/{run-id}
        mock_exp = MockExperiment(
            workdir=Path("/tmp/experiments/exp1/run1"),
            run_id="run1",
            status=ExperimentStatus.RUNNING,
        )
        mock_state_provider.get_experiments.return_value = [mock_exp]

        result = server_with_mock._handle_get_experiments({"since": None})

        assert len(result) == 1
        assert result[0]["experiment_id"] == "exp1"
        mock_state_provider.get_experiments.assert_called_once_with(since=None)

    def test_handle_get_experiment(self, server_with_mock, mock_state_provider):
        """Test handling get_experiment request"""
        from experimaestro.scheduler.interfaces import ExperimentStatus

        # New layout: experiments/{exp-id}/{run-id}
        mock_exp = MockExperiment(
            workdir=Path("/tmp/experiments/exp1/run1"),
            run_id="run1",
            status=ExperimentStatus.RUNNING,
        )
        mock_state_provider.get_experiment.return_value = mock_exp

        result = server_with_mock._handle_get_experiment({"experiment_id": "exp1"})

        assert result["experiment_id"] == "exp1"
        assert result["run_id"] == "run1"
        mock_state_provider.get_experiment.assert_called_once_with("exp1")

    def test_handle_get_experiment_not_found(
        self, server_with_mock, mock_state_provider
    ):
        """Test handling get_experiment when experiment not found"""
        mock_state_provider.get_experiment.return_value = None

        result = server_with_mock._handle_get_experiment(
            {"experiment_id": "nonexistent"}
        )

        assert result is None

    def test_handle_get_experiment_runs(self, server_with_mock, mock_state_provider):
        """Test handling get_experiment_runs request"""
        from experimaestro.scheduler.interfaces import ExperimentStatus

        mock_run = MockExperiment(
            workdir=Path("/tmp/experiments/exp1/run1"),
            run_id="run1",
            status=ExperimentStatus.DONE,
            hostname="server1",
            started_at=1704067200.0,
            ended_at=1704070800.0,
        )
        mock_state_provider.get_experiment_runs.return_value = [mock_run]

        result = server_with_mock._handle_get_experiment_runs({"experiment_id": "exp1"})

        assert len(result) == 1
        assert result[0]["run_id"] == "run1"
        assert result[0]["status"] == "done"

    def test_handle_get_jobs(self, server_with_mock, mock_state_provider):
        """Test handling get_jobs request"""
        mock_job = MockJob(
            identifier="job1",
            task_id="task.Test",
            path=Path("/tmp/jobs/job1"),
            state="done",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )
        mock_state_provider.get_jobs.return_value = [mock_job]

        result = server_with_mock._handle_get_jobs(
            {
                "experiment_id": "exp1",
                "run_id": "run1",
            }
        )

        assert len(result) == 1
        assert result[0]["job_id"] == "job1"

    def test_handle_get_job(self, server_with_mock, mock_state_provider):
        """Test handling get_job request"""
        mock_job = MockJob(
            identifier="job1",
            task_id="task.Test",
            path=Path("/tmp/jobs/job1"),
            state="running",
            submittime=1704067200.0,
            starttime=1704067300.0,
            endtime=None,
            progress=[],
            updated_at="2024-01-01T00:00:00",
        )
        mock_state_provider.get_job.return_value = mock_job

        result = server_with_mock._handle_get_job(
            {"job_id": "job1", "experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["job_id"] == "job1"
        assert result["task_id"] == "task.Test"

    def test_handle_get_job_not_found(self, server_with_mock, mock_state_provider):
        """Test handling get_job when job not found"""
        mock_state_provider.get_job.return_value = None

        result = server_with_mock._handle_get_job(
            {"job_id": "nonexistent", "experiment_id": "exp1", "run_id": "run1"}
        )

        assert result is None

    def test_handle_get_all_jobs(self, server_with_mock, mock_state_provider):
        """Test handling get_all_jobs request"""
        mock_job1 = MockJob(
            identifier="job1",
            task_id="task.Test1",
            path=Path("/tmp/jobs/job1"),
            state="done",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )
        mock_job2 = MockJob(
            identifier="job2",
            task_id="task.Test2",
            path=Path("/tmp/jobs/job2"),
            state="running",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )
        mock_state_provider.get_all_jobs.return_value = [mock_job1, mock_job2]

        result = server_with_mock._handle_get_all_jobs({"state": None, "tags": None})

        assert len(result) == 2
        assert result[0]["job_id"] == "job1"
        assert result[1]["job_id"] == "job2"

    def test_handle_get_services(self, server_with_mock, mock_state_provider):
        """Test handling get_services request"""
        mock_service = MockService(
            service_id="svc1",
            description_text="Test service",
            state_dict_data={"port": 8080},
            service_class="mymodule.MyService",
            experiment_id="exp1",
            run_id="run1",
            url="http://localhost:8080",
        )
        mock_state_provider.get_services.return_value = [mock_service]

        result = server_with_mock._handle_get_services(
            {"experiment_id": "exp1", "run_id": "run1"}
        )

        assert len(result) == 1
        assert result[0]["service_id"] == "svc1"
        assert result[0]["description"] == "Test service"

    def test_handle_get_tags_map(self, server_with_mock, mock_state_provider):
        """Test handling get_tags_map request"""
        mock_state_provider.get_tags_map.return_value = {
            "job1": {"model": "bert", "dataset": "squad"},
            "job2": {"model": "gpt"},
        }

        result = server_with_mock._handle_get_tags_map(
            {"experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["job1"]["model"] == "bert"
        assert result["job2"]["model"] == "gpt"

    def test_handle_get_tags_map_missing_experiment(self, server_with_mock):
        """Test get_tags_map raises TypeError when experiment_id missing"""
        with pytest.raises(TypeError, match="experiment_id is required"):
            server_with_mock._handle_get_tags_map({"run_id": "run1"})

    def test_handle_get_dependencies_map(self, server_with_mock, mock_state_provider):
        """Test handling get_dependencies_map request"""
        mock_state_provider.get_dependencies_map.return_value = {
            "job2": ["job1"],
            "job3": ["job1", "job2"],
        }

        result = server_with_mock._handle_get_dependencies_map(
            {"experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["job2"] == ["job1"]
        assert result["job3"] == ["job1", "job2"]

    def test_handle_get_dependencies_map_missing_experiment(self, server_with_mock):
        """Test get_dependencies_map raises TypeError when experiment_id missing"""
        with pytest.raises(TypeError, match="experiment_id is required"):
            server_with_mock._handle_get_dependencies_map({"run_id": "run1"})

    def test_handle_kill_job(self, server_with_mock, mock_state_provider):
        """Test handling kill_job request"""
        mock_job = MockJob(
            identifier="job1",
            task_id="task.Test",
            path=Path("/tmp/jobs/job1"),
            state="running",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )
        mock_state_provider.get_job.return_value = mock_job
        mock_state_provider.kill_job.return_value = True

        result = server_with_mock._handle_kill_job(
            {"job_id": "job1", "experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["success"] is True

    def test_handle_kill_job_not_found(self, server_with_mock, mock_state_provider):
        """Test handling kill_job when job not found"""
        mock_state_provider.get_job.return_value = None

        result = server_with_mock._handle_kill_job(
            {"job_id": "nonexistent", "experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["success"] is False
        assert "error" in result

    def test_handle_clean_job(self, server_with_mock, mock_state_provider):
        """Test handling clean_job request"""
        mock_job = MockJob(
            identifier="job1",
            task_id="task.Test",
            path=Path("/tmp/jobs/job1"),
            state="done",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )
        mock_state_provider.get_job.return_value = mock_job
        mock_state_provider.clean_job.return_value = True

        result = server_with_mock._handle_clean_job(
            {"job_id": "job1", "experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["success"] is True

    def test_handle_clean_job_not_found(self, server_with_mock, mock_state_provider):
        """Test handling clean_job when job not found"""
        mock_state_provider.get_job.return_value = None

        result = server_with_mock._handle_clean_job(
            {"job_id": "nonexistent", "experiment_id": "exp1", "run_id": "run1"}
        )

        assert result["success"] is False
        assert "error" in result

    def test_handle_get_sync_info(self, server_with_mock):
        """Test handling get_sync_info request"""
        result = server_with_mock._handle_get_sync_info({})

        assert "workspace_path" in result
        assert "last_sync_time" in result

    def test_handle_unknown_method(self, server_with_mock):
        """Test that unknown methods are not in handlers"""
        assert "unknown_method" not in server_with_mock._handlers


# =============================================================================
# Client-Server Integration Tests (using pipes)
# =============================================================================


class TestClientServerIntegration:
    """Integration tests using pipes instead of SSH"""

    @pytest.fixture
    def pipe_pair(self):
        """Create a pair of pipes for client-server communication"""
        # Server reads from client_to_server, writes to server_to_client
        # Client reads from server_to_client, writes to client_to_server
        client_to_server_r, client_to_server_w = io.BytesIO(), io.BytesIO()
        server_to_client_r, server_to_client_w = io.BytesIO(), io.BytesIO()

        return {
            "server_stdin": client_to_server_r,
            "server_stdout": server_to_client_w,
            "client_stdin": client_to_server_w,
            "client_stdout": server_to_client_r,
        }

    def test_request_response_cycle(self):
        """Test a complete request-response cycle"""
        # Simulate server response
        request = create_request(RPCMethod.GET_EXPERIMENTS, {"since": None}, 1)
        response = create_success_response(1, [])

        # Parse request
        req_msg = parse_message(request)
        assert isinstance(req_msg, RPCRequest)
        assert req_msg.method == "get_experiments"

        # Parse response
        resp_msg = parse_message(response)
        assert isinstance(resp_msg, RPCResponse)
        assert resp_msg.result == []

    def test_notification_job_updated(self):
        """Test job_updated notification message handling via STATE_EVENT"""
        notification = create_notification(
            NotificationMethod.STATE_EVENT,
            {
                "event_type": "JobStateChangedEvent",
                "data": {
                    "job_id": "job1",
                    "experiment_id": "exp1",
                    "run_id": "run1",
                },
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.state_event"
        assert msg.params["event_type"] == "JobStateChangedEvent"
        assert msg.params["data"]["job_id"] == "job1"

    def test_notification_experiment_updated(self):
        """Test experiment_updated notification message handling via STATE_EVENT"""
        notification = create_notification(
            NotificationMethod.STATE_EVENT,
            {
                "event_type": "ExperimentUpdatedEvent",
                "data": {
                    "experiment_id": "exp1",
                },
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.state_event"
        assert msg.params["event_type"] == "ExperimentUpdatedEvent"
        assert msg.params["data"]["experiment_id"] == "exp1"

    def test_notification_run_updated(self):
        """Test run_updated notification message handling via STATE_EVENT"""
        notification = create_notification(
            NotificationMethod.STATE_EVENT,
            {
                "event_type": "RunUpdatedEvent",
                "data": {
                    "experiment_id": "exp1",
                    "run_id": "run1",
                },
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.state_event"
        assert msg.params["event_type"] == "RunUpdatedEvent"
        assert msg.params["data"]["run_id"] == "run1"

    def test_notification_service_updated(self):
        """Test service_updated notification message handling via STATE_EVENT"""
        notification = create_notification(
            NotificationMethod.STATE_EVENT,
            {
                "event_type": "ServiceAddedEvent",
                "data": {
                    "experiment_id": "exp1",
                    "run_id": "run1",
                    "service_id": "svc1",
                },
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.state_event"
        assert msg.params["event_type"] == "ServiceAddedEvent"
        assert msg.params["data"]["service_id"] == "svc1"

    def test_notification_file_changed(self):
        """Test file_changed notification message handling"""
        notification = create_notification(
            NotificationMethod.FILE_CHANGED,
            {
                "path": "/workspace/logs/job.log",
                "event_type": "modified",
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.file_changed"
        assert msg.params["path"] == "/workspace/logs/job.log"

    def test_notification_shutdown(self):
        """Test shutdown notification message handling"""
        notification = create_notification(
            NotificationMethod.SHUTDOWN,
            {
                "reason": "server_stop",
                "code": 0,
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.shutdown"
        assert msg.params["reason"] == "server_stop"


# =============================================================================
# Client Tests
# =============================================================================


class TestClientDataConversion:
    """Test client data conversion methods"""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a client instance with mocked temp directory"""
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        client = SSHStateProviderClient(
            host="testhost",
            remote_workspace="/remote/workspace",
        )
        # Manually set up temp directory for testing (normally done in connect())
        client._temp_dir = str(tmp_path)
        client.local_cache_dir = tmp_path
        client.workspace_path = tmp_path

        return client

    def test_dict_to_job(self, client, tmp_path):
        """Test converting dictionary to MockJob"""
        # Note: tags, experiment_id, run_id are not part of MockJob
        # as they are experiment-specific
        job_dict = {
            "job_id": "job123",
            "task_id": "task.MyTask",
            "path": "/remote/workspace/jobs/job123",
            "state": "running",
            "submitted_time": "2024-01-01T10:00:00",
            "started_time": "2024-01-01T10:01:00",
            "ended_time": None,
            "progress": [],
        }

        job = client._dict_to_job(job_dict)

        assert job.identifier == "job123"
        assert job.task_id == "task.MyTask"
        # Path should be mapped to local cache
        assert job.path == tmp_path / "jobs/job123"

    def test_dict_to_experiment(self, client, tmp_path):
        """Test converting dictionary to MockExperiment"""
        # New layout: experiments/{experiment_id}/{run_id}
        exp_dict = {
            "experiment_id": "myexp",
            "workdir": "/remote/workspace/experiments/myexp/run1",
            "run_id": "run1",
            "status": "running",
            "hostname": "server1",
        }

        exp = client._dict_to_experiment(exp_dict)

        assert exp.experiment_id == "myexp"
        # Path should be mapped to local cache
        assert exp.workdir == tmp_path / "experiments/myexp/run1"
        assert exp.hostname == "server1"

    def test_path_mapping_outside_workspace(self, client):
        """Test path mapping for paths outside remote workspace"""
        job_dict = {
            "job_id": "job123",
            "task_id": "task.MyTask",
            "path": "/other/path/job123",  # Not under remote_workspace
            "state": "done",
            "progress": [],
        }

        job = client._dict_to_job(job_dict)

        # Path outside workspace should be kept as-is
        assert job.path == Path("/other/path/job123")


# =============================================================================
# Synchronizer Tests
# =============================================================================


class TestRemoteFileSynchronizer:
    """Test file synchronization logic"""

    @pytest.fixture
    def synchronizer(self, tmp_path):
        """Create a synchronizer instance"""
        from experimaestro.scheduler.remote.sync import RemoteFileSynchronizer

        local_cache = tmp_path / "cache"
        local_cache.mkdir()

        return RemoteFileSynchronizer(
            host="testhost",
            remote_workspace=Path("/remote/workspace"),
            local_cache=local_cache,
        )

    def test_get_local_path(self, synchronizer):
        """Test mapping remote path to local path"""
        remote_path = "/remote/workspace/xp/exp1/jobs.jsonl"
        local_path = synchronizer.get_local_path(remote_path)

        assert local_path == synchronizer.local_cache / "xp/exp1/jobs.jsonl"

    def test_get_local_path_outside_workspace(self, synchronizer):
        """Test mapping path outside workspace"""
        remote_path = "/other/path/file.txt"
        local_path = synchronizer.get_local_path(remote_path)

        # Should return the original path
        assert local_path == Path("/other/path/file.txt")

    @patch("subprocess.run")
    def test_rsync_command_construction(self, mock_run, synchronizer):
        """Test that rsync command is constructed correctly"""
        mock_run.return_value = MagicMock(returncode=0)

        synchronizer._rsync(
            "testhost:/remote/workspace/logs/",
            str(synchronizer.local_cache / "logs") + "/",
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert "rsync" in cmd
        assert "--inplace" in cmd
        assert "--delete" in cmd
        assert "-L" in cmd
        assert "-a" in cmd
        assert "-z" in cmd
        assert "-v" in cmd
        assert "testhost:/remote/workspace/logs/" in cmd


# =============================================================================
# Version and Temp Directory Tests
# =============================================================================


class TestVersionStripping:
    """Test version string manipulation"""

    def test_strip_dev_version(self):
        """Test stripping .devN suffix from versions"""
        from experimaestro.scheduler.remote.client import _strip_dev_version

        assert _strip_dev_version("2.0.0b3.dev8") == "2.0.0b3"
        assert _strip_dev_version("1.2.3.dev1") == "1.2.3"
        assert _strip_dev_version("1.2.3") == "1.2.3"
        assert _strip_dev_version("2.0.0a1.dev100") == "2.0.0a1"
        assert _strip_dev_version("0.1.0.dev0") == "0.1.0"

    def test_strip_dev_preserves_prerelease(self):
        """Test that pre-release tags are preserved"""
        from experimaestro.scheduler.remote.client import _strip_dev_version

        assert _strip_dev_version("1.0.0a1.dev5") == "1.0.0a1"
        assert _strip_dev_version("1.0.0b2.dev3") == "1.0.0b2"
        assert _strip_dev_version("1.0.0rc1.dev1") == "1.0.0rc1"


class TestTempDirectory:
    """Test temporary directory handling for client cache"""

    def test_client_temp_dir_not_created_until_connect(self):
        """Test that temp directory is not created until connect() is called"""
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        client = SSHStateProviderClient(
            host="testhost",
            remote_workspace="/remote",
        )

        # Before connect, temp_dir should be None
        assert client._temp_dir is None
        assert client.local_cache_dir is None


# =============================================================================
# Event Serialization Tests (SSH Pipeline)
# =============================================================================


class TestEventSerialization:
    """Test event serialization/deserialization through SSH pipeline

    These tests verify that all event types can be correctly serialized
    on the server side and deserialized on the client side.
    """

    def test_job_state_changed_event_serialization(self):
        """Test JobStateChangedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import JobStateChangedEvent
        from dataclasses import asdict

        # Create event with all fields
        event = JobStateChangedEvent(
            job_id="job123",
            state="running",
            failure_reason=None,
            submitted_time="2024-01-01T10:00:00",
            started_time="2024-01-01T10:01:00",
            ended_time=None,
            exit_code=None,
            retry_count=2,
            progress=[],
            from_experiment=False,
            timestamp=1704067260.0,
        )

        # Server side: serialize using asdict (as server.py does)
        event_dict = asdict(event)

        # Verify from_experiment is in the serialized dict
        assert "from_experiment" in event_dict
        assert event_dict["from_experiment"] is False

        # Client side: deserialize using EventBase.from_dict
        from experimaestro.scheduler.state_status import EventBase

        restored = EventBase.from_dict(
            {"event_type": "JobStateChangedEvent", **event_dict}
        )

        assert isinstance(restored, JobStateChangedEvent)
        assert restored.job_id == "job123"
        assert restored.state == "running"
        assert restored.from_experiment is False
        assert restored.retry_count == 2

    def test_job_state_changed_event_from_experiment_true(self):
        """Test that from_experiment=True is preserved through serialization"""
        from experimaestro.scheduler.state_status import JobStateChangedEvent, EventBase
        from dataclasses import asdict

        # Create event with from_experiment=True (as experiment events set it)
        event = JobStateChangedEvent(
            job_id="job456",
            state="done",
            from_experiment=True,
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)
        assert event_dict["from_experiment"] is True

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "JobStateChangedEvent", **event_dict}
        )

        assert isinstance(restored, JobStateChangedEvent)
        assert restored.from_experiment is True

    def test_job_submitted_event_serialization(self):
        """Test JobSubmittedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import (
            JobSubmittedEvent,
            EventBase,
            JobTag,
        )
        from dataclasses import asdict

        event = JobSubmittedEvent(
            job_id="job789",
            experiment_id="exp1",
            task_id="my.Task",
            run_id="run_001",
            transient=0,
            tags=[JobTag(key="model", value="bert")],
            depends_on=["job123", "job456"],
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "JobSubmittedEvent", **event_dict}
        )

        assert isinstance(restored, JobSubmittedEvent)
        assert restored.job_id == "job789"
        assert restored.experiment_id == "exp1"
        assert restored.task_id == "my.Task"
        assert restored.depends_on == ["job123", "job456"]

    def test_job_progress_event_serialization(self):
        """Test JobProgressEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import JobProgressEvent, EventBase
        from dataclasses import asdict

        event = JobProgressEvent(
            job_id="job123",
            level=0,
            progress=0.75,
            desc="Processing batch 75/100",
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict({"event_type": "JobProgressEvent", **event_dict})

        assert isinstance(restored, JobProgressEvent)
        assert restored.job_id == "job123"
        assert restored.level == 0
        assert restored.progress == 0.75
        assert restored.desc == "Processing batch 75/100"

    def test_experiment_updated_event_serialization(self):
        """Test ExperimentUpdatedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import (
            ExperimentUpdatedEvent,
            EventBase,
        )
        from dataclasses import asdict

        event = ExperimentUpdatedEvent(
            experiment_id="myexp",
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "ExperimentUpdatedEvent", **event_dict}
        )

        assert isinstance(restored, ExperimentUpdatedEvent)
        assert restored.experiment_id == "myexp"

    def test_run_updated_event_serialization(self):
        """Test RunUpdatedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import RunUpdatedEvent, EventBase
        from dataclasses import asdict

        event = RunUpdatedEvent(
            experiment_id="myexp",
            run_id="run_001",
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict({"event_type": "RunUpdatedEvent", **event_dict})

        assert isinstance(restored, RunUpdatedEvent)
        assert restored.experiment_id == "myexp"
        assert restored.run_id == "run_001"

    def test_run_completed_event_serialization(self):
        """Test RunCompletedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import RunCompletedEvent, EventBase
        from dataclasses import asdict

        event = RunCompletedEvent(
            experiment_id="myexp",
            run_id="run_001",
            status="completed",
            ended_at="2024-01-01T12:00:00",
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "RunCompletedEvent", **event_dict}
        )

        assert isinstance(restored, RunCompletedEvent)
        assert restored.experiment_id == "myexp"
        assert restored.run_id == "run_001"
        assert restored.status == "completed"

    def test_service_added_event_serialization(self):
        """Test ServiceAddedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import ServiceAddedEvent, EventBase
        from dataclasses import asdict

        event = ServiceAddedEvent(
            experiment_id="myexp",
            service_id="svc123",
            run_id="run_001",
            description="Tensorboard service",
            service_class="experimaestro.services.TensorboardService",
            state_dict={"log_dir": "/tmp/logs"},
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "ServiceAddedEvent", **event_dict}
        )

        assert isinstance(restored, ServiceAddedEvent)
        assert restored.experiment_id == "myexp"
        assert restored.service_id == "svc123"
        assert restored.description == "Tensorboard service"
        assert restored.state_dict == {"log_dir": "/tmp/logs"}

    def test_service_state_changed_event_serialization(self):
        """Test ServiceStateChangedEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import (
            ServiceStateChangedEvent,
            EventBase,
        )
        from dataclasses import asdict

        event = ServiceStateChangedEvent(
            experiment_id="myexp",
            service_id="svc123",
            run_id="run_001",
            state="RUNNING",
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "ServiceStateChangedEvent", **event_dict}
        )

        assert isinstance(restored, ServiceStateChangedEvent)
        assert restored.service_id == "svc123"
        assert restored.state == "RUNNING"

    def test_carbon_metrics_event_serialization(self):
        """Test CarbonMetricsEvent serialization through SSH pipeline"""
        from experimaestro.scheduler.state_status import CarbonMetricsEvent, EventBase
        from dataclasses import asdict

        event = CarbonMetricsEvent(
            job_id="job123",
            co2_kg=0.05,
            energy_kwh=0.1,
            cpu_power_w=50.0,
            gpu_power_w=150.0,
            ram_power_w=10.0,
            duration_s=3600.0,
            region="FR",
            is_final=True,
            written=True,
            timestamp=1704067260.0,
        )

        # Server side: serialize
        event_dict = asdict(event)

        # Client side: deserialize
        restored = EventBase.from_dict(
            {"event_type": "CarbonMetricsEvent", **event_dict}
        )

        assert isinstance(restored, CarbonMetricsEvent)
        assert restored.job_id == "job123"
        assert restored.co2_kg == 0.05
        assert restored.is_final is True


class TestServerEventNotification:
    """Test server event notification mechanism"""

    @pytest.fixture
    def mock_state_provider(self):
        """Create a mock state provider"""
        provider = MagicMock()
        provider.workspace_path = Path("/tmp/workspace")
        return provider

    def test_server_on_state_event_job_state_changed(
        self, mock_state_provider, tmp_path
    ):
        """Test server converts JobStateChangedEvent to notification"""
        from experimaestro.scheduler.remote.server import SSHStateProviderServer
        from experimaestro.scheduler.state_status import JobStateChangedEvent
        import io

        # Create server with mock stdout
        stdout = io.BytesIO()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".experimaestro").mkdir()

        server = SSHStateProviderServer(workspace, stdout=stdout)
        server._state_provider = mock_state_provider

        # Create event
        event = JobStateChangedEvent(
            job_id="job123",
            state="running",
            from_experiment=True,
            timestamp=1704067260.0,
        )

        # Call the event handler
        server._on_state_event(event)

        # Parse the notification
        stdout.seek(0)
        notification_json = stdout.read().decode("utf-8").strip()
        notification_data = json.loads(notification_json)

        assert notification_data["method"] == "notification.state_event"
        assert notification_data["params"]["event_type"] == "JobStateChangedEvent"
        assert notification_data["params"]["data"]["job_id"] == "job123"
        assert notification_data["params"]["data"]["state"] == "running"
        # The from_experiment field should be in the notification
        assert notification_data["params"]["data"]["from_experiment"] is True

    def test_server_on_state_event_job_progress(self, mock_state_provider, tmp_path):
        """Test server converts JobProgressEvent to notification"""
        from experimaestro.scheduler.remote.server import SSHStateProviderServer
        from experimaestro.scheduler.state_status import JobProgressEvent
        import io

        stdout = io.BytesIO()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".experimaestro").mkdir()

        server = SSHStateProviderServer(workspace, stdout=stdout)
        server._state_provider = mock_state_provider

        event = JobProgressEvent(
            job_id="job123",
            level=0,
            progress=0.5,
            desc="halfway",
            timestamp=1704067260.0,
        )

        server._on_state_event(event)

        stdout.seek(0)
        notification_json = stdout.read().decode("utf-8").strip()
        notification_data = json.loads(notification_json)

        assert notification_data["params"]["event_type"] == "JobProgressEvent"
        assert notification_data["params"]["data"]["progress"] == 0.5


class TestClientEventHandling:
    """Test client event handling and conversion"""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a client instance for testing"""
        from experimaestro.scheduler.remote.client import SSHStateProviderClient

        client = SSHStateProviderClient(
            host="testhost",
            remote_workspace="/remote/workspace",
        )
        # Manually set up temp directory for testing
        client._temp_dir = str(tmp_path)
        client.local_cache_dir = tmp_path
        client.workspace_path = tmp_path
        return client

    def test_client_notification_to_event_job_state_changed(self, client):
        """Test client converts notification to JobStateChangedEvent"""
        from experimaestro.scheduler.state_status import JobStateChangedEvent
        from experimaestro.scheduler.remote.protocol import NotificationMethod

        event = client._notification_to_event(
            NotificationMethod.STATE_EVENT.value,
            {
                "event_type": "JobStateChangedEvent",
                "data": {
                    "job_id": "job123",
                    "state": "running",
                    "from_experiment": True,
                    "timestamp": 1704067260.0,
                },
            },
        )

        assert isinstance(event, JobStateChangedEvent)
        assert event.job_id == "job123"
        assert event.state == "running"
        assert event.from_experiment is True

    def test_client_notification_to_event_job_progress(self, client):
        """Test client converts notification to JobProgressEvent"""
        from experimaestro.scheduler.state_status import JobProgressEvent
        from experimaestro.scheduler.remote.protocol import NotificationMethod

        event = client._notification_to_event(
            NotificationMethod.STATE_EVENT.value,
            {
                "event_type": "JobProgressEvent",
                "data": {
                    "job_id": "job123",
                    "level": 0,
                    "progress": 0.75,
                    "desc": "Processing",
                    "timestamp": 1704067260.0,
                },
            },
        )

        assert isinstance(event, JobProgressEvent)
        assert event.progress == 0.75

    def test_client_notification_to_event_experiment_updated(self, client):
        """Test client converts notification to ExperimentUpdatedEvent"""
        from experimaestro.scheduler.state_status import ExperimentUpdatedEvent
        from experimaestro.scheduler.remote.protocol import NotificationMethod

        event = client._notification_to_event(
            NotificationMethod.STATE_EVENT.value,
            {
                "event_type": "ExperimentUpdatedEvent",
                "data": {
                    "experiment_id": "myexp",
                    "timestamp": 1704067260.0,
                },
            },
        )

        assert isinstance(event, ExperimentUpdatedEvent)
        assert event.experiment_id == "myexp"

    def test_client_notification_to_event_service_added(self, client):
        """Test client converts notification to ServiceAddedEvent"""
        from experimaestro.scheduler.state_status import ServiceAddedEvent
        from experimaestro.scheduler.remote.protocol import NotificationMethod

        event = client._notification_to_event(
            NotificationMethod.STATE_EVENT.value,
            {
                "event_type": "ServiceAddedEvent",
                "data": {
                    "experiment_id": "myexp",
                    "service_id": "svc123",
                    "run_id": "run_001",
                    "description": "Test service",
                    "service_class": "mymodule.MyService",
                    "state_dict": {"port": 8080},
                    "timestamp": 1704067260.0,
                },
            },
        )

        assert isinstance(event, ServiceAddedEvent)
        assert event.service_id == "svc123"
        assert event.state_dict == {"port": 8080}

    def test_client_notification_unknown_event_type(self, client):
        """Test client handles unknown event type gracefully"""
        from experimaestro.scheduler.remote.protocol import NotificationMethod

        event = client._notification_to_event(
            NotificationMethod.STATE_EVENT.value,
            {
                "event_type": "UnknownEventType",
                "data": {"foo": "bar"},
            },
        )

        # Should return None for unknown event types
        assert event is None

    def test_client_notification_non_state_event(self, client):
        """Test client handles non-state notifications"""
        from experimaestro.scheduler.remote.protocol import NotificationMethod

        # SHUTDOWN notifications are handled separately
        event = client._notification_to_event(
            NotificationMethod.SHUTDOWN.value,
            {"reason": "test"},
        )

        assert event is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestWorkspaceToServerEventFlow:
    """Integration tests for event flow from WorkspaceStateProvider to SSH Server

    These tests verify the full event flow:
    1. Event file is written by experiment/job
    2. WorkspaceStateProvider detects the event
    3. WorkspaceStateProvider notifies listeners (including SSH server)
    4. SSH server serializes and sends notification to client
    """

    def test_listener_receives_job_state_changed_event(self, tmp_path):
        """Test that listener receives JobStateChangedEvent when written to event file"""
        import time
        from experimaestro.scheduler.workspace_state_provider import (
            WorkspaceStateProvider,
        )
        from experimaestro.scheduler.state_status import (
            JobStateChangedEvent,
            ExperimentEventWriter,
        )
        from experimaestro.scheduler.interfaces import ExperimentStatus

        # Create workspace structure
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".events" / "experiments").mkdir(parents=True)
        (workspace / "experiments" / "test_exp" / "run_001").mkdir(parents=True)

        # Create a mock experiment
        exp = MockExperiment(
            workdir=workspace / "experiments" / "test_exp" / "run_001",
            run_id="run_001",
            status=ExperimentStatus.RUNNING,
        )

        # Write status.json
        status_path = workspace / "experiments" / "test_exp" / "run_001" / "status.json"
        status_path.write_text(json.dumps(exp.state_dict()))

        # Create event writer for experiment
        event_writer = ExperimentEventWriter(
            experiment=exp,
            workspace_path=workspace,
        )

        # Create symlink for "current" run
        events_dir = workspace / ".events" / "experiments" / "test_exp"
        events_dir.mkdir(parents=True, exist_ok=True)
        (events_dir / "current").symlink_to(
            workspace / "experiments" / "test_exp" / "run_001"
        )

        # Create provider and register listener
        received_events = []

        def listener(event):
            received_events.append(event)

        # Use a new provider instance (not singleton) for testing
        provider = WorkspaceStateProvider(workspace)
        provider.add_listener(listener)

        # Write an event
        event = JobStateChangedEvent(
            job_id="job123",
            state="running",
            timestamp=time.time(),
        )
        event_writer.write_event(event)
        event_writer.flush()

        # Give the watcher time to detect and process the event
        time.sleep(0.5)

        # Clean up
        provider.close()
        event_writer.close()

        # Verify event was received
        assert len(received_events) >= 1
        # Find our event (there might be others)
        job_events = [
            e
            for e in received_events
            if isinstance(e, JobStateChangedEvent) and e.job_id == "job123"
        ]
        assert len(job_events) >= 1
        assert job_events[0].state == "running"
        # from_experiment should be True because it came from experiment events
        assert job_events[0].from_experiment is True

    def test_listener_receives_job_progress_event(self, tmp_path):
        """Test that listener receives JobProgressEvent when written to event file"""
        import time
        from experimaestro.scheduler.workspace_state_provider import (
            WorkspaceStateProvider,
        )
        from experimaestro.scheduler.state_status import (
            JobProgressEvent,
            ExperimentEventWriter,
        )
        from experimaestro.scheduler.interfaces import ExperimentStatus

        # Create workspace structure
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".events" / "experiments").mkdir(parents=True)
        (workspace / "experiments" / "test_exp" / "run_001").mkdir(parents=True)

        # Create a mock experiment
        exp = MockExperiment(
            workdir=workspace / "experiments" / "test_exp" / "run_001",
            run_id="run_001",
            status=ExperimentStatus.RUNNING,
        )

        # Write status.json
        status_path = workspace / "experiments" / "test_exp" / "run_001" / "status.json"
        status_path.write_text(json.dumps(exp.state_dict()))

        # Create event writer
        event_writer = ExperimentEventWriter(
            experiment=exp,
            workspace_path=workspace,
        )

        # Create symlink for "current" run
        events_dir = workspace / ".events" / "experiments" / "test_exp"
        events_dir.mkdir(parents=True, exist_ok=True)
        (events_dir / "current").symlink_to(
            workspace / "experiments" / "test_exp" / "run_001"
        )

        # Create provider and register listener
        received_events = []

        def listener(event):
            received_events.append(event)

        provider = WorkspaceStateProvider(workspace)
        provider.add_listener(listener)

        # Write a progress event
        event = JobProgressEvent(
            job_id="job456",
            level=0,
            progress=0.5,
            desc="halfway",
            timestamp=time.time(),
        )
        event_writer.write_event(event)
        event_writer.flush()

        # Give the watcher time to detect and process the event
        time.sleep(0.5)

        # Clean up
        provider.close()
        event_writer.close()

        # Verify event was received
        progress_events = [
            e for e in received_events if isinstance(e, JobProgressEvent)
        ]
        assert len(progress_events) >= 1
        assert any(e.job_id == "job456" and e.progress == 0.5 for e in progress_events)


class TestErrorHandling:
    """Test error handling in protocol and server"""

    def test_rpc_error_creation(self):
        """Test creating RPC error objects"""
        error = RPCError(
            code=-32600, message="Invalid request", data={"field": "method"}
        )

        assert error.code == -32600
        assert error.message == "Invalid request"
        assert error.data == {"field": "method"}

        error_dict = error.to_dict()
        assert error_dict["code"] == -32600
        assert error_dict["message"] == "Invalid request"
        assert error_dict["data"] == {"field": "method"}

    def test_rpc_error_from_dict(self):
        """Test creating RPC error from dictionary"""
        error = RPCError.from_dict(
            {
                "code": -32601,
                "message": "Method not found",
            }
        )

        assert error.code == -32601
        assert error.message == "Method not found"
        assert error.data is None

    def test_response_with_error(self):
        """Test response with error is parsed correctly"""
        resp = RPCResponse(
            id=1,
            error=RPCError(code=-32600, message="Invalid request"),
        )

        resp_dict = resp.to_dict()
        assert "error" in resp_dict
        assert resp_dict["error"]["code"] == -32600
        assert "result" not in resp_dict
