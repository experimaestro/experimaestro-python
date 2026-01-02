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
    serialize_job,
    serialize_experiment,
    serialize_run,
)


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
            NotificationMethod.JOB_UPDATED, {"job_id": "job1", "state": "running"}
        )
        data = json.loads(notif_json)

        assert data["jsonrpc"] == JSONRPC_VERSION
        assert data["method"] == "notification.job_updated"
        assert data["params"] == {"job_id": "job1", "state": "running"}
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
    """Test job serialization"""

    def test_serialize_mock_job(self):
        """Test serializing a MockJob-like object"""
        from experimaestro.scheduler.state_provider import MockJob

        job = MockJob(
            identifier="job123",
            task_id="task.MyTask",
            locator="job123",
            path=Path("/tmp/jobs/job123"),
            state="running",
            submittime=1704067200.0,
            starttime=1704067300.0,
            endtime=None,
            progress=[],
            tags={"tag1": "value1"},
            experiment_id="exp1",
            run_id="run1",
            updated_at="2024-01-01T00:00:00",
        )

        result = serialize_job(job)

        assert result["identifier"] == "job123"
        assert result["task_id"] == "task.MyTask"
        assert result["path"] == "/tmp/jobs/job123"
        # State is serialized from JobState enum - case may vary
        assert result["state"].upper() == "RUNNING"
        assert result["tags"] == {"tag1": "value1"}
        assert result["experiment_id"] == "exp1"
        assert result["run_id"] == "run1"


class TestExperimentSerialization:
    """Test experiment serialization"""

    def test_serialize_mock_experiment(self):
        """Test serializing a MockExperiment-like object"""
        from experimaestro.scheduler.state_provider import MockExperiment

        exp = MockExperiment(
            workdir=Path("/tmp/xp/myexp"),
            current_run_id="run_20240101",
            total_jobs=10,
            finished_jobs=5,
            failed_jobs=1,
            updated_at="2024-01-01T12:00:00",
            started_at=1704067200.0,
            ended_at=None,
            hostname="server1",
        )

        result = serialize_experiment(exp)

        assert result["experiment_id"] == "myexp"
        assert result["workdir"] == "/tmp/xp/myexp"
        assert result["current_run_id"] == "run_20240101"
        assert result["total_jobs"] == 10
        assert result["finished_jobs"] == 5
        assert result["failed_jobs"] == 1
        assert result["hostname"] == "server1"


class TestRunSerialization:
    """Test run serialization"""

    def test_serialize_run_dict(self):
        """Test serializing a run dictionary"""
        run_dict = {
            "run_id": "run_20240101",
            "experiment_id": "exp1",
            "hostname": "server1",
            "started_at": "2024-01-01T10:00:00",
            "ended_at": None,
            "status": "active",
        }

        result = serialize_run(run_dict)

        assert result["run_id"] == "run_20240101"
        assert result["experiment_id"] == "exp1"
        assert result["hostname"] == "server1"
        assert result["status"] == "active"


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
        from experimaestro.scheduler.state_provider import MockExperiment

        mock_exp = MockExperiment(
            workdir=Path("/tmp/xp/exp1"),
            current_run_id="run1",
            total_jobs=5,
            finished_jobs=3,
            failed_jobs=0,
            updated_at="2024-01-01T00:00:00",
        )
        mock_state_provider.get_experiments.return_value = [mock_exp]

        result = server_with_mock._handle_get_experiments({"since": None})

        assert len(result) == 1
        assert result[0]["experiment_id"] == "exp1"
        mock_state_provider.get_experiments.assert_called_once_with(since=None)

    def test_handle_get_jobs(self, server_with_mock, mock_state_provider):
        """Test handling get_jobs request"""
        from experimaestro.scheduler.state_provider import MockJob

        mock_job = MockJob(
            identifier="job1",
            task_id="task.Test",
            locator="job1",
            path=Path("/tmp/jobs/job1"),
            state="done",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            tags={},
            experiment_id="exp1",
            run_id="run1",
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
        assert result[0]["identifier"] == "job1"

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

    def test_notification_handling(self):
        """Test notification message handling"""
        notification = create_notification(
            NotificationMethod.JOB_UPDATED,
            {
                "job_id": "job1",
                "experiment_id": "exp1",
                "run_id": "run1",
                "state": "running",
            },
        )

        msg = parse_message(notification)
        assert isinstance(msg, RPCNotification)
        assert msg.method == "notification.job_updated"
        assert msg.params["job_id"] == "job1"


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
        job_dict = {
            "identifier": "job123",
            "task_id": "task.MyTask",
            "locator": "job123",
            "path": "/remote/workspace/jobs/job123",
            "state": "running",
            "submittime": "2024-01-01T10:00:00",
            "starttime": "2024-01-01T10:01:00",
            "endtime": None,
            "progress": [],
            "tags": {"key": "value"},
            "experiment_id": "exp1",
            "run_id": "run1",
        }

        job = client._dict_to_job(job_dict)

        assert job.identifier == "job123"
        assert job.task_id == "task.MyTask"
        # Path should be mapped to local cache
        assert job.path == tmp_path / "jobs/job123"
        assert job.tags == {"key": "value"}

    def test_dict_to_experiment(self, client, tmp_path):
        """Test converting dictionary to MockExperiment"""
        exp_dict = {
            "experiment_id": "myexp",
            "workdir": "/remote/workspace/xp/myexp",
            "current_run_id": "run1",
            "total_jobs": 10,
            "finished_jobs": 5,
            "failed_jobs": 1,
            "updated_at": "2024-01-01T12:00:00",
            "hostname": "server1",
        }

        exp = client._dict_to_experiment(exp_dict)

        assert exp.experiment_id == "myexp"
        # Path should be mapped to local cache
        assert exp.workdir == tmp_path / "xp/myexp"
        assert exp.total_jobs == 10
        assert exp.hostname == "server1"

    def test_path_mapping_outside_workspace(self, client, tmp_path):
        """Test path mapping for paths outside remote workspace"""
        job_dict = {
            "identifier": "job123",
            "task_id": "task.MyTask",
            "locator": "job123",
            "path": "/other/path/job123",  # Not under remote_workspace
            "state": "done",
            "progress": [],
            "tags": {},
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
# Error Handling Tests
# =============================================================================


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
