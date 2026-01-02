"""JSON-RPC 2.0 protocol utilities for SSH-based remote monitoring

This module provides JSON-RPC message types, serialization utilities,
and protocol constants for communication between SSHStateProviderServer
and SSHStateProviderClient.

Message format: Newline-delimited JSON (one JSON object per line)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import JobState

logger = logging.getLogger("xpm.remote.protocol")

# JSON-RPC 2.0 version
JSONRPC_VERSION = "2.0"

# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Custom error codes
CONNECTION_ERROR = -32001
WORKSPACE_NOT_FOUND = -32002
PERMISSION_DENIED = -32003
TIMEOUT_ERROR = -32004


class NotificationMethod(str, Enum):
    """Server-to-client notification methods"""

    EXPERIMENT_UPDATED = "notification.experiment_updated"
    RUN_UPDATED = "notification.run_updated"
    JOB_UPDATED = "notification.job_updated"
    SERVICE_UPDATED = "notification.service_updated"
    FILE_CHANGED = "notification.file_changed"
    SHUTDOWN = "notification.shutdown"


class RPCMethod(str, Enum):
    """Client-to-server RPC methods"""

    GET_EXPERIMENTS = "get_experiments"
    GET_EXPERIMENT = "get_experiment"
    GET_EXPERIMENT_RUNS = "get_experiment_runs"
    GET_JOBS = "get_jobs"
    GET_JOB = "get_job"
    GET_ALL_JOBS = "get_all_jobs"
    GET_SERVICES = "get_services"
    KILL_JOB = "kill_job"
    CLEAN_JOB = "clean_job"
    GET_SYNC_INFO = "get_sync_info"


@dataclass
class RPCError:
    """JSON-RPC error object"""

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict:
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "RPCError":
        return cls(code=d["code"], message=d["message"], data=d.get("data"))


@dataclass
class RPCRequest:
    """JSON-RPC request message"""

    method: str
    params: Dict = field(default_factory=dict)
    id: Optional[int] = None  # None for notifications

    def to_dict(self) -> Dict:
        result = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict) -> "RPCRequest":
        return cls(method=d["method"], params=d.get("params", {}), id=d.get("id"))


@dataclass
class RPCResponse:
    """JSON-RPC response message"""

    id: int
    result: Optional[Any] = None
    error: Optional[RPCError] = None

    def to_dict(self) -> Dict:
        d = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict) -> "RPCResponse":
        error = None
        if "error" in d:
            error = RPCError.from_dict(d["error"])
        return cls(id=d["id"], result=d.get("result"), error=error)


@dataclass
class RPCNotification:
    """JSON-RPC notification (no id, no response expected)"""

    method: str
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict) -> "RPCNotification":
        return cls(method=d["method"], params=d.get("params", {}))


def parse_message(line: str) -> Union[RPCRequest, RPCResponse, RPCNotification]:
    """Parse a JSON-RPC message from a line of text

    Args:
        line: A single line of JSON text

    Returns:
        RPCRequest, RPCResponse, or RPCNotification

    Raises:
        ValueError: If the message is malformed
    """
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if "jsonrpc" not in d or d["jsonrpc"] != JSONRPC_VERSION:
        raise ValueError("Invalid or missing jsonrpc version")

    # Response: has "id" and either "result" or "error"
    if "result" in d or "error" in d:
        return RPCResponse.from_dict(d)

    # Request or notification: has "method"
    if "method" in d:
        if "id" in d:
            return RPCRequest.from_dict(d)
        else:
            return RPCNotification.from_dict(d)

    raise ValueError("Cannot determine message type")


def create_error_response(id: int, code: int, message: str, data: Any = None) -> str:
    """Create a JSON-RPC error response

    Args:
        id: Request ID
        code: Error code
        message: Error message
        data: Optional additional data

    Returns:
        JSON string
    """
    response = RPCResponse(id=id, error=RPCError(code=code, message=message, data=data))
    return response.to_json()


def create_success_response(id: int, result: Any) -> str:
    """Create a JSON-RPC success response

    Args:
        id: Request ID
        result: Result data

    Returns:
        JSON string
    """
    response = RPCResponse(id=id, result=result)
    return response.to_json()


def create_notification(method: Union[str, NotificationMethod], params: Dict) -> str:
    """Create a JSON-RPC notification

    Args:
        method: Notification method name
        params: Notification parameters

    Returns:
        JSON string
    """
    if isinstance(method, NotificationMethod):
        method = method.value
    notification = RPCNotification(method=method, params=params)
    return notification.to_json()


def create_request(method: Union[str, RPCMethod], params: Dict, id: int) -> str:
    """Create a JSON-RPC request

    Args:
        method: Method name
        params: Request parameters
        id: Request ID

    Returns:
        JSON string
    """
    if isinstance(method, RPCMethod):
        method = method.value
    request = RPCRequest(method=method, params=params, id=id)
    return request.to_json()


# Serialization helpers for data types


def serialize_datetime(dt) -> Optional[str]:
    """Serialize datetime or timestamp to ISO format string

    Handles:
    - None: returns None
    - datetime: returns ISO format string
    - float/int: treats as Unix timestamp, converts to ISO format
    - str: returns as-is (already serialized)
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt  # Already serialized
    if isinstance(dt, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(dt).isoformat()
    if isinstance(dt, datetime):
        return dt.isoformat()
    # Try to convert to string as fallback
    return str(dt)


def deserialize_datetime(s: Optional[str]) -> Optional[datetime]:
    """Deserialize ISO format string to datetime"""
    if s is None:
        return None
    return datetime.fromisoformat(s)


def serialize_job(job) -> Dict:
    """Serialize a job (MockJob or Job) to a dictionary for JSON-RPC"""
    from experimaestro.scheduler.interfaces import JobState

    result = {
        "identifier": job.identifier,
        "task_id": job.task_id,
        "locator": job.locator,
        "path": str(job.path) if job.path else None,
        "state": job.state.name if isinstance(job.state, JobState) else str(job.state),
        "submittime": serialize_datetime(job.submittime),
        "starttime": serialize_datetime(job.starttime),
        "endtime": serialize_datetime(job.endtime),
        "progress": job.progress,
        "tags": job.tags,
        "experiment_id": getattr(job, "experiment_id", None),
        "run_id": getattr(job, "run_id", None),
    }
    return result


def deserialize_job(d: Dict) -> "MockJobData":
    """Deserialize a dictionary to MockJobData"""
    from experimaestro.scheduler.interfaces import JobState, STATE_NAME_TO_JOBSTATE
    from pathlib import Path

    state = STATE_NAME_TO_JOBSTATE.get(d["state"], JobState.WAITING)
    return MockJobData(
        identifier=d["identifier"],
        task_id=d["task_id"],
        locator=d["locator"],
        path=Path(d["path"]) if d["path"] else None,
        state=state,
        submittime=deserialize_datetime(d.get("submittime")),
        starttime=deserialize_datetime(d.get("starttime")),
        endtime=deserialize_datetime(d.get("endtime")),
        progress=d.get("progress"),
        tags=d.get("tags", {}),
        experiment_id=d.get("experiment_id"),
        run_id=d.get("run_id"),
    )


def serialize_experiment(experiment) -> Dict:
    """Serialize a MockExperiment to a dictionary for JSON-RPC"""
    result = {
        "experiment_id": experiment.experiment_id,
        "workdir": str(experiment.workdir) if experiment.workdir else None,
        "current_run_id": experiment.current_run_id,
        "total_jobs": experiment.total_jobs,
        "finished_jobs": experiment.finished_jobs,
        "failed_jobs": experiment.failed_jobs,
        "updated_at": serialize_datetime(experiment.updated_at),
        "started_at": serialize_datetime(experiment.started_at),
        "ended_at": serialize_datetime(experiment.ended_at),
        "hostname": experiment.hostname,
    }
    return result


def deserialize_experiment(d: Dict) -> "MockExperimentData":
    """Deserialize a dictionary to MockExperimentData"""
    from pathlib import Path

    return MockExperimentData(
        experiment_id=d["experiment_id"],
        workdir=Path(d["workdir"]) if d["workdir"] else None,
        current_run_id=d.get("current_run_id"),
        total_jobs=d.get("total_jobs", 0),
        finished_jobs=d.get("finished_jobs", 0),
        failed_jobs=d.get("failed_jobs", 0),
        updated_at=deserialize_datetime(d.get("updated_at")),
        started_at=deserialize_datetime(d.get("started_at")),
        ended_at=deserialize_datetime(d.get("ended_at")),
        hostname=d.get("hostname"),
    )


def serialize_service(service) -> Dict:
    """Serialize a service to a dictionary for JSON-RPC"""
    from experimaestro.scheduler.services import Service

    # Service has id attribute, description() method, state property, state_dict() method
    state = service.state
    if hasattr(state, "name"):
        state = state.name  # Convert ServiceState enum to string
    elif hasattr(state, "value"):
        state = state.value

    # Get URL if service has it (e.g., TensorboardService)
    url = None
    if hasattr(service, "url"):
        url = service.url
    elif hasattr(service, "get_url"):
        try:
            url = service.get_url()
        except Exception:
            pass

    # Get state_dict with __class__ and serialize paths
    if hasattr(service, "_full_state_dict"):
        state_dict = Service.serialize_state_dict(service._full_state_dict())
    elif callable(getattr(service, "state_dict", None)):
        # Fallback: serialize paths in the raw state_dict
        state_dict = Service.serialize_state_dict(service.state_dict())
    else:
        state_dict = getattr(service, "state_dict", {})

    return {
        "service_id": getattr(service, "id", None),
        "description": (
            service.description()
            if callable(service.description)
            else service.description
        ),
        "state": state,
        "state_dict": state_dict,
        "experiment_id": getattr(service, "experiment_id", None),
        "run_id": getattr(service, "run_id", None),
        "url": url,
    }


def serialize_run(run) -> Dict:
    """Serialize an experiment run to a dictionary for JSON-RPC

    Handles both dictionary and object inputs (get_experiment_runs returns dicts).
    """
    if isinstance(run, dict):
        # Already a dictionary - just ensure datetime serialization
        return {
            "run_id": run.get("run_id"),
            "experiment_id": run.get("experiment_id"),
            "hostname": run.get("hostname"),
            "started_at": run.get("started_at"),  # Already serialized
            "ended_at": run.get("ended_at"),  # Already serialized
            "status": run.get("status"),
        }
    else:
        # Object with attributes
        return {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "hostname": getattr(run, "hostname", None),
            "started_at": serialize_datetime(run.started_at),
            "ended_at": serialize_datetime(run.ended_at),
            "status": run.status,
        }


@dataclass
class MockJobData:
    """Deserialized job data from remote"""

    identifier: str
    task_id: str
    locator: str
    path: Optional["Path"]
    state: "JobState"
    submittime: Optional[datetime]
    starttime: Optional[datetime]
    endtime: Optional[datetime]
    progress: Optional[float]
    tags: Dict
    experiment_id: Optional[str]
    run_id: Optional[str]


@dataclass
class MockExperimentData:
    """Deserialized experiment data from remote"""

    experiment_id: str
    workdir: Optional["Path"]
    current_run_id: Optional[str]
    total_jobs: int
    finished_jobs: int
    failed_jobs: int
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    hostname: Optional[str]
