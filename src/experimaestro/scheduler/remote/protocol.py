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
from typing import Any, Dict, Optional, Union

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

    # Generic state event notification (serialized dataclass)
    STATE_EVENT = "notification.state_event"

    # Control notifications
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
    GET_TAGS_MAP = "get_tags_map"
    GET_DEPENDENCIES_MAP = "get_dependencies_map"
    KILL_JOB = "kill_job"
    CLEAN_JOB = "clean_job"
    GET_SYNC_INFO = "get_sync_info"
    GET_PROCESS_INFO = "get_process_info"


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
