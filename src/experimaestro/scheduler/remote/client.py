"""SSH State Provider Client

Client that connects via SSH to a remote SSHStateProviderServer and implements
the StateProvider-like interface for local TUI/web UI usage.

Usage:
    client = SSHStateProviderClient(host="server", remote_workspace="/path/to/workspace")
    client.connect()
    experiments = client.get_experiments()
"""

import atexit
import logging
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from termcolor import colored

from experimaestro.scheduler.state_provider import (
    OfflineStateProvider,
    StateListener,
    MockJob,
    MockExperiment,
    MockService,
)
from experimaestro.scheduler.state_status import EventBase
from experimaestro.scheduler.interfaces import (
    BaseJob,
    BaseExperiment,
    BaseService,
)
from experimaestro.scheduler.remote.protocol import (
    RPCMethod,
    NotificationMethod,
    RPCResponse,
    RPCNotification,
    parse_message,
    create_request,
    serialize_datetime,
)

# Type for SSH output callback
OutputCallback = Optional["Callable[[str], None]"]

if TYPE_CHECKING:
    from experimaestro.scheduler.remote.sync import RemoteFileSynchronizer


logger = logging.getLogger("xpm.remote.client")

# Default timeout for RPC requests (seconds)
DEFAULT_TIMEOUT = 30.0


def _strip_dev_version(version: str) -> str:
    """Strip the .devN suffix from a version string.

    Examples:
        '2.0.0b3.dev8' -> '2.0.0b3'
        '1.2.3.dev1' -> '1.2.3'
        '1.2.3' -> '1.2.3' (unchanged)
    """
    import re

    return re.sub(r"\.dev\d+$", "", version)


class SSHStateProviderClient(OfflineStateProvider):
    """Client that connects to SSHStateProviderServer via SSH

    This client implements the StateProvider interface for remote experiment
    monitoring via SSH.

    Features:
    - JSON-RPC over SSH stdin/stdout
    - Async request/response handling with futures
    - Server push notifications converted to EventBases
    - On-demand rsync for specific paths (used by services like TensorboardService)
    """

    def __init__(
        self,
        host: str,
        remote_workspace: str,
        ssh_options: Optional[List[str]] = None,
        remote_xpm_path: Optional[str] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the client

        Args:
            host: SSH host (user@host or host)
            remote_workspace: Path to workspace on the remote host
            ssh_options: Additional SSH options (e.g., ["-p", "2222"])
            remote_xpm_path: Path to experimaestro executable on remote host.
                If None, uses 'uv tool run experimaestro==<version>'.
            output_callback: Callback for SSH process output (stderr).
                If None, a default callback prints with colored prefix.
                Set to False (or a no-op lambda) to disable output display.
        """
        # Initialize base class (includes service cache)
        super().__init__()

        self.host = host
        self.remote_workspace = remote_workspace
        self.ssh_options = ssh_options or []
        self.remote_xpm_path = remote_xpm_path
        self._output_callback = output_callback

        # Session-specific temporary cache directory (created on connect)
        self._temp_dir: Optional[str] = None
        self.local_cache_dir: Optional[Path] = None
        self.workspace_path: Optional[Path] = None  # For compatibility

        self._process: Optional[subprocess.Popen] = None
        self._stdin = None
        self._stdout = None
        self._stderr = None

        self._listeners: Set[StateListener] = set()
        self._listener_lock = threading.Lock()

        self._response_handlers: Dict[int, Future] = {}
        self._response_lock = threading.Lock()
        self._request_id = 0

        self._read_thread: Optional[threading.Thread] = None
        self._notify_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        self._synchronizer: Optional["RemoteFileSynchronizer"] = None

        # Throttled notification delivery to avoid flooding UI
        self._pending_events: List[EventBase] = []
        self._pending_events_lock = threading.Lock()
        self._notify_interval = 2.0  # Seconds between notification batches

    def connect(self, timeout: float = 30.0):
        """Establish SSH connection and start remote server

        Args:
            timeout: Connection timeout in seconds
        """
        if self._connected:
            logger.warning("Already connected")
            return

        # Create session-specific temporary cache directory
        self._temp_dir = tempfile.mkdtemp(prefix="xpm_remote_")
        self.local_cache_dir = Path(self._temp_dir)
        self.workspace_path = self.local_cache_dir
        logger.debug("Created temporary cache directory: %s", self._temp_dir)

        # Register cleanup on exit (in case disconnect isn't called)
        atexit.register(self._cleanup_temp_dir)

        # Build SSH command
        cmd = ["ssh"]
        cmd.extend(self.ssh_options)
        cmd.append(self.host)

        # Build remote command (workdir is passed to experiments group)
        if self.remote_xpm_path:
            # Use specified path to experimaestro
            remote_cmd = f"{self.remote_xpm_path} experiments --workdir {self.remote_workspace} monitor-server"
        else:
            # Use uv tool run with version pinning
            try:
                xpm_version = get_package_version("experimaestro")
                # Strip .devN suffix for release compatibility
                xpm_version = _strip_dev_version(xpm_version)
            except Exception:
                xpm_version = None

            if xpm_version:
                remote_cmd = f"uv tool run experimaestro=={xpm_version} experiments --workdir {self.remote_workspace} monitor-server"
            else:
                remote_cmd = f"uv tool run experimaestro experiments --workdir {self.remote_workspace} monitor-server"
        cmd.append(remote_cmd)

        logger.info("Connecting to %s, workspace: %s", self.host, self.remote_workspace)
        logger.debug("SSH command: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
            )
            self._stdin = self._process.stdin
            self._stdout = self._process.stdout
            self._stderr = self._process.stderr
        except Exception as e:
            logger.error("Failed to start SSH process: %s", e)
            raise ConnectionError(f"Failed to connect to {self.host}: {e}")

        self._running = True

        # Start read thread for responses and notifications
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="SSHClient-Read"
        )
        self._read_thread.start()

        # Start notification thread for throttled event delivery
        self._notify_thread = threading.Thread(
            target=self._notify_loop, daemon=True, name="SSHClient-Notify"
        )
        self._notify_thread.start()

        # Start stderr thread to display SSH output
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, daemon=True, name="SSHClient-Stderr"
        )
        self._stderr_thread.start()

        # Wait for connection to be established by sending a test request
        try:
            sync_info = self._call_sync(RPCMethod.GET_SYNC_INFO, {}, timeout=timeout)
            logger.info(
                "Connected to remote workspace: %s", sync_info.get("workspace_path")
            )
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to establish connection: {e}")

        self._connected = True

    def disconnect(self):
        """Disconnect from the remote server"""
        self._running = False
        self._connected = False

        # Close stdin to signal EOF to remote server
        if self._stdin:
            try:
                self._stdin.close()
            except Exception:
                pass

        # Terminate the SSH process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass

        # Wait for threads to finish
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
        if self._notify_thread and self._notify_thread.is_alive():
            self._notify_thread.join(timeout=2.0)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2.0)

        # Cancel any pending requests
        with self._response_lock:
            for future in self._response_handlers.values():
                if not future.done():
                    future.set_exception(ConnectionError("Disconnected"))
            self._response_handlers.clear()

        # Clear service cache (using base class method)
        self._clear_service_cache()

        # Clean up temporary cache directory
        self._cleanup_temp_dir()

        logger.info("Disconnected from %s", self.host)

    def _cleanup_temp_dir(self):
        """Clean up the temporary cache directory"""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug("Cleaned up temporary cache directory: %s", self._temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up temp dir %s: %s", self._temp_dir, e)
            finally:
                self._temp_dir = None
                self.local_cache_dir = None
                # Unregister atexit handler if we cleaned up successfully
                try:
                    atexit.unregister(self._cleanup_temp_dir)
                except Exception:
                    pass

    def close(self):
        """Alias for disconnect() for compatibility with WorkspaceStateProvider"""
        self.disconnect()

    def _read_loop(self):
        """Read responses and notifications from SSH stdout"""
        while self._running:
            try:
                line = self._stdout.readline()
                if not line:
                    # EOF - connection closed
                    logger.debug("SSH stdout closed")
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                self._process_message(line_str)

            except Exception as e:
                if self._running:
                    logger.exception("Error in read loop: %s", e)
                break

        # Connection lost
        if self._running:
            logger.warning("Connection to %s lost", self.host)
            self._connected = False

    def _stderr_loop(self):
        """Read and display SSH stderr output with colored prefix"""
        while self._running:
            try:
                line = self._stderr.readline()
                if not line:
                    # EOF - stderr closed
                    logger.debug("SSH stderr closed")
                    break

                line_str = line.decode("utf-8").rstrip("\n\r")
                if not line_str:
                    continue

                # Call output callback or use default
                if self._output_callback is not None:
                    self._output_callback(line_str)
                else:
                    # Default: print with colored prefix
                    prefix = colored("[SSH] ", "cyan", attrs=["bold"])
                    print(f"{prefix}{line_str}")  # noqa: T201

            except Exception as e:
                if self._running:
                    logger.debug("Error reading stderr: %s", e)
                break

    def _process_message(self, line: str):
        """Process a single message from the server"""
        try:
            msg = parse_message(line)
        except ValueError as e:
            logger.warning("Failed to parse message: %s", e)
            return

        if isinstance(msg, RPCResponse):
            self._handle_response(msg)
        elif isinstance(msg, RPCNotification):
            self._handle_notification(msg)
        else:
            logger.debug("Unexpected message type: %s", type(msg).__name__)

    def _handle_response(self, response: RPCResponse):
        """Handle a response from the server"""
        with self._response_lock:
            future = self._response_handlers.pop(response.id, None)

        if future is None:
            logger.warning("Received response for unknown request ID: %s", response.id)
            return

        if response.error:
            future.set_exception(
                RuntimeError(
                    f"RPC error {response.error.code}: {response.error.message}"
                )
            )
        else:
            future.set_result(response.result)

    def _handle_notification(self, notification: RPCNotification):
        """Handle a notification from the server

        Queues events for throttled delivery to avoid flooding the UI.
        """
        method = notification.method
        params = notification.params

        logger.debug("Received notification: %s", method)

        # Convert notification to EventBase and queue for throttled delivery
        event = self._notification_to_event(method, params)
        if event:
            with self._pending_events_lock:
                self._pending_events.append(event)

        # Handle shutdown notification immediately
        if method == NotificationMethod.SHUTDOWN.value:
            reason = params.get("reason", "unknown")
            logger.info("Server shutdown: %s", reason)
            self._connected = False

    def _notify_loop(self):
        """Background thread that delivers pending events to listeners periodically

        This throttles notification delivery to avoid flooding the UI with
        rapid state changes.
        """
        import time

        while self._running:
            time.sleep(self._notify_interval)

            if not self._running:
                break

            # Get and clear pending events atomically
            with self._pending_events_lock:
                if not self._pending_events:
                    continue
                events = self._pending_events.copy()
                self._pending_events.clear()

            # Deduplicate events by type (keep latest of each type)
            # This prevents redundant refreshes for rapidly changing state
            seen_types = set()
            unique_events = []
            for event in reversed(events):
                event_type = type(event)
                if event_type not in seen_types:
                    seen_types.add(event_type)
                    unique_events.append(event)
            unique_events.reverse()

            # Notify listeners
            for event in unique_events:
                self._notify_listeners(event)

    def _notification_to_event(self, method: str, params: Dict) -> Optional[EventBase]:
        """Convert a notification to a EventBase"""
        if method != NotificationMethod.STATE_EVENT.value:
            # Don't warn for known control notifications (handled elsewhere)
            if method not in (
                NotificationMethod.SHUTDOWN.value,
                NotificationMethod.FILE_CHANGED.value,
            ):
                logger.warning("Unhandled notification method: %s", method)
            return None

        event_type = params.get("event_type")
        data = params.get("data", {})
        event_class = EventBase.get_class(event_type)
        if event_class is None:
            logger.warning("Unknown event type: %s", event_type)
            return None

        try:
            return event_class(**data)
        except TypeError as e:
            logger.warning("Error deserializing event %s: %s", event_type, e)
            return None

    def _notify_listeners(self, event: EventBase):
        """Notify all registered listeners of a state event"""
        with self._listener_lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.exception("Error in listener: %s", e)

    def _call(self, method: RPCMethod, params: Dict) -> Future:
        """Send an RPC request and return a Future for the response

        Args:
            method: RPC method to call
            params: Method parameters

        Returns:
            Future that resolves to the response result
        """
        if not self._running:
            future = Future()
            future.set_exception(ConnectionError("Not connected"))
            return future

        with self._response_lock:
            self._request_id += 1
            request_id = self._request_id
            future = Future()
            self._response_handlers[request_id] = future

        request_json = create_request(method, params, request_id)
        try:
            self._stdin.write((request_json + "\n").encode("utf-8"))
            self._stdin.flush()
        except Exception as e:
            with self._response_lock:
                self._response_handlers.pop(request_id, None)
            future.set_exception(e)

        return future

    def _call_sync(
        self, method: RPCMethod, params: Dict, timeout: float = DEFAULT_TIMEOUT
    ):
        """Send an RPC request and wait for the response

        Args:
            method: RPC method to call
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            Response result

        Raises:
            TimeoutError: If the request times out
            RuntimeError: If the RPC call returns an error
        """
        future = self._call(method, params)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            raise TimeoutError(f"Request {method.value} timed out after {timeout}s")

    # -------------------------------------------------------------------------
    # StateProvider-like Interface
    # -------------------------------------------------------------------------

    def add_listener(self, listener: StateListener):
        """Register a listener for state change events"""
        with self._listener_lock:
            self._listeners.add(listener)

    def remove_listener(self, listener: StateListener):
        """Unregister a listener"""
        with self._listener_lock:
            self._listeners.discard(listener)

    def get_experiments(self, since: Optional[datetime] = None) -> List[BaseExperiment]:
        """Get list of all experiments"""
        params = {"since": serialize_datetime(since)}
        result = self._call_sync(RPCMethod.GET_EXPERIMENTS, params)
        return [self._dict_to_experiment(d) for d in result]

    def get_experiment(self, experiment_id: str) -> Optional[BaseExperiment]:
        """Get a specific experiment by ID"""
        params = {"experiment_id": experiment_id}
        result = self._call_sync(RPCMethod.GET_EXPERIMENT, params)
        if result is None:
            return None
        return self._dict_to_experiment(result)

    def get_experiment_runs(self, experiment_id: str) -> List[Dict]:
        """Get all runs for an experiment"""
        params = {"experiment_id": experiment_id}
        return self._call_sync(RPCMethod.GET_EXPERIMENT_RUNS, params)

    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current run ID for an experiment"""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            return None
        return exp.run_id

    def get_jobs(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[BaseJob]:
        """Query jobs with optional filters"""
        params = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "task_id": task_id,
            "state": state,
            "tags": tags,
            "since": serialize_datetime(since),
        }
        result = self._call_sync(RPCMethod.GET_JOBS, params)
        return [self._dict_to_job(d) for d in result]

    def get_job(
        self, job_id: str, experiment_id: str, run_id: Optional[str] = None
    ) -> Optional[BaseJob]:
        """Get a specific job"""
        params = {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
        result = self._call_sync(RPCMethod.GET_JOB, params)
        if result is None:
            return None
        return self._dict_to_job(result)

    def get_all_jobs(
        self,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[BaseJob]:
        """Get all jobs across all experiments"""
        params = {
            "state": state,
            "tags": tags,
            "since": serialize_datetime(since),
        }
        result = self._call_sync(RPCMethod.GET_ALL_JOBS, params)
        return [self._dict_to_job(d) for d in result]

    def get_tags_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Get tags map for jobs in an experiment/run"""
        params = {
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
        result = self._call_sync(RPCMethod.GET_TAGS_MAP, params)
        return result or {}

    def get_dependencies_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """Get dependencies map for jobs in an experiment/run"""
        params = {
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
        result = self._call_sync(RPCMethod.GET_DEPENDENCIES_MAP, params)
        return result or {}

    def _fetch_services_from_storage(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> List[BaseService]:
        """Fetch services from remote server.

        Called by base class get_services when cache is empty.
        """
        params = {
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
        result = self._call_sync(RPCMethod.GET_SERVICES, params)

        services = []
        for d in result:
            service = self._dict_to_service(d)
            services.append(service)

        return services

    def kill_job(self, job: BaseJob, perform: bool = False) -> bool:
        """Kill a running job"""
        if not perform:
            # Dry run - just check if job is running
            return job.state.running()

        params = {
            "job_id": job.identifier,
            "experiment_id": getattr(job, "experiment_id", ""),
            "run_id": getattr(job, "run_id", ""),
        }
        result = self._call_sync(RPCMethod.KILL_JOB, params)
        return result.get("success", False)

    def clean_job(self, job: BaseJob, perform: bool = False) -> bool:
        """Clean a finished job"""
        if not perform:
            # Dry run - just check if job is finished
            return job.state.finished()

        params = {
            "job_id": job.identifier,
            "experiment_id": getattr(job, "experiment_id", ""),
            "run_id": getattr(job, "run_id", ""),
        }
        result = self._call_sync(RPCMethod.CLEAN_JOB, params)
        return result.get("success", False)

    def get_process_info(self, job: BaseJob):
        """Get process information for a job

        Returns None if the remote server doesn't support this method.
        """
        from experimaestro.scheduler.state_provider import ProcessInfo

        params = {
            "job_id": job.identifier,
            "experiment_id": getattr(job, "experiment_id", ""),
            "run_id": getattr(job, "run_id", ""),
        }

        try:
            result = self._call_sync(RPCMethod.GET_PROCESS_INFO, params)
        except RuntimeError:
            # Server doesn't support this method (older version)
            return None

        if result is None:
            return None

        return ProcessInfo(
            pid=result["pid"],
            type=result["type"],
            running=result.get("running", False),
        )

    # -------------------------------------------------------------------------
    # Data Conversion
    # -------------------------------------------------------------------------

    def _dict_to_job(self, d: Dict) -> MockJob:
        """Convert a dictionary to a MockJob using from_state_dict"""
        # Translate remote path to local cache path
        if d.get("path"):
            remote_path = d["path"]
            if remote_path.startswith(self.remote_workspace):
                relative = remote_path[len(self.remote_workspace) :].lstrip("/")
                d["path"] = self.local_cache_dir / relative
            else:
                d["path"] = Path(remote_path)

        # Note: timestamp conversion is handled by MockJob.from_state_dict
        return MockJob.from_state_dict(d, self.local_cache_dir)

    def _dict_to_experiment(self, d: Dict) -> MockExperiment:
        """Convert a dictionary to a MockExperiment using from_state_dict"""
        # Translate remote workdir to local cache path
        if d.get("workdir"):
            remote_path = d["workdir"]
            if remote_path.startswith(self.remote_workspace):
                relative = remote_path[len(self.remote_workspace) :].lstrip("/")
                d["workdir"] = self.local_cache_dir / relative
            else:
                d["workdir"] = Path(remote_path)

        # Note: timestamp conversion is handled by MockExperiment.from_state_dict
        return MockExperiment.from_state_dict(d, self.local_cache_dir)

    def _dict_to_service(self, d: Dict) -> BaseService:
        """Convert a dictionary to a Service or MockService

        Tries to recreate the actual Service from state_dict first.
        Falls back to MockService with error message if module is missing.
        """
        state_dict = d.get("state_dict", {})
        service_class = d.get("class", "")
        service_id = d.get("service_id", "")

        # Check for unserializable marker
        if state_dict.get("__unserializable__"):
            reason = state_dict.get("__reason__", "Service cannot be recreated")
            return MockService(
                service_id=service_id,
                description_text=f"[{reason}]",
                state_dict_data=state_dict,
                service_class=service_class,
                experiment_id=d.get("experiment_id"),
                run_id=d.get("run_id"),
                url=d.get("url"),
            )

        # Try to recreate actual Service from state_dict
        if service_class:
            try:
                from experimaestro.scheduler.services import Service

                # Create path translator that syncs and translates paths
                def path_translator(remote_path: str) -> Path:
                    """Translate remote path to local, syncing if needed"""
                    local_path = self.sync_path(remote_path)
                    if local_path:
                        return local_path
                    # Fallback: map to local cache without sync
                    if remote_path.startswith(self.remote_workspace):
                        relative = remote_path[len(self.remote_workspace) :].lstrip("/")
                        return self.local_cache_dir / relative
                    return Path(remote_path)

                service = Service.from_state_dict(
                    service_class, state_dict, path_translator
                )
                service.id = service_id
                # Copy additional attributes
                if d.get("experiment_id"):
                    service.experiment_id = d["experiment_id"]
                if d.get("run_id"):
                    service.run_id = d["run_id"]
                return service
            except ModuleNotFoundError as e:
                # Module not available locally - show error in description
                missing_module = str(e).replace("No module named ", "").strip("'\"")
                return MockService(
                    service_id=service_id,
                    description_text=f"[Missing module: {missing_module}]",
                    state_dict_data=state_dict,
                    service_class=service_class,
                    experiment_id=d.get("experiment_id"),
                    run_id=d.get("run_id"),
                    url=d.get("url"),
                )
            except Exception as e:
                # Other error - show in description
                return MockService(
                    service_id=service_id,
                    description_text=f"[Error: {e}]",
                    state_dict_data=state_dict,
                    service_class=service_class,
                    experiment_id=d.get("experiment_id"),
                    run_id=d.get("run_id"),
                    url=d.get("url"),
                )

        # No class - use MockService.from_full_state_dict
        return MockService.from_full_state_dict(d)

    def _parse_datetime_to_timestamp(self, value) -> Optional[float]:
        """Convert datetime value to Unix timestamp

        Handles: None, ISO string, float timestamp, datetime object
        """
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

    # -------------------------------------------------------------------------
    # File Synchronization
    # -------------------------------------------------------------------------

    def sync_path(self, path: str) -> Optional[Path]:
        """Sync a specific path from remote on-demand

        Used by services (e.g., TensorboardService) that need access to
        specific remote directories.

        Args:
            path: Can be:
                - Remote absolute path (e.g., /remote/workspace/jobs/xxx)
                - Local cache path (e.g., /tmp/xpm_remote_xxx/jobs/xxx)
                - Relative path within workspace (e.g., jobs/xxx)

        Returns:
            Local path where the files were synced to, or None if sync failed
        """
        if not self._connected or not self.local_cache_dir:
            logger.warning("Cannot sync: not connected")
            return None

        # Convert local cache path back to remote path if needed
        local_cache_str = str(self.local_cache_dir)
        if path.startswith(local_cache_str):
            # Path is in local cache - extract relative path
            relative = path[len(local_cache_str) :].lstrip("/")
            remote_path = f"{self.remote_workspace}/{relative}"
        elif path.startswith(self.remote_workspace):
            # Already a remote path
            remote_path = path
        else:
            # Assume it's a relative path
            remote_path = f"{self.remote_workspace}/{path.lstrip('/')}"

        from experimaestro.scheduler.remote.sync import RemoteFileSynchronizer

        # Create synchronizer lazily
        if self._synchronizer is None:
            self._synchronizer = RemoteFileSynchronizer(
                host=self.host,
                remote_workspace=Path(self.remote_workspace),
                local_cache=self.local_cache_dir,
                ssh_options=self.ssh_options,
            )

        try:
            return self._synchronizer.sync_path(remote_path)
        except Exception as e:
            logger.warning("Failed to sync path %s: %s", remote_path, e)
            return None

    @property
    def read_only(self) -> bool:
        """Client is always read-only"""
        return True

    @property
    def is_remote(self) -> bool:
        """This is a remote provider"""
        return True
