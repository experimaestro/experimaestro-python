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
    from experimaestro.scheduler.remote.adaptive_sync import AdaptiveSynchronizer
    from experimaestro.scheduler.state_status import WarningEvent


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


class SSHLocalService(BaseService):
    """Local service wrapper that manages sync lifecycle for SSH remote monitoring.

    Wraps a live Service instance and manages file synchronization:
    - get_url(): starts adaptive sync for service paths, then starts the inner service
    - stop(): stops the inner service, then stops the sync

    Supports an optional callback for sync status changes to enable dynamic UI updates.
    """

    def __init__(
        self,
        inner_service: "BaseService",
        state_provider: "SSHStateProviderClient",
        remote_paths: List[str],
        on_status_change: Optional[Callable[[], None]] = None,
    ):
        """Initialize SSH local service wrapper.

        Args:
            inner_service: The actual live Service instance
            state_provider: SSH state provider for sync operations
            remote_paths: List of remote paths to sync for this service
            on_status_change: Optional callback invoked when sync status changes
        """
        self._inner = inner_service
        self._state_provider = state_provider
        self._remote_paths = remote_paths
        self._synchronizers: List["AdaptiveSynchronizer"] = []
        self._on_status_change = on_status_change

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def experiment_id(self) -> str:
        return self._inner.experiment_id

    @property
    def run_id(self) -> str:
        return self._inner.run_id

    @property
    def state(self):
        return self._inner.state

    @property
    def url(self) -> Optional[str]:
        return getattr(self._inner, "url", None)

    def description(self) -> str:
        return self._inner.description()

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def _notify_status_change(self) -> None:
        """Notify that sync status has changed."""
        if self._on_status_change:
            try:
                self._on_status_change()
            except Exception as e:
                logger.debug(f"Error in status change callback: {e}")

    def get_url(self) -> str:
        """Start adaptive sync for service paths, then start the inner service."""
        from experimaestro.scheduler.remote.adaptive_sync import AdaptiveSynchronizer

        # Do initial sync and start adaptive synchronizers for each path
        for remote_path in self._remote_paths:
            # Initial sync
            self._state_provider.sync_path(remote_path)

            # Start adaptive sync for continuous updates
            sync = AdaptiveSynchronizer(
                sync_func=self._state_provider.sync_path,
                remote_path=remote_path,
                name=f"service:{self._inner.id}",
                on_sync_start=lambda: self._notify_status_change(),
                on_sync_complete=lambda _: self._notify_status_change(),
            )
            sync.start()
            self._synchronizers.append(sync)

        # Notify initial status
        self._notify_status_change()

        # Start the inner service
        return self._inner.get_url()

    @property
    def sync_status(self) -> Optional[str]:
        """Return sync status for display."""
        if not self._synchronizers:
            return None

        # Check if any sync is actively syncing
        syncing = any(s.syncing for s in self._synchronizers)
        if syncing:
            return "⟳ Syncing"

        # Return interval of first synchronizer
        if self._synchronizers:
            interval = self._synchronizers[0].interval
            return f"✓ in {interval:.0f}s"

        return None

    @property
    def error(self) -> Optional[str]:
        """Return error message if service failed to start."""
        return self._inner.error

    def stop(self) -> None:
        """Stop the inner service and stop all syncs."""
        # Stop the inner service first
        if hasattr(self._inner, "stop"):
            self._inner.stop()

        # Stop all synchronizers
        for sync in self._synchronizers:
            sync.stop()
        self._synchronizers.clear()


class SSHMockService(MockService):
    """MockService specialized for SSH remote monitoring.

    Extends MockService to:
    - Store reference to SSH state provider
    - Create SSHLocalService via to_service() with sync management
    - Handle path translation for remote paths
    """

    def __init__(
        self,
        service_id: str,
        description_text: str,
        state_dict_data: dict,
        state_provider: "SSHStateProviderClient",
        service_class: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        url: Optional[str] = None,
        state: Optional[str] = None,
    ):
        super().__init__(
            service_id=service_id,
            description_text=description_text,
            state_dict_data=state_dict_data,
            service_class=service_class,
            experiment_id=experiment_id,
            run_id=run_id,
            url=url,
            state=state,
        )
        self._state_provider = state_provider
        self._on_status_change: Optional[Callable[[], None]] = None

    def set_status_change_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        """Set callback to be invoked when sync status changes.

        This should be called before to_service() to enable dynamic UI updates.
        If to_service() was already called, updates the existing SSHLocalService.
        """
        self._on_status_change = callback
        # Update existing live service if already created
        if self._live_service is not None and hasattr(
            self._live_service, "_on_status_change"
        ):
            self._live_service._on_status_change = callback

    def _extract_paths(self) -> List[str]:
        """Extract path strings from state_dict for syncing."""
        paths = []

        def find_paths(d):
            if isinstance(d, dict):
                if "__path__" in d:
                    paths.append(d["__path__"])
                else:
                    for v in d.values():
                        find_paths(v)
            elif isinstance(d, (list, tuple)):
                for item in d:
                    find_paths(item)

        find_paths(self._state_dict_data)
        return paths

    def to_service(self) -> "BaseService":
        """Create an SSHLocalService that manages sync lifecycle.

        Returns:
            SSHLocalService wrapping the live service, or self if creation fails
        """
        # Return cached live service if available
        if self._live_service is not None:
            return self._live_service

        from experimaestro.scheduler.services import Service

        if not self._service_class:
            return self

        # Create path translator using state provider
        def path_translator(remote_path: str) -> Path:
            """Translate remote path to local, syncing if needed."""
            local_path = self._state_provider.sync_path(remote_path)
            if local_path:
                return local_path
            # Fallback: map to local cache without sync
            remote_workspace = self._state_provider.remote_workspace
            local_cache = self._state_provider.local_cache_dir
            if remote_path.startswith(remote_workspace):
                relative = remote_path[len(remote_workspace) :].lstrip("/")
                return local_cache / relative
            return Path(remote_path)

        try:
            # Create the actual service with path translation
            inner_service = Service.from_state_dict(
                self._service_class,
                self._state_dict_data,
                path_translator,
            )
            inner_service.id = self.id

            # Extract remote paths for sync management
            remote_paths = self._extract_paths()

            # Wrap in SSHLocalService for sync lifecycle management
            self._live_service = SSHLocalService(
                inner_service=inner_service,
                state_provider=self._state_provider,
                remote_paths=remote_paths,
                on_status_change=self._on_status_change,
            )
            return self._live_service
        except Exception as e:
            logger.warning("Failed to create live service for %s: %s", self.id, e)
            self.set_error(str(e))
            return self


class SSHStateProviderClient(OfflineStateProvider):
    """Client that connects to SSHStateProviderServer via SSH

    This client implements the StateProvider interface for remote experiment
    monitoring via SSH.

    Features:
    - JSON-RPC over SSH stdin/stdout
    - Async request/response handling with futures
    - Server push notifications converted to EventBases
    - On-demand rsync for specific paths (used by services like TensorboardService)
    - Job caching with event-driven updates (similar to WorkspaceStateProvider)
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

        # Job and experiment caches are inherited from OfflineStateProvider

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
        # Use shlex.split to properly handle complex commands with arguments
        import shlex

        if self.remote_xpm_path:
            # Use specified path/command to experimaestro
            # Parse the remote_xpm_path in case it contains arguments
            remote_cmd_parts = shlex.split(self.remote_xpm_path)
        else:
            # Use uv tool run with version pinning
            try:
                xpm_version = get_package_version("experimaestro")
                # Strip .devN suffix for release compatibility
                xpm_version = _strip_dev_version(xpm_version)
            except Exception:
                xpm_version = None

            if xpm_version:
                remote_cmd_parts = [
                    "uv",
                    "tool",
                    "run",
                    f"experimaestro=={xpm_version}",
                ]
            else:
                remote_cmd_parts = ["uv", "tool", "run", "experimaestro"]

        # Add the subcommand arguments
        remote_cmd_parts.extend(
            ["experiments", "--workdir", self.remote_workspace, "monitor-server"]
        )
        cmd.extend(remote_cmd_parts)

        logger.info("Connecting to %s, workspace: %s", self.host, self.remote_workspace)
        logger.debug("SSH command: %s", shlex.join(cmd))

        # Start SSH process
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

        # Clear job and experiment caches (using inherited methods)
        self._clear_job_cache()
        self._clear_experiment_cache_all()

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
        Also applies events to cached jobs/experiments for state consistency.
        """
        method = notification.method
        params = notification.params

        logger.debug("Received notification: %s", method)

        # Convert notification to EventBase and queue for throttled delivery
        event = self._notification_to_event(method, params)
        if event:
            # Apply event to cached objects (like WorkspaceStateProvider does)
            self._apply_event_to_cache(event)

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
        return [self._get_or_load_experiment(d) for d in result]

    def get_experiment(self, experiment_id: str) -> Optional[BaseExperiment]:
        """Get a specific experiment by ID"""
        params = {"experiment_id": experiment_id}
        result = self._call_sync(RPCMethod.GET_EXPERIMENT, params)
        if result is None:
            return None
        return self._get_or_load_experiment(result)

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
        return [self._get_or_load_job(d) for d in result]

    def get_job(self, task_id: str, job_id: str) -> Optional[BaseJob]:
        """Get a specific job"""
        params = {
            "task_id": task_id,
            "job_id": job_id,
        }
        result = self._call_sync(RPCMethod.GET_JOB, params)
        if result is None:
            return None
        return self._get_or_load_job(result)

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
        return [self._get_or_load_job(d) for d in result]

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
        """Kill a running job.

        Raises:
            RuntimeError: If the job cannot be killed.
        """
        if not perform:
            # Dry run - just check if job is running
            return job.state.running()

        params = {
            "job_id": job.identifier,
            "experiment_id": getattr(job, "experiment_id", ""),
            "run_id": getattr(job, "run_id", ""),
        }
        result = self._call_sync(RPCMethod.KILL_JOB, params)
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Failed to kill job: {error}")
        return True

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

    def delete_job_safely(self, job: BaseJob, perform: bool = True) -> tuple[bool, str]:
        """Safely delete a job and its data via remote server"""
        if not perform:
            # Dry run - check if job can be deleted (not running)
            if job.state and job.state.running():
                return False, f"Cannot delete running job {job.identifier}"
            return True, f"Job {job.identifier} can be deleted"

        params = {
            "job_id": job.identifier,
            "experiment_id": getattr(job, "experiment_id", ""),
            "run_id": getattr(job, "run_id", ""),
            "perform": True,
        }
        result = self._call_sync(RPCMethod.DELETE_JOB_SAFELY, params)
        return result.get("success", False), result.get("message", "")

    def delete_experiment(
        self, experiment_id: str, delete_jobs: bool = False, perform: bool = True
    ) -> tuple[bool, str]:
        """Delete an experiment and optionally its job data via remote server"""
        params = {
            "experiment_id": experiment_id,
            "delete_jobs": delete_jobs,
            "perform": perform,
        }
        result = self._call_sync(RPCMethod.DELETE_EXPERIMENT, params)
        return result.get("success", False), result.get("message", "")

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

    def execute_warning_action(
        self,
        warning_key: str,
        action_key: str,
        experiment_id: str = "",
        run_id: str = "",
    ) -> None:
        """Execute a warning action on the remote server

        Args:
            warning_key: The warning identifier
            action_key: The action to execute
            experiment_id: Experiment ID for error events
            run_id: Run ID for error events

        Raises:
            RuntimeError: If the action execution fails on remote server
        """
        params = {
            "warning_key": warning_key,
            "action_key": action_key,
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
        self._call_sync(RPCMethod.EXECUTE_WARNING_ACTION, params)

    def get_unresolved_warnings(self) -> List["WarningEvent"]:
        """Get all unresolved warnings from the remote server

        Returns:
            List of WarningEvent objects with metadata for all pending warnings
        """
        from experimaestro.scheduler.state_status import WarningEvent

        result = self._call_sync(RPCMethod.GET_UNRESOLVED_WARNINGS, {})
        warnings = []
        for w in result.get("warnings", []):
            warnings.append(
                WarningEvent(
                    experiment_id=w.get("experiment_id", ""),
                    run_id=w.get("run_id", ""),
                    warning_key=w.get("warning_key", ""),
                    description=w.get("description", ""),
                    actions=w.get("actions", {}),
                    context=w.get("context", {}),
                    severity=w.get("severity", "warning"),
                )
            )
        return warnings

    # -------------------------------------------------------------------------
    # Data Conversion
    # -------------------------------------------------------------------------

    def _get_or_load_job(self, d: Dict) -> MockJob:
        """Get or load job from dict (uses base class caching)"""
        job_id = d.get("identifier", "")
        task_id = d.get("task_id", "")
        full_id = BaseJob.make_full_id(task_id, job_id) if task_id and job_id else ""

        if not full_id:
            # Can't cache without identifiers, just create
            return self._create_job_from_dict(d)

        return super()._get_or_load_job(full_id, d=d)

    def _create_job(self, full_id: str, *, d: Dict) -> MockJob:
        """Create job from server dict"""
        return self._create_job_from_dict(d)

    def _on_cached_job_found(self, job: MockJob, *, d: Dict) -> None:
        """Update cached job with fresh data from server"""
        self._update_job_from_dict(job, d)

    def _create_job_from_dict(self, d: Dict) -> MockJob:
        """Internal: create MockJob from dict with path translation"""
        # Translate remote path to local cache path
        if d.get("path"):
            remote_path = d["path"]
            if remote_path.startswith(self.remote_workspace):
                relative = remote_path[len(self.remote_workspace) :].lstrip("/")
                d["path"] = self.local_cache_dir / relative
            else:
                d["path"] = Path(remote_path)

        return MockJob.from_state_dict(d, self.local_cache_dir)

    def _update_job_from_dict(self, job: MockJob, d: Dict) -> None:
        """Update a cached MockJob with fresh data from server

        This updates the job's state from server data while preserving
        any event-driven state updates that may have arrived since.
        """
        from experimaestro.scheduler.interfaces import (
            STATE_NAME_TO_JOBSTATE,
            deserialize_to_datetime,
        )

        # Only update state from server if we don't have event-driven state
        # (event state takes priority over server snapshot)
        if not job._has_event_state:
            state_str = d.get("state", "unscheduled")
            job._state = STATE_NAME_TO_JOBSTATE.get(state_str, job._state)

        # Update timestamps (these don't conflict with events)
        if d.get("submittime"):
            job.submittime = deserialize_to_datetime(d["submittime"])
        if d.get("starttime"):
            job.starttime = deserialize_to_datetime(d["starttime"])
        if d.get("endtime"):
            job.endtime = deserialize_to_datetime(d["endtime"])

        # Update progress if provided
        if d.get("progress"):
            from experimaestro.notifications import get_progress_information_from_dict

            job.progress = [
                get_progress_information_from_dict(p) for p in d["progress"]
            ]

    def _get_or_load_experiment(self, d: Dict) -> MockExperiment:
        """Get or load experiment from dict (uses base class caching)"""
        experiment_id = d.get("experiment_id", "")
        run_id = d.get("run_id", "")

        if not experiment_id or not run_id:
            # Can't cache without identifiers, just create
            return self._create_experiment_from_dict(d)

        return super()._get_or_load_experiment(experiment_id, run_id, d=d)

    def _create_experiment(
        self, experiment_id: str, run_id: str, *, d: Dict
    ) -> MockExperiment:
        """Create experiment from server dict"""
        return self._create_experiment_from_dict(d)

    def _on_cached_experiment_found(self, exp: MockExperiment, *, d: Dict) -> None:
        """Update cached experiment with fresh data from server"""
        self._update_experiment_from_dict(exp, d)

    def _create_experiment_from_dict(self, d: Dict) -> MockExperiment:
        """Internal: create MockExperiment from dict with path translation"""
        # Translate remote workdir to local cache path
        if d.get("workdir"):
            remote_path = d["workdir"]
            if remote_path.startswith(self.remote_workspace):
                relative = remote_path[len(self.remote_workspace) :].lstrip("/")
                d["workdir"] = self.local_cache_dir / relative
            else:
                d["workdir"] = Path(remote_path)

        return MockExperiment.from_state_dict(d, self.local_cache_dir)

    def _update_experiment_from_dict(self, exp: MockExperiment, d: Dict) -> None:
        """Update a cached MockExperiment with fresh data from server"""
        from experimaestro.scheduler.interfaces import (
            ExperimentStatus,
            deserialize_to_datetime,
        )

        # Update status
        if d.get("status"):
            try:
                exp._status = ExperimentStatus(d["status"])
            except ValueError:
                pass

        # Update counts
        if "total_jobs" in d:
            exp._total_jobs = d["total_jobs"]
        if "finished_jobs" in d:
            exp._finished_jobs = d["finished_jobs"]
        if "failed_jobs" in d:
            exp._failed_jobs = d["failed_jobs"]

        # Update timestamps
        if d.get("started_at"):
            exp._started_at = deserialize_to_datetime(d["started_at"])
        if d.get("ended_at"):
            exp._ended_at = deserialize_to_datetime(d["ended_at"])

        # Update job_infos (for tracking job states)
        if d.get("job_infos"):
            from experimaestro.scheduler.interfaces import ExperimentJobInformation

            for job_id, info_dict in d["job_infos"].items():
                if job_id not in exp.job_infos:
                    exp.job_infos[job_id] = ExperimentJobInformation(
                        job_id=job_id,
                        task_id=info_dict.get("task_id", ""),
                        tags=info_dict.get("tags", {}),
                        timestamp=info_dict.get("timestamp"),
                    )

    def _dict_to_service(self, d: Dict) -> SSHMockService:
        """Convert a dictionary to an SSHMockService

        Returns SSHMockService which handles:
        - Path translation via state provider's sync_path
        - Sync lifecycle management when service starts/stops
        - Lazy live service creation via to_service()
        """
        service_id = d.get("service_id", "")
        state_dict = d.get("state_dict", {})
        service_class = d.get("class", "")

        # Check for unserializable marker
        description = d.get("description", "")
        if state_dict.get("__unserializable__"):
            reason = state_dict.get("__reason__", "Service cannot be recreated")
            description = f"[{reason}]"

        return SSHMockService(
            service_id=service_id,
            description_text=description,
            state_dict_data=state_dict,
            state_provider=self,
            service_class=service_class,
            experiment_id=d.get("experiment_id"),
            run_id=d.get("run_id"),
            url=d.get("url"),
            state=d.get("state"),
        )

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

    def sync_path(self, path: str, include: list[str] | None = None) -> Optional[Path]:
        """Sync a specific path from remote on-demand

        Used by services (e.g., TensorboardService) that need access to
        specific remote directories.

        Args:
            path: Can be:
                - Remote absolute path (e.g., /remote/workspace/jobs/xxx)
                - Local cache path (e.g., /tmp/xpm_remote_xxx/jobs/xxx)
                - Relative path within workspace (e.g., jobs/xxx)
            include: Optional list of filename patterns to include (e.g., ["*.out", "*.err"]).
                    If provided, only files matching these patterns will be synced.

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
            return self._synchronizer.sync_path(remote_path, include=include)
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
