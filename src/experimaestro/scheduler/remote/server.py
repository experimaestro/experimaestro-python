"""SSH State Provider Server

JSON-RPC server that wraps WorkspaceStateProvider and communicates via stdio.
Designed to be run over SSH for remote experiment monitoring.

Usage:
    experimaestro experiments monitor-server --workdir /path/to/workspace
"""

import logging
import sys
import threading
from pathlib import Path
from typing import IO, Callable, Dict, Optional

from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from dataclasses import asdict

from experimaestro.scheduler.state_status import EventBase
from experimaestro.scheduler.remote.protocol import (
    RPCMethod,
    NotificationMethod,
    parse_message,
    create_success_response,
    create_error_response,
    create_notification,
    serialize_datetime,
    deserialize_datetime,
    PARSE_ERROR,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    WORKSPACE_NOT_FOUND,
)

logger = logging.getLogger("xpm.remote.server")


class SSHStateProviderServer:
    """JSON-RPC server that wraps WorkspaceStateProvider for SSH-based monitoring

    This server reads JSON-RPC requests from stdin and writes responses to stdout.
    It registers as a listener with the WorkspaceStateProvider to push notifications
    when state changes occur.

    Thread safety:
    - Writes to stdout are serialized with a lock
    - The main read loop runs in the calling thread
    - Event notifications may come from the state provider's change detector thread
    """

    def __init__(
        self,
        workspace_path: Path,
        stdin: IO[bytes] = None,
        stdout: IO[bytes] = None,
    ):
        """Initialize the server

        Args:
            workspace_path: Path to the workspace directory
            stdin: Input stream for reading requests (default: sys.stdin.buffer)
            stdout: Output stream for writing responses (default: sys.stdout.buffer)
        """
        self.workspace_path = workspace_path
        self.stdin = stdin if stdin is not None else sys.stdin.buffer
        self.stdout = stdout if stdout is not None else sys.stdout.buffer
        self._state_provider: Optional[WorkspaceStateProvider] = None
        self._running = False
        self._write_lock = threading.Lock()

        # Map of method names to handler functions
        self._handlers: Dict[str, Callable] = {
            RPCMethod.GET_EXPERIMENTS.value: self._handle_get_experiments,
            RPCMethod.GET_EXPERIMENT.value: self._handle_get_experiment,
            RPCMethod.GET_EXPERIMENT_RUNS.value: self._handle_get_experiment_runs,
            RPCMethod.GET_JOBS.value: self._handle_get_jobs,
            RPCMethod.GET_JOB.value: self._handle_get_job,
            RPCMethod.GET_ALL_JOBS.value: self._handle_get_all_jobs,
            RPCMethod.GET_SERVICES.value: self._handle_get_services,
            RPCMethod.GET_TAGS_MAP.value: self._handle_get_tags_map,
            RPCMethod.GET_DEPENDENCIES_MAP.value: self._handle_get_dependencies_map,
            RPCMethod.KILL_JOB.value: self._handle_kill_job,
            RPCMethod.CLEAN_JOB.value: self._handle_clean_job,
            RPCMethod.GET_SYNC_INFO.value: self._handle_get_sync_info,
            RPCMethod.GET_PROCESS_INFO.value: self._handle_get_process_info,
        }

    def start(self):
        """Start the server and begin processing requests

        This method blocks until the server is stopped or stdin is closed.
        """
        # Verify workspace exists
        if not self.workspace_path.exists():
            logger.error("Workspace path does not exist: %s", self.workspace_path)
            self._send_error_and_exit(
                WORKSPACE_NOT_FOUND,
                f"Workspace path does not exist: {self.workspace_path}",
            )
            return

        # Initialize state provider in read-only mode with event watcher
        try:
            self._state_provider = WorkspaceStateProvider.get_instance(
                self.workspace_path
            )
        except Exception as e:
            logger.exception("Failed to initialize state provider")
            self._send_error_and_exit(INTERNAL_ERROR, f"Failed to initialize: {e}")
            return

        # Register as listener for state changes
        self._state_provider.add_listener(self._on_state_event)

        self._running = True
        logger.info("SSH State Provider Server started for %s", self.workspace_path)

        try:
            self._read_loop()
        finally:
            self.stop()

    def stop(self):
        """Stop the server and clean up resources"""
        self._running = False

        # Unregister listener
        if self._state_provider is not None:
            try:
                self._state_provider.remove_listener(self._on_state_event)
            except Exception:
                pass

        # Send shutdown notification
        try:
            self._send_notification(
                NotificationMethod.SHUTDOWN, {"reason": "server_shutdown"}
            )
        except Exception:
            pass

        logger.info("SSH State Provider Server stopped")

    def _read_loop(self):
        """Main loop: read JSON-RPC requests from stdin and process them"""
        while self._running:
            try:
                line = self.stdin.readline()
                if not line:
                    # EOF - stdin closed
                    logger.debug("stdin closed, stopping server")
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                self._process_request(line_str)

            except Exception as e:
                logger.exception("Error in read loop: %s", e)
                # Continue processing - don't crash on individual request errors

    def _process_request(self, line: str):
        """Process a single JSON-RPC request"""
        try:
            msg = parse_message(line)
        except ValueError as e:
            self._send_response(create_error_response(0, PARSE_ERROR, str(e)))
            return

        # We only handle requests (with id), not responses or notifications
        from experimaestro.scheduler.remote.protocol import RPCRequest

        if not isinstance(msg, RPCRequest):
            logger.warning("Received non-request message: %s", type(msg).__name__)
            return

        request = msg
        method = request.method
        params = request.params
        request_id = request.id

        if request_id is None:
            # Notification from client - we don't handle these currently
            logger.debug("Received notification: %s", method)
            return

        # Dispatch to handler
        handler = self._handlers.get(method)
        if handler is None:
            self._send_response(
                create_error_response(
                    request_id, METHOD_NOT_FOUND, f"Unknown method: {method}"
                )
            )
            return

        try:
            result = handler(params)
            self._send_response(create_success_response(request_id, result))
        except TypeError as e:
            self._send_response(
                create_error_response(request_id, INVALID_PARAMS, str(e))
            )
        except Exception as e:
            logger.exception("Error handling %s", method)
            self._send_response(
                create_error_response(request_id, INTERNAL_ERROR, str(e))
            )

    def _send_response(self, response: str):
        """Send a JSON-RPC response (thread-safe)"""
        with self._write_lock:
            self.stdout.write((response + "\n").encode("utf-8"))
            self.stdout.flush()

    def _send_notification(self, method: NotificationMethod, params: Dict):
        """Send a JSON-RPC notification (thread-safe)"""
        notification = create_notification(method, params)
        self._send_response(notification)

    def _send_error_and_exit(self, code: int, message: str):
        """Send an error notification and exit"""
        self._send_notification(
            NotificationMethod.SHUTDOWN,
            {"reason": "error", "code": code, "message": message},
        )

    def _on_state_event(self, event: EventBase):
        """Handle state change events from the state provider

        Converts events to JSON-RPC notifications and sends them to the client.
        Uses generic serialization via dataclasses.asdict.
        """
        try:
            # Serialize event to dict, filtering out None values and non-serializable objects
            event_dict = {}
            for key, value in asdict(event).items():
                # Skip None values and complex objects (like job references)
                if value is not None and not isinstance(value, (Path,)):
                    # Try to serialize - skip if not JSON-serializable
                    try:
                        import json

                        json.dumps(value)
                        event_dict[key] = value
                    except (TypeError, ValueError):
                        pass

            self._send_notification(
                NotificationMethod.STATE_EVENT,
                {
                    "event_type": type(event).__name__,
                    "data": event_dict,
                },
            )
        except Exception as e:
            logger.exception("Error sending notification: %s", e)

    # -------------------------------------------------------------------------
    # Request Handlers
    # -------------------------------------------------------------------------

    def _handle_get_experiments(self, params: Dict) -> list:
        """Handle get_experiments request"""
        since = deserialize_datetime(params.get("since"))
        experiments = self._state_provider.get_experiments(since=since)
        return [exp.state_dict() for exp in experiments]

    def _handle_get_experiment(self, params: Dict) -> Optional[Dict]:
        """Handle get_experiment request"""
        experiment_id = params.get("experiment_id")
        if not experiment_id:
            raise TypeError("experiment_id is required")

        experiment = self._state_provider.get_experiment(experiment_id)
        if experiment is None:
            return None
        return experiment.state_dict()

    def _handle_get_experiment_runs(self, params: Dict) -> list:
        """Handle get_experiment_runs request"""
        experiment_id = params.get("experiment_id")
        if not experiment_id:
            raise TypeError("experiment_id is required")

        runs = self._state_provider.get_experiment_runs(experiment_id)
        return [run.state_dict() for run in runs]

    def _handle_get_jobs(self, params: Dict) -> list:
        """Handle get_jobs request"""
        since = deserialize_datetime(params.get("since"))
        jobs = self._state_provider.get_jobs(
            experiment_id=params.get("experiment_id"),
            run_id=params.get("run_id"),
            task_id=params.get("task_id"),
            state=params.get("state"),
            tags=params.get("tags"),
            since=since,
        )
        return [job.state_dict() for job in jobs]

    def _handle_get_job(self, params: Dict) -> Optional[Dict]:
        """Handle get_job request"""
        job_id = params.get("job_id")
        experiment_id = params.get("experiment_id")
        if not job_id or not experiment_id:
            raise TypeError("job_id and experiment_id are required")

        job = self._state_provider.get_job(
            job_id=job_id,
            experiment_id=experiment_id,
            run_id=params.get("run_id"),
        )
        if job is None:
            return None
        return job.state_dict()

    def _handle_get_all_jobs(self, params: Dict) -> list:
        """Handle get_all_jobs request"""
        since = deserialize_datetime(params.get("since"))
        jobs = self._state_provider.get_all_jobs(
            state=params.get("state"),
            tags=params.get("tags"),
            since=since,
        )
        return [job.state_dict() for job in jobs]

    def _handle_get_services(self, params: Dict) -> list:
        """Handle get_services request

        Returns serialized service data using full_state_dict().
        """
        services = self._state_provider.get_services(
            experiment_id=params.get("experiment_id"),
            run_id=params.get("run_id"),
        )
        return [svc.full_state_dict() for svc in services]

    def _handle_get_tags_map(self, params: Dict) -> Dict[str, Dict[str, str]]:
        """Handle get_tags_map request

        Returns tags map for jobs in an experiment/run.
        """
        experiment_id = params.get("experiment_id")
        if not experiment_id:
            raise TypeError("experiment_id is required")

        return self._state_provider.get_tags_map(
            experiment_id=experiment_id,
            run_id=params.get("run_id"),
        )

    def _handle_get_dependencies_map(self, params: Dict) -> dict[str, list[str]]:
        """Handle get_dependencies_map request

        Returns dependencies map for jobs in an experiment/run.
        """
        experiment_id = params.get("experiment_id")
        if not experiment_id:
            raise TypeError("experiment_id is required")

        return self._state_provider.get_dependencies_map(
            experiment_id=experiment_id,
            run_id=params.get("run_id"),
        )

    def _handle_kill_job(self, params: Dict) -> Dict:
        """Handle kill_job request"""
        job_id = params.get("job_id")
        experiment_id = params.get("experiment_id")
        run_id = params.get("run_id")

        if not job_id or not experiment_id or not run_id:
            raise TypeError("job_id, experiment_id, and run_id are required")

        # Get the job first
        job = self._state_provider.get_job(job_id, experiment_id, run_id)
        if job is None:
            return {"success": False, "error": "Job not found"}

        # Kill the job
        try:
            result = self._state_provider.kill_job(job, perform=True)
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_clean_job(self, params: Dict) -> Dict:
        """Handle clean_job request"""
        job_id = params.get("job_id")
        experiment_id = params.get("experiment_id")
        run_id = params.get("run_id")

        if not job_id or not experiment_id or not run_id:
            raise TypeError("job_id, experiment_id, and run_id are required")

        # Get the job first
        job = self._state_provider.get_job(job_id, experiment_id, run_id)
        if job is None:
            return {"success": False, "error": "Job not found"}

        # Clean the job
        try:
            result = self._state_provider.clean_job(job, perform=True)
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_get_sync_info(self, params: Dict) -> Dict:
        """Handle get_sync_info request"""
        return {
            "workspace_path": str(self.workspace_path),
            "last_sync_time": (
                serialize_datetime(self._state_provider.get_last_sync_time())
                if hasattr(self._state_provider, "get_last_sync_time")
                else None
            ),
        }

    def _handle_get_process_info(self, params: Dict) -> Optional[Dict]:
        """Handle get_process_info request"""
        job_id = params.get("job_id")
        experiment_id = params.get("experiment_id")
        run_id = params.get("run_id")

        if not job_id or not experiment_id:
            raise TypeError("job_id and experiment_id are required")

        # Get the job first
        job = self._state_provider.get_job(job_id, experiment_id, run_id)
        if job is None:
            return None

        # Get process info
        pinfo = self._state_provider.get_process_info(job)
        if pinfo is None:
            return None

        # Serialize ProcessInfo to dict
        return {
            "pid": pinfo.pid,
            "type": pinfo.type,
            "running": pinfo.running,
        }
