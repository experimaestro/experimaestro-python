"""WebSocket handler for real-time communication

Manages WebSocket connections and message routing.
Uses native WebSocket with JSON protocol.

Serialization is consistent with SSHStateProviderServer, using db_state_dict()
for Job/Experiment serialization, then transforming to frontend format.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.base import Scheduler, Job
from experimaestro.scheduler.jobs import JobDependency
from experimaestro.scheduler.interfaces import BaseJob, BaseExperiment, BaseService

logger = logging.getLogger("xpm.webui.websocket")


# =============================================================================
# Serialization helpers - Transform db_state_dict to frontend format
# =============================================================================


def job_db_to_frontend(db_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Transform job db_state_dict to frontend format

    db_state_dict format (snake_case):
        identifier, task_id, path, state, submittime, starttime, endtime,
        progress, exit_code, retry_count, failure_reason

    Frontend format (camelCase):
        jobId, taskId, locator, status, submitted, start, end,
        tags, progress, experimentIds, dependsOn
    """
    return {
        "jobId": db_dict.get("identifier"),
        "taskId": db_dict.get("task_id"),
        "locator": db_dict.get("path") or "",
        "status": (db_dict.get("state") or "unknown").lower(),
        "submitted": db_dict.get("submittime") or "",
        "start": db_dict.get("starttime") or "",
        "end": db_dict.get("endtime") or "",
        "tags": db_dict.get("tags", []),
        "progress": db_dict.get("progress", []),
        "experimentIds": db_dict.get("experiment_ids", []),
        "dependsOn": db_dict.get("depends_on", []),
    }


def experiment_db_to_frontend(db_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Transform experiment db_state_dict to frontend format"""
    return {
        "experiment_id": db_dict.get("experiment_id"),
        "workdir": db_dict.get("workdir"),
        "current_run_id": db_dict.get("current_run_id"),
        "total_jobs": db_dict.get("total_jobs", 0),
        "finished_jobs": db_dict.get("finished_jobs", 0),
        "failed_jobs": db_dict.get("failed_jobs", 0),
    }


def service_db_to_frontend(db_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Transform service db_state_dict to frontend format"""
    return {
        "id": db_dict.get("service_id"),
        "description": db_dict.get("description"),
        "state": db_dict.get("state"),
    }


def serialize_progress(progress: List) -> List[Dict[str, Any]]:
    """Convert progress list to JSON-serializable format

    Handles both LevelInformation objects and plain dicts.
    """
    result = []
    for item in progress:
        if hasattr(item, "level"):
            # LevelInformation object
            result.append(
                {
                    "level": item.level,
                    "progress": item.progress,
                    "desc": item.desc,
                }
            )
        elif isinstance(item, dict):
            # Already a dict
            result.append(item)
        else:
            # Unknown format, skip
            pass
    return result


def serialize_job(
    job: BaseJob,
    tags: List = None,
    depends_on: List = None,
    experiment_ids: List = None,
) -> Dict[str, Any]:
    """Serialize job using db_state_dict and transform to frontend format

    Args:
        job: Job or MockJob instance
        tags: Optional tags list (for live jobs from scheduler)
        depends_on: Optional dependencies list (for live jobs from scheduler)
        experiment_ids: Optional experiment IDs (for live jobs from scheduler)
    """
    db_dict = job.db_state_dict()

    # Convert progress to JSON-serializable format
    if "progress" in db_dict:
        db_dict["progress"] = serialize_progress(db_dict["progress"])

    # Add additional fields not in db_state_dict
    if tags is not None:
        db_dict["tags"] = tags
    if depends_on is not None:
        db_dict["depends_on"] = depends_on
    if experiment_ids is not None:
        db_dict["experiment_ids"] = experiment_ids

    return job_db_to_frontend(db_dict)


def serialize_live_job(job: Job) -> Dict[str, Any]:
    """Serialize a live Job from scheduler with full metadata"""
    # Get experiment IDs
    experiment_ids = [xp.workdir.name for xp in job.experiments]

    # Get dependencies
    depends_on = [
        dep.origin.identifier
        for dep in job.dependencies
        if isinstance(dep, JobDependency)
    ]

    # Get tags
    tags = list(job.tags.items())

    return serialize_job(
        job,
        tags=tags,
        depends_on=depends_on,
        experiment_ids=experiment_ids,
    )


def serialize_live_job_update(job: Job) -> Dict[str, Any]:
    """Serialize a live Job update (partial data)"""
    experiment_ids = [xp.workdir.name for xp in job.experiments]

    return {
        "jobId": job.identifier,
        "status": job.state.name.lower(),
        "progress": serialize_progress(job.progress),
        "experimentIds": experiment_ids,
    }


def serialize_experiment(exp: BaseExperiment) -> Dict[str, Any]:
    """Serialize experiment using db_state_dict and transform to frontend format"""
    db_dict = exp.db_state_dict()
    return experiment_db_to_frontend(db_dict)


def serialize_service(service: BaseService) -> Dict[str, Any]:
    """Serialize service using db_state_dict and transform to frontend format"""
    db_dict = service.db_state_dict()
    return service_db_to_frontend(db_dict)


# =============================================================================
# WebSocket Handler
# =============================================================================


class WebSocketHandler:
    """Manages WebSocket connections and message routing

    Message Protocol (JSON):
        Client -> Server:
            {"type": "refresh", "payload": {"experimentId": "..."}}
            {"type": "experiments"}
            {"type": "services"}
            {"type": "job.details", "payload": {"jobId": "...", "experimentId": "..."}}
            {"type": "job.kill", "payload": {"jobId": "...", "experimentId": "..."}}
            {"type": "quit"}

        Server -> Client:
            {"type": "job.add", "payload": {...}}
            {"type": "job.update", "payload": {...}}
            {"type": "experiment.add", "payload": {...}}
            {"type": "service.add", "payload": {...}}
            {"type": "service.update", "payload": {...}}
            {"type": "error", "payload": {"message": "..."}}
    """

    def __init__(self, state_provider: StateProvider, token: str):
        """Initialize WebSocket handler

        Args:
            state_provider: StateProvider for data access
            token: Authentication token
        """
        self.state_provider = state_provider
        self.token = token
        self.connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

        # Check if we have a scheduler (active experiment mode)
        self._scheduler: Optional[Scheduler] = None
        if isinstance(state_provider, Scheduler):
            self._scheduler = state_provider

    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection

        Args:
            websocket: FastAPI WebSocket connection
        """
        # Accept connection first
        await websocket.accept()

        # Validate token from query params or cookies
        token = websocket.query_params.get("token")
        if not token:
            # Try to get from cookies
            cookies = websocket.cookies
            token = cookies.get("experimaestro_token")

        if token != self.token:
            await websocket.send_json(
                {"type": "error", "payload": {"message": "Invalid token"}}
            )
            await websocket.close(code=1008, reason="Invalid token")
            return

        async with self._lock:
            self.connections.add(websocket)

        logger.info("WebSocket client connected (total: %d)", len(self.connections))

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        async with self._lock:
            self.connections.discard(websocket)

        logger.info("WebSocket client disconnected (total: %d)", len(self.connections))

    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Route incoming message to appropriate handler

        Args:
            websocket: Source WebSocket connection
            message: Parsed JSON message with 'type' and optional 'payload'
        """
        msg_type = message.get("type")
        payload = message.get("payload", {})

        handlers = {
            "refresh": self._handle_refresh,
            "experiments": self._handle_experiments,
            "services": self._handle_services,
            "job.details": self._handle_job_details,
            "job.kill": self._handle_job_kill,
            "quit": self._handle_quit,
        }

        handler = handlers.get(msg_type)
        if handler:
            try:
                await handler(websocket, payload)
            except Exception as e:
                logger.error("Error handling message %s: %s", msg_type, e)
                await websocket.send_json(
                    {"type": "error", "payload": {"message": str(e)}}
                )
        else:
            logger.warning("Unknown message type: %s", msg_type)

    async def broadcast(self, msg_type: str, payload: Dict[str, Any]):
        """Broadcast message to all connected clients

        Args:
            msg_type: Message type (e.g., "job.add", "job.update")
            payload: Message payload
        """
        message = {"type": msg_type, "payload": payload}
        disconnected = []

        async with self._lock:
            for websocket in self.connections:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(websocket)

            # Clean up disconnected clients
            for ws in disconnected:
                self.connections.discard(ws)

    async def _handle_refresh(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle refresh request - send all jobs for experiment(s)"""
        experiment_id = payload.get("experimentId")

        if experiment_id:
            # Refresh specific experiment
            jobs = self.state_provider.get_jobs(experiment_id)
            for job in jobs:
                await websocket.send_json(
                    {
                        "type": "job.add",
                        "payload": job_db_to_frontend(job.db_state_dict()),
                    }
                )
        else:
            # Refresh all experiments
            if self._scheduler:
                # Active mode: get jobs from scheduler (live Job objects)
                for job in self._scheduler.jobs.values():
                    await websocket.send_json(
                        {"type": "job.add", "payload": serialize_live_job(job)}
                    )
            else:
                # Monitoring mode: get from state provider (MockJob objects)
                for exp in self.state_provider.get_experiments():
                    exp_id = exp.experiment_id
                    jobs = self.state_provider.get_jobs(exp_id)
                    for job in jobs:
                        await websocket.send_json(
                            {
                                "type": "job.add",
                                "payload": job_db_to_frontend(job.db_state_dict()),
                            }
                        )

    async def _handle_experiments(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle experiments request - send all experiments"""
        experiments = self.state_provider.get_experiments()
        for exp in experiments:
            await websocket.send_json(
                {"type": "experiment.add", "payload": serialize_experiment(exp)}
            )

    async def _handle_services(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle services request - send all services"""
        if self._scheduler:
            # Get services from scheduler's experiments
            for xp in self._scheduler.experiments.values():
                for service in xp.services.values():
                    await websocket.send_json(
                        {"type": "service.add", "payload": serialize_service(service)}
                    )

    async def _handle_job_details(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle job details request"""
        job_id = payload.get("jobId")
        experiment_id = payload.get("experimentId")

        if self._scheduler and job_id in self._scheduler.jobs:
            # Get from scheduler (live Job)
            job = self._scheduler.jobs[job_id]
            await websocket.send_json(
                {"type": "job.update", "payload": serialize_live_job(job)}
            )
        elif experiment_id:
            # Get from state provider (MockJob)
            job = self.state_provider.get_job(job_id, experiment_id)
            if job:
                await websocket.send_json(
                    {
                        "type": "job.update",
                        "payload": job_db_to_frontend(job.db_state_dict()),
                    }
                )

    async def _handle_job_kill(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle job kill request"""
        job_id = payload.get("jobId")
        experiment_id = payload.get("experimentId")

        if self._scheduler and job_id in self._scheduler.jobs:
            # Kill via scheduler
            job = self._scheduler.jobs[job_id]
            future = asyncio.run_coroutine_threadsafe(
                job.aio_process(), self._scheduler.loop
            )
            process = future.result()
            if process is not None:
                process.kill()
                logger.info("Killed job %s", job_id)
        else:
            # Try state provider (may not be supported)
            try:
                self.state_provider.kill_job(experiment_id, job_id)
            except NotImplementedError:
                logger.warning("kill_job not supported for this state provider")

    async def _handle_quit(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle quit request from web interface"""
        # Get server reference to trigger quit
        # This is called from app context where server is available
        from experimaestro.webui.server import WebUIServer

        if WebUIServer._instance:
            WebUIServer._instance.request_quit()
