"""WebSocket handler for real-time communication

Manages WebSocket connections and message routing.
Uses native WebSocket with JSON protocol.

Serialization mirrors the remote SSH server (``scheduler/remote/server.py``):
jobs/experiments/services are serialized through the canonical
``state_dict()`` / ``full_state_dict()`` interface and the StateProvider query
methods, then transformed into the frontend (camelCase) format.
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
# Serialization helpers - Transform the canonical state_dict() schema to the
# frontend (camelCase) format.
#
# The single source of truth is ``state_dict()`` / ``full_state_dict()`` defined
# on BaseJob / BaseExperiment / BaseService (scheduler/interfaces.py). It already
# carries process info, full carbon metrics, failure reason, exit code and the
# scheduler lifecycle state, so the frontend gets all of those for free.
# =============================================================================


def serialize_progress(progress: List) -> List[Dict[str, Any]]:
    """Convert progress list to JSON-serializable format.

    Handles both LevelInformation objects and plain dicts (state_dict() already
    converts progress entries via ``to_dict()``).
    """
    result = []
    for item in progress:
        if isinstance(item, dict):
            result.append(item)
        elif hasattr(item, "level"):
            result.append(
                {
                    "level": item.level,
                    "progress": item.progress,
                    "desc": item.desc,
                }
            )
    return result


def _carbon_to_frontend(carbon: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Map a carbon_metrics dict (from state_dict()) to the frontend format."""
    if not carbon:
        return None
    return {
        "co2kg": carbon.get("co2_kg"),
        "energyKwh": carbon.get("energy_kwh"),
        "cpuPowerW": carbon.get("cpu_power_w"),
        "gpuPowerW": carbon.get("gpu_power_w"),
        "ramPowerW": carbon.get("ram_power_w"),
        "durationS": carbon.get("duration_s"),
        "region": carbon.get("region") or None,
        "isFinal": carbon.get("is_final", False),
    }


def _process_to_frontend(process: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Map a process dict (from state_dict()/get_process_info) to frontend format."""
    if not process:
        return None
    return {
        "pid": process.get("pid"),
        "type": process.get("type"),
        "running": process.get("running", False),
        "cpuPercent": process.get("cpu_percent"),
        "memoryMb": process.get("memory_mb"),
        "numThreads": process.get("num_threads"),
    }


def _job_dict_to_frontend(d: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a job ``state_dict()`` into the frontend (camelCase) format.

    state_dict() schema (snake_case):
        job_id, task_id, path, state, scheduler_state, failure_reason,
        started_time, ended_time, exit_code, retry_count, progress, process,
        carbon_metrics, ...  (+ enriched: tags, depends_on, experiment_ids,
        submitted_time)
    """
    scheduler_state = d.get("scheduler_state")
    return {
        "jobId": d.get("job_id"),
        "taskId": d.get("task_id"),
        "locator": d.get("path") or "",
        "status": (d.get("state") or "unknown").lower(),
        "schedulerState": scheduler_state.lower() if scheduler_state else None,
        "start": d.get("started_time") or "",
        "end": d.get("ended_time") or "",
        "submitted": d.get("submitted_time") or "",
        "failureReason": d.get("failure_reason"),
        "exitCode": d.get("exit_code"),
        "retryCount": d.get("retry_count", 0),
        "tags": d.get("tags", []),
        "progress": serialize_progress(d.get("progress", []) or []),
        "experimentIds": d.get("experiment_ids", []),
        "dependsOn": d.get("depends_on", []),
        "carbon": _carbon_to_frontend(d.get("carbon_metrics")),
        "process": _process_to_frontend(d.get("process")),
    }


def serialize_job(
    job: BaseJob,
    tags: List = None,
    depends_on: List = None,
    experiment_ids: List = None,
    submitted: str = None,
    process_info: Any = None,
) -> Dict[str, Any]:
    """Serialize a job using ``state_dict()`` and transform to frontend format.

    Args:
        job: Job or MockJob instance
        tags: Optional tags list (experiment-specific, not part of state_dict)
        depends_on: Optional dependency job ids (experiment-specific)
        experiment_ids: Optional experiment IDs the job belongs to
        submitted: Optional submit-time ISO string (from ExperimentJobInformation)
        process_info: Optional live ProcessInfo (from get_process_info) used to
            enrich the static process dict with cpu/memory/threads
    """
    d = job.state_dict()

    if tags is not None:
        d["tags"] = tags
    if depends_on is not None:
        d["depends_on"] = depends_on
    if experiment_ids is not None:
        d["experiment_ids"] = experiment_ids
    if submitted is not None:
        d["submitted_time"] = submitted
    if process_info is not None:
        d["process"] = {
            "pid": process_info.pid,
            "type": process_info.type,
            "running": process_info.running,
            "cpu_percent": process_info.cpu_percent,
            "memory_mb": process_info.memory_mb,
            "num_threads": process_info.num_threads,
        }

    return _job_dict_to_frontend(d)


def serialize_live_job(job: Job) -> Dict[str, Any]:
    """Serialize a live Job from scheduler with full metadata"""
    # Get experiment IDs (must match serialize_experiment's experiment_id, which
    # is exp.experiment_id — NOT workdir.name, which is the run_id).
    experiment_ids = [xp.experiment_id for xp in job.experiments]

    # Get dependencies
    depends_on = [
        dep.origin.identifier
        for dep in job.dependencies
        if isinstance(dep, JobDependency)
    ]

    # Get tags from config (tags are experiment-specific)
    tags = list(job.config.tags().items())

    return serialize_job(
        job,
        tags=tags,
        depends_on=depends_on,
        experiment_ids=experiment_ids,
    )


def serialize_experiment(exp: BaseExperiment) -> Dict[str, Any]:
    """Serialize an experiment using ``state_dict()`` plus aggregate counts."""
    d = exp.state_dict()

    def _count(attr: str) -> int:
        try:
            return int(getattr(exp, attr))
        except Exception:
            return 0

    # Flatten the experiment carbon impact ({sum, latest} aggregates) to the same
    # shape the frontend uses for jobs; prefer the per-job "latest" aggregate.
    carbon = None
    impact = d.get("carbon_impact") or {}
    agg = impact.get("latest") or impact.get("sum")
    if agg:
        carbon = {"co2kg": agg.get("co2_kg"), "energyKwh": agg.get("energy_kwh")}

    workdir = getattr(exp, "workdir", None)
    return {
        "experiment_id": d.get("experiment_id"),
        "workdir": str(workdir) if workdir else None,
        "current_run_id": d.get("run_id"),
        "status": d.get("status"),
        "hostname": d.get("hostname"),
        "started_at": d.get("started_at"),
        "ended_at": d.get("ended_at"),
        "total_jobs": _count("total_jobs"),
        "finished_jobs": _count("finished_jobs"),
        "failed_jobs": _count("failed_jobs"),
        "carbon": carbon,
    }


def serialize_service(service: BaseService) -> Dict[str, Any]:
    """Serialize a service using ``full_state_dict()`` to frontend format."""
    d = service.full_state_dict()
    return {
        "id": d.get("service_id"),
        "description": d.get("description"),
        "state": d.get("state"),
        "url": d.get("url"),
        "syncStatus": _safe_attr(service, "sync_status"),
        "error": _safe_attr(service, "error"),
    }


def _safe_attr(obj: Any, name: str) -> Any:
    """Read an optional property that may raise NotImplementedError."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def serialize_warning(event: Any) -> Dict[str, Any]:
    """Serialize a WarningEvent to the frontend format."""
    return {
        "warningKey": getattr(event, "warning_key", ""),
        "experimentId": getattr(event, "experiment_id", ""),
        "runId": getattr(event, "run_id", ""),
        "description": getattr(event, "description", ""),
        "severity": getattr(event, "severity", "warning"),
        # action_key -> label
        "actions": getattr(event, "actions", {}) or {},
        "context": getattr(event, "context", {}) or {},
    }


def serialize_action(action: Any, experiment_id: str = "") -> Dict[str, Any]:
    """Serialize a BaseAction (live or mock) to the frontend format."""
    return {
        "actionId": getattr(action, "action_id", ""),
        "experimentId": experiment_id,
        "description": action.description() if hasattr(action, "description") else "",
        "actionClass": getattr(action, "action_class", ""),
    }


def serialize_run(run: BaseExperiment, current_run_id: Optional[str]) -> Dict[str, Any]:
    """Serialize one experiment run for the run-history view."""
    payload = serialize_experiment(run)
    run_id = payload.get("current_run_id")
    payload["run_id"] = run_id
    payload["isCurrent"] = run_id is not None and run_id == current_run_id
    return payload


def _dir_size(path) -> int:
    """Best-effort recursive size of a directory in bytes."""
    import os

    total = 0
    try:
        for root, _dirs, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def serialize_orphan_job(job: BaseJob, is_stray: bool) -> Dict[str, Any]:
    """Serialize an orphan/stray job with its on-disk size."""
    payload = serialize_job(job)
    payload["isStray"] = is_stray
    payload["sizeBytes"] = _dir_size(job.path) if job.path else 0
    return payload


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
            "experiment.runs": self._handle_experiment_runs,
            "services": self._handle_services,
            "service.start": self._handle_service_start,
            "service.stop": self._handle_service_stop,
            "job.details": self._handle_job_details,
            "job.kill": self._handle_job_kill,
            "job.delete": self._handle_job_delete,
            "warnings": self._handle_warnings,
            "warning.action": self._handle_warning_action,
            "actions": self._handle_actions,
            "action.execute": self._handle_action_execute,
            "orphans": self._handle_orphans,
            "orphan.kill": self._handle_orphan_kill,
            "orphan.delete": self._handle_orphan_delete,
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

    def _serialize_jobs_for_experiment(
        self, experiment_id: str, run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Serialize jobs of an experiment, enriched with tags/deps/submit time.

        Used for offline (MockJob) providers, where the state_dict alone does not
        carry experiment-specific tags, dependencies or submit time.
        """
        from experimaestro.scheduler.interfaces import serialize_timestamp

        jobs = self.state_provider.get_jobs(experiment_id, run_id=run_id)
        tags_map = self.state_provider.get_tags_map(experiment_id, run_id) or {}
        deps_map = self.state_provider.get_dependencies_map(experiment_id, run_id) or {}
        job_info = (
            self.state_provider.get_experiment_job_info(experiment_id, run_id) or {}
        )

        payloads = []
        for job in jobs:
            jid = job.identifier
            info = job_info.get(jid)
            submitted = (
                serialize_timestamp(info.timestamp)
                if info and info.timestamp is not None
                else None
            )
            payloads.append(
                serialize_job(
                    job,
                    tags=list(tags_map.get(jid, {}).items()),
                    depends_on=deps_map.get(jid, []),
                    experiment_ids=[experiment_id],
                    submitted=submitted,
                )
            )
        return payloads

    async def _handle_refresh(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle refresh request - send all jobs for experiment(s)"""
        experiment_id = payload.get("experimentId")
        run_id = payload.get("runId")

        if experiment_id:
            # Refresh specific experiment (optionally a specific run)
            for job_payload in self._serialize_jobs_for_experiment(
                experiment_id, run_id
            ):
                await websocket.send_json({"type": "job.add", "payload": job_payload})
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
                    for job_payload in self._serialize_jobs_for_experiment(
                        exp.experiment_id
                    ):
                        await websocket.send_json(
                            {"type": "job.add", "payload": job_payload}
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

        if self._scheduler and job_id in self._scheduler.jobs:
            # Get from scheduler (live Job)
            job = self._scheduler.jobs[job_id]
            await websocket.send_json(
                {"type": "job.update", "payload": serialize_live_job(job)}
            )
        else:
            # Get from state provider (MockJob)
            task_id = payload.get("taskId")
            if task_id:
                job = self.state_provider.get_job(task_id, job_id)
                if job:
                    # Enrich with live process info (cpu/memory/threads) when the
                    # provider supports it.
                    try:
                        process_info = self.state_provider.get_process_info(job)
                    except Exception:
                        process_info = None
                    await websocket.send_json(
                        {
                            "type": "job.update",
                            "payload": serialize_job(job, process_info=process_info),
                        }
                    )

    async def _handle_job_kill(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle job kill request"""
        job_id = payload.get("jobId")
        task_id = payload.get("taskId")

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
            # Offline provider: kill_job expects a BaseJob instance
            try:
                job = self.state_provider.get_job(task_id, job_id) if task_id else None
                if job is None:
                    raise ValueError(f"Unknown job {task_id}:{job_id}")
                self.state_provider.kill_job(job, perform=True)
            except NotImplementedError:
                logger.warning("kill_job not supported for this state provider")
            except Exception as e:
                logger.warning("Failed to kill job %s: %s", job_id, e)
                await websocket.send_json(
                    {"type": "error", "payload": {"message": str(e)}}
                )

    async def _handle_job_delete(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle job delete request (removes the job directory from disk)."""
        job = self._lookup_job(payload)
        if job is None:
            await websocket.send_json(
                {"type": "error", "payload": {"message": "Job not found"}}
            )
            return
        try:
            ok, message = self.state_provider.delete_job_safely(job, perform=True)
            if ok:
                await self.broadcast("job.removed", {"jobId": payload.get("jobId")})
            else:
                await websocket.send_json(
                    {"type": "error", "payload": {"message": message}}
                )
        except Exception as e:
            logger.warning("Failed to delete job %s: %s", payload.get("jobId"), e)
            await websocket.send_json({"type": "error", "payload": {"message": str(e)}})

    async def _handle_experiment_runs(
        self, websocket: WebSocket, payload: Dict[str, Any]
    ):
        """Handle run-history request for an experiment."""
        experiment_id = payload.get("experimentId")
        if not experiment_id:
            return
        runs = self.state_provider.get_experiment_runs(experiment_id) or []
        current = self.state_provider.get_current_run(experiment_id)
        await websocket.send_json(
            {
                "type": "experiment.runs",
                "payload": {
                    "experimentId": experiment_id,
                    "runs": [serialize_run(r, current) for r in runs],
                },
            }
        )

    def _find_live_service(self, service_id: str):
        """Find a live service across the scheduler's experiments."""
        if self._scheduler:
            for xp in self._scheduler.experiments.values():
                svc = xp.services.get(service_id)
                if svc is not None:
                    return svc
        return None

    async def _handle_service_start(
        self, websocket: WebSocket, payload: Dict[str, Any]
    ):
        """Start a service (active experiment only)."""
        await self._service_control(websocket, payload, stop=False)

    async def _handle_service_stop(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Stop a service (active experiment only)."""
        await self._service_control(websocket, payload, stop=True)

    async def _service_control(
        self, websocket: WebSocket, payload: Dict[str, Any], *, stop: bool
    ):
        service_id = payload.get("id")
        if self._scheduler is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "payload": {
                        "message": "Service control requires an active experiment"
                    },
                }
            )
            return

        service = self._find_live_service(service_id)
        if service is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "payload": {"message": f"Service {service_id} not found"},
                }
            )
            return

        loop = asyncio.get_running_loop()

        def control():
            live = service.to_service()
            if stop:
                if hasattr(live, "stop"):
                    live.stop()
            elif hasattr(live, "get_url"):
                # Services start lazily on first URL request
                live.get_url()
            return live

        try:
            live = await loop.run_in_executor(None, control)
            await self.broadcast("service.update", serialize_service(live))
        except Exception as e:
            logger.warning("Service %s control failed: %s", service_id, e)
            if hasattr(service, "set_error"):
                service.set_error(str(e))
            await self.broadcast("service.update", serialize_service(service))
            await websocket.send_json({"type": "error", "payload": {"message": str(e)}})

    async def _handle_warnings(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Send all unresolved warnings."""
        for warning in self.state_provider.get_unresolved_warnings():
            await websocket.send_json(
                {"type": "warning.add", "payload": serialize_warning(warning)}
            )

    async def _handle_warning_action(
        self, websocket: WebSocket, payload: Dict[str, Any]
    ):
        """Execute a warning action."""
        self.state_provider.execute_warning_action(
            payload.get("warningKey", ""),
            payload.get("actionKey", ""),
            payload.get("experimentId", ""),
            payload.get("runId", ""),
        )
        await websocket.send_json(
            {
                "type": "warning.resolved",
                "payload": {"warningKey": payload.get("warningKey", "")},
            }
        )

    async def _handle_actions(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Send the actions declared by the (selected) experiment(s)."""
        experiment_id = payload.get("experimentId")
        if experiment_id:
            experiments = [self.state_provider.get_experiment(experiment_id)]
        else:
            experiments = self.state_provider.get_experiments()
        for exp in experiments:
            if exp is None:
                continue
            for action in (exp.actions or {}).values():
                await websocket.send_json(
                    {
                        "type": "action.add",
                        "payload": serialize_action(action, exp.experiment_id),
                    }
                )

    async def _handle_action_execute(
        self, websocket: WebSocket, payload: Dict[str, Any]
    ):
        """Execute an action, prompting the frontend for any missing inputs."""
        from experimaestro.webui.actions import WebInteraction, ActionInputRequired

        experiment_id = payload.get("experimentId")
        action_id = payload.get("actionId")
        inputs = payload.get("inputs") or {}

        exp = (
            self.state_provider.get_experiment(experiment_id) if experiment_id else None
        )
        if exp is None:
            await websocket.send_json(
                {"type": "error", "payload": {"message": "Experiment not found"}}
            )
            return

        def run():
            action_obj = (exp.actions or {}).get(action_id)
            if action_obj is not None and hasattr(type(action_obj), "execute"):
                action = action_obj
            else:
                from experimaestro.core.serialization import load_xp_info

                xp_info = load_xp_info(exp.workdir)
                if action_id not in xp_info.actions:
                    raise ValueError(f"Action '{action_id}' not found")
                action = xp_info.actions[action_id]
            action.execute(WebInteraction(inputs))

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, run)
            await websocket.send_json(
                {
                    "type": "action.result",
                    "payload": {"actionId": action_id, "ok": True},
                }
            )
        except ActionInputRequired as e:
            await websocket.send_json(
                {
                    "type": "action.prompt",
                    "payload": {
                        "actionId": action_id,
                        "experimentId": experiment_id,
                        "field": e.field,
                        "inputs": inputs,
                    },
                }
            )
        except Exception as e:
            logger.warning("Action %s failed: %s", action_id, e)
            await websocket.send_json(
                {
                    "type": "action.result",
                    "payload": {"actionId": action_id, "ok": False, "error": str(e)},
                }
            )

    async def _handle_orphans(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Send orphan/stray jobs with their on-disk sizes."""
        loop = asyncio.get_running_loop()

        def gather():
            orphans = self.state_provider.get_orphan_jobs() or []
            stray_ids = {
                j.identifier for j in (self.state_provider.get_stray_jobs() or [])
            }
            return [serialize_orphan_job(j, j.identifier in stray_ids) for j in orphans]

        jobs = await loop.run_in_executor(None, gather)
        await websocket.send_json({"type": "orphans", "payload": {"jobs": jobs}})

    async def _handle_orphan_kill(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Kill an orphan/stray job."""
        job = self._lookup_job(payload)
        if job is None:
            return
        try:
            self.state_provider.kill_job(job, perform=True)
        except Exception as e:
            await websocket.send_json({"type": "error", "payload": {"message": str(e)}})

    async def _handle_orphan_delete(
        self, websocket: WebSocket, payload: Dict[str, Any]
    ):
        """Delete an orphan job from disk."""
        job = self._lookup_job(payload)
        if job is None:
            return
        try:
            ok, message = self.state_provider.delete_job_safely(job, perform=True)
            if ok:
                await websocket.send_json(
                    {
                        "type": "orphan.removed",
                        "payload": {"jobId": payload.get("jobId")},
                    }
                )
            else:
                await websocket.send_json(
                    {"type": "error", "payload": {"message": message}}
                )
        except Exception as e:
            await websocket.send_json({"type": "error", "payload": {"message": str(e)}})

    def _lookup_job(self, payload: Dict[str, Any]):
        task_id = payload.get("taskId")
        job_id = payload.get("jobId")
        if not task_id or not job_id:
            return None
        return self.state_provider.get_job(task_id, job_id)

    async def _handle_quit(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Handle quit request from web interface"""
        # Get server reference to trigger quit
        # This is called from app context where server is available
        from experimaestro.webui.server import WebUIServer

        if WebUIServer._instance:
            WebUIServer._instance.request_quit()
