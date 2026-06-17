"""State Bridge - connects StateProvider events to WebSocket broadcasts

Mirrors the TUI's STATE_EVENT_HANDLERS pattern for consistent event handling.
Serialization goes through the canonical ``state_dict()`` / ``full_state_dict()``
interface (see ``scheduler/remote/server.py``), reusing the ``serialize_*``
helpers in ``webui/websocket.py``.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.state_status import (
    EventBase,
    ExperimentUpdatedEvent,
    ExperimentJobStateEvent,
    JobStateChangedEvent,
    JobProgressEvent,
    CarbonMetricsEvent,
    JobSubmittedEvent,
    ServiceAddedEvent,
    ServiceStateChangedEvent,
    ActionAddedEvent,
    WarningEvent,
    ErrorEvent,
    RunUpdatedEvent,
    RunCompletedEvent,
)

if TYPE_CHECKING:
    from experimaestro.webui.websocket import WebSocketHandler

logger = logging.getLogger("xpm.webui.state_bridge")


class StateBridge:
    """Bridges StateProvider events to WebSocket broadcasts

    Similar to TUI's STATE_EVENT_HANDLERS pattern, converts events
    to WebSocket messages and broadcasts to all connected clients.
    """

    def __init__(
        self,
        state_provider: StateProvider,
        ws_handler: "WebSocketHandler",
    ):
        """Initialize state bridge

        Args:
            state_provider: StateProvider to listen to
            ws_handler: WebSocket handler for broadcasting
        """
        self.state_provider = state_provider
        self.ws_handler = ws_handler
        self._loop: asyncio.AbstractEventLoop = None

        # Register as listener
        state_provider.add_listener(self._on_state_event)

    def _on_state_event(self, event: EventBase):
        """Handle state event from provider

        Called from provider's thread, schedules async broadcast.
        """
        # Get or create event loop for async operations
        try:
            # Try to get running loop (if called from async context)
            loop = asyncio.get_running_loop()
            loop.create_task(self._handle_event_async(event))
        except RuntimeError:
            # No running loop - create new one or use thread-safe call
            asyncio.run(self._handle_event_async(event))

    async def _handle_event_async(self, event: EventBase):
        """Handle state event asynchronously"""
        handler = self._get_handler(event)
        if handler:
            await handler(event)

    def _get_handler(self, event: EventBase):
        """Get handler for event type"""
        handlers = {
            ExperimentUpdatedEvent: self._handle_experiment_updated,
            JobStateChangedEvent: self._handle_job_state_changed,
            ExperimentJobStateEvent: self._handle_experiment_job_state,
            JobProgressEvent: self._handle_job_progress,
            CarbonMetricsEvent: self._handle_carbon_metrics,
            JobSubmittedEvent: self._handle_job_submitted,
            ServiceAddedEvent: self._handle_service_added,
            ServiceStateChangedEvent: self._handle_service_state_changed,
            ActionAddedEvent: self._handle_action_added,
            WarningEvent: self._handle_warning,
            ErrorEvent: self._handle_error,
            RunUpdatedEvent: self._handle_run_updated,
            RunCompletedEvent: self._handle_run_updated,
        }
        return handlers.get(type(event))

    async def _handle_experiment_updated(self, event: ExperimentUpdatedEvent):
        """Handle experiment update - broadcast experiment.add"""
        from experimaestro.webui.websocket import serialize_experiment

        # Fetch experiment from state provider
        experiment = self.state_provider.get_experiment(event.experiment_id)
        if experiment:
            payload = serialize_experiment(experiment)
            await self.ws_handler.broadcast("experiment.add", payload)

    async def _handle_job_state_changed(self, event: JobStateChangedEvent):
        """Handle job execution state change - broadcast job.update"""
        await self._broadcast_job_update(event.job_id)

    async def _handle_experiment_job_state(self, event: ExperimentJobStateEvent):
        """Handle scheduler lifecycle state change - broadcast job.update"""
        await self._broadcast_job_update(event.job_id)

    async def _handle_job_progress(self, event: JobProgressEvent):
        """Handle progress update - broadcast job.update"""
        await self._broadcast_job_update(event.job_id)

    async def _handle_carbon_metrics(self, event: CarbonMetricsEvent):
        """Handle carbon metrics update - broadcast job.update (carries carbon)"""
        await self._broadcast_job_update(event.job_id)

    async def _broadcast_job_update(self, job_id: str):
        """Broadcast job.update for a given job_id"""
        from experimaestro.webui.websocket import serialize_job

        # Fetch job from state provider
        jobs = self.state_provider.get_all_jobs()
        job = next((j for j in jobs if j.identifier == job_id), None)
        if job:
            await self.ws_handler.broadcast("job.update", serialize_job(job))

    async def _handle_job_submitted(self, event: JobSubmittedEvent):
        """Handle job added to experiment - broadcast job.add"""
        from experimaestro.webui.websocket import serialize_job

        # Fetch the full job data from state provider
        job = self.state_provider.get_job(event.task_id, event.job_id)
        if job:
            tags = [(tag.key, tag.value) for tag in event.tags] if event.tags else []
            payload = serialize_job(
                job,
                tags=tags,
                depends_on=event.depends_on or [],
                experiment_ids=[event.experiment_id] if event.experiment_id else None,
                submitted=event.submitted_time,
            )
            await self.ws_handler.broadcast("job.add", payload)

    async def _handle_service_added(self, event: ServiceAddedEvent):
        """Handle service added - broadcast service.add"""
        from experimaestro.webui.websocket import serialize_service

        # Fetch service from state provider
        services = self.state_provider.get_services(event.experiment_id, event.run_id)
        service = next((s for s in services if s.id == event.service_id), None)
        if service:
            await self.ws_handler.broadcast("service.add", serialize_service(service))

    async def _handle_service_state_changed(self, event: ServiceStateChangedEvent):
        """Handle service state change - broadcast service.update"""
        from experimaestro.webui.websocket import serialize_service

        services = self.state_provider.get_services(event.experiment_id, event.run_id)
        service = next((s for s in services if s.id == event.service_id), None)
        if service:
            await self.ws_handler.broadcast(
                "service.update", serialize_service(service)
            )

    async def _handle_action_added(self, event: ActionAddedEvent):
        """Handle action added - broadcast action.add"""
        await self.ws_handler.broadcast(
            "action.add",
            {
                "actionId": event.action_id,
                "experimentId": event.experiment_id,
                "description": event.description,
                "actionClass": event.action_class,
            },
        )

    async def _handle_warning(self, event: WarningEvent):
        """Handle warning - broadcast warning.add"""
        from experimaestro.webui.websocket import serialize_warning

        await self.ws_handler.broadcast("warning.add", serialize_warning(event))

    async def _handle_error(self, event: ErrorEvent):
        """Handle error - broadcast error toast"""
        await self.ws_handler.broadcast("error", {"message": event.error_message})

    async def _handle_run_updated(self, event):
        """Handle run update - triggers experiment refresh"""
        from experimaestro.webui.websocket import serialize_experiment

        # Run updates affect experiment stats, so broadcast experiment update
        experiment = self.state_provider.get_experiment(event.experiment_id)
        if experiment:
            payload = serialize_experiment(experiment)
            await self.ws_handler.broadcast("experiment.add", payload)

    def close(self):
        """Clean up - remove listener"""
        try:
            self.state_provider.remove_listener(self._on_state_event)
        except ValueError:
            pass  # Already removed
