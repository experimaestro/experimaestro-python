"""State Bridge - connects StateProvider events to WebSocket broadcasts

Mirrors the TUI's STATE_EVENT_HANDLERS pattern for consistent event handling.
Uses db_state_dict() serialization consistent with SSHStateProviderServer.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from experimaestro.scheduler.state_provider import (
    StateProvider,
    StateEvent,
    ExperimentUpdatedEvent,
    JobUpdatedEvent,
    JobExperimentUpdatedEvent,
    ServiceUpdatedEvent,
    RunUpdatedEvent,
)

if TYPE_CHECKING:
    from experimaestro.webui.websocket import WebSocketHandler

logger = logging.getLogger("xpm.webui.state_bridge")


class StateBridge:
    """Bridges StateProvider events to WebSocket broadcasts

    Similar to TUI's STATE_EVENT_HANDLERS pattern, converts StateEvents
    to WebSocket messages and broadcasts to all connected clients.

    Uses db_state_dict() for consistent serialization with SSHStateProviderServer.
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

    def _on_state_event(self, event: StateEvent):
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

    async def _handle_event_async(self, event: StateEvent):
        """Handle state event asynchronously"""
        handler = self._get_handler(event)
        if handler:
            await handler(event)

    def _get_handler(self, event: StateEvent):
        """Get handler for event type"""
        handlers = {
            ExperimentUpdatedEvent: self._handle_experiment_updated,
            JobUpdatedEvent: self._handle_job_updated,
            JobExperimentUpdatedEvent: self._handle_job_experiment_updated,
            ServiceUpdatedEvent: self._handle_service_updated,
            RunUpdatedEvent: self._handle_run_updated,
        }
        return handlers.get(type(event))

    async def _handle_experiment_updated(self, event: ExperimentUpdatedEvent):
        """Handle experiment update - broadcast experiment.add"""
        from experimaestro.webui.websocket import serialize_experiment

        if event.experiment:
            payload = serialize_experiment(event.experiment)
            await self.ws_handler.broadcast("experiment.add", payload)

    async def _handle_job_updated(self, event: JobUpdatedEvent):
        """Handle job update - broadcast job.update"""
        from experimaestro.webui.websocket import (
            serialize_live_job_update,
            job_db_to_frontend,
        )

        if event.job:
            # Check if it's a live Job or MockJob
            if hasattr(event.job, "experiments"):
                # Live Job from scheduler
                payload = serialize_live_job_update(event.job)
            else:
                # MockJob from database
                payload = job_db_to_frontend(event.job.db_state_dict())

            await self.ws_handler.broadcast("job.update", payload)

    async def _handle_job_experiment_updated(self, event: JobExperimentUpdatedEvent):
        """Handle job added to experiment - broadcast job.add"""
        from experimaestro.webui.websocket import job_db_to_frontend

        # This event means a job was linked to an experiment
        # Fetch the full job data and broadcast
        job = self.state_provider.get_job(event.job_id, event.experiment_id)
        if job:
            # Get base dict from db_state_dict
            db_dict = job.db_state_dict()

            # Add tags and dependencies from event
            db_dict["tags"] = list(event.tags.items()) if event.tags else []
            db_dict["depends_on"] = event.depends_on or []

            payload = job_db_to_frontend(db_dict)
            await self.ws_handler.broadcast("job.add", payload)

    async def _handle_service_updated(self, event: ServiceUpdatedEvent):
        """Handle service update - broadcast service.add or service.update"""
        from experimaestro.webui.websocket import service_db_to_frontend

        if event.service:
            payload = service_db_to_frontend(event.service.db_state_dict())
            # Use service.add for new services, service.update for state changes
            await self.ws_handler.broadcast("service.add", payload)

    async def _handle_run_updated(self, event: RunUpdatedEvent):
        """Handle run update - triggers experiment refresh"""
        from experimaestro.webui.websocket import serialize_experiment

        # Run updates affect experiment stats, so broadcast experiment update
        experiments = self.state_provider.get_experiments()
        for exp in experiments:
            if exp.experiment_id == event.experiment_id:
                payload = serialize_experiment(exp)
                await self.ws_handler.broadcast("experiment.add", payload)
                break

    def close(self):
        """Clean up - remove listener"""
        try:
            self.state_provider.remove_listener(self._on_state_event)
        except ValueError:
            pass  # Already removed
