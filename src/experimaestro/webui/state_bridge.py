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
    JobEventBase,
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
    ConnectionStateEvent,
)

if TYPE_CHECKING:
    from experimaestro.webui.websocket import WebSocketHandler

logger = logging.getLogger("xpm.webui.state_bridge")


class StateBridge:
    """Bridges StateProvider events to WebSocket broadcasts

    Similar to TUI's STATE_EVENT_HANDLERS pattern, converts events
    to WebSocket messages and sends them to the connected clients that watch
    them (see :class:`~experimaestro.scheduler.event_filter.EventFilter`).
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
        self._loop: asyncio.AbstractEventLoop | None = None

        # Register as listener
        state_provider.add_listener(self._on_state_event)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that serves the WebSocket connections.

        Called from a startup hook running on the uvicorn loop. Broadcasts
        scheduled from other threads are handed to this loop, which owns the
        connections and their asyncio primitives.
        """
        self._loop = loop

    def _on_state_event(self, event: EventBase):
        """Handle state event from provider

        Called from the provider's (scheduler) thread, schedules the async
        broadcast onto the WebSocket serving loop.
        """
        try:
            # Already on an event loop (event raised from async code): just
            # schedule the coroutine there.
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            running.create_task(self._handle_event_async(event))
            return

        # Called from another thread: dispatch to the serving loop. Until it is
        # known (server not started yet) there are no clients to notify, so the
        # event is simply dropped.
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.debug("No serving event loop; dropping %s", type(event).__name__)
            return
        asyncio.run_coroutine_threadsafe(self._handle_event_async(event), loop)

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
            ConnectionStateEvent: self._handle_connection_state,
        }
        return handlers.get(type(event))

    async def _handle_experiment_updated(self, event: ExperimentUpdatedEvent):
        """Handle experiment update - broadcast experiment.add"""
        from experimaestro.webui.websocket import serialize_experiment

        # Fetch experiment from state provider
        experiment = self.state_provider.get_experiment(event.experiment_id)
        if experiment:
            payload = serialize_experiment(experiment)
            await self.ws_handler.send_event(event, "experiment.add", payload)

    async def _handle_job_state_changed(self, event: JobStateChangedEvent):
        """Handle job execution state change - broadcast job.update"""
        await self._broadcast_job_update(event)

    async def _handle_experiment_job_state(self, event: ExperimentJobStateEvent):
        """Handle scheduler lifecycle state change - broadcast job.update"""
        await self._broadcast_job_update(event)

    async def _handle_job_progress(self, event: JobProgressEvent):
        """Handle progress update - broadcast job.update"""
        await self._broadcast_job_update(event)

    async def _handle_carbon_metrics(self, event: CarbonMetricsEvent):
        """Handle carbon metrics update - broadcast job.update (carries carbon)"""
        await self._broadcast_job_update(event)

    async def _broadcast_job_update(self, event: JobEventBase):
        """Broadcast the execution state of the job an event is about"""
        from experimaestro.webui.websocket import serialize_job_update

        recipients = self.ws_handler.recipients(event)
        if not recipients:
            return
        if event.task_id:
            job = self.state_provider.get_job(event.task_id, event.job_id)
        else:
            job = next(
                (
                    j
                    for j in self.state_provider.get_all_jobs()
                    if j.identifier == event.job_id
                ),
                None,
            )
        if job:
            # Scheduler state events say which experiment the state comes from
            experiment = (
                self.state_provider.get_experiment(event.experiment_id)
                if isinstance(event, ExperimentJobStateEvent) and event.experiment_id
                else None
            )
            await self.ws_handler.send_event(
                event,
                "job.update",
                serialize_job_update(job, experiment=experiment),
                recipients,
            )

    async def _handle_job_submitted(self, event: JobSubmittedEvent):
        """Handle job added to experiment - broadcast job.add"""
        from experimaestro.webui.websocket import serialize_job

        recipients = self.ws_handler.recipients(event)
        if not recipients:
            return
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
                experiment=(
                    self.state_provider.get_experiment(event.experiment_id)
                    if event.experiment_id
                    else None
                ),
            )
            await self.ws_handler.send_event(event, "job.add", payload, recipients)

    async def _handle_service_added(self, event: ServiceAddedEvent):
        """Handle service added - broadcast service.add"""
        from experimaestro.webui.websocket import serialize_service

        recipients = self.ws_handler.recipients(event)
        if not recipients:
            return
        # Fetch service from state provider
        services = self.state_provider.get_services(event.experiment_id, event.run_id)
        service = next((s for s in services if s.id == event.service_id), None)
        if service:
            await self.ws_handler.send_event(
                event, "service.add", serialize_service(service), recipients
            )

    async def _handle_service_state_changed(self, event: ServiceStateChangedEvent):
        """Handle service state change - broadcast service.update"""
        from experimaestro.webui.websocket import serialize_service

        recipients = self.ws_handler.recipients(event)
        if not recipients:
            return
        services = self.state_provider.get_services(event.experiment_id, event.run_id)
        service = next((s for s in services if s.id == event.service_id), None)
        if service:
            await self.ws_handler.send_event(
                event, "service.update", serialize_service(service), recipients
            )

    async def _handle_action_added(self, event: ActionAddedEvent):
        """Handle action added - broadcast action.add"""
        await self.ws_handler.send_event(
            event,
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

        await self.ws_handler.send_event(event, "warning.add", serialize_warning(event))

    async def _handle_error(self, event: ErrorEvent):
        """Handle error - broadcast error toast"""
        await self.ws_handler.send_event(
            event, "error", {"message": event.error_message}
        )

    async def _handle_run_updated(self, event):
        """Handle run update - triggers experiment refresh"""
        from experimaestro.webui.websocket import serialize_experiment

        # Run updates affect experiment stats, so broadcast experiment update
        experiment = self.state_provider.get_experiment(event.experiment_id)
        if experiment:
            payload = serialize_experiment(experiment)
            await self.ws_handler.send_event(event, "experiment.add", payload)

    async def _handle_connection_state(self, event: ConnectionStateEvent):
        """Handle a change of the provider connection - broadcast
        connection.state (once reconnected, the pages request the state
        again)"""
        from experimaestro.webui.websocket import serialize_connection_state

        await self.ws_handler.send_event(
            event, "connection.state", serialize_connection_state(event)
        )

    def close(self):
        """Clean up - remove listener"""
        self.state_provider.unwatch(self.ws_handler)
        try:
            self.state_provider.remove_listener(self._on_state_event)
        except ValueError:
            pass  # Already removed
