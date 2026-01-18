"""Event viewer for streaming events to console

This module provides an EventStreamViewer class that outputs events to the console
in either JSON or human-readable format. It's designed to be used with state
providers for monitoring experiments in real-time.
"""

import sys
from datetime import datetime
from typing import TextIO

from experimaestro.scheduler.state_status import (
    EventBase,
    JobSubmittedEvent,
    JobStateChangedEvent,
    JobProgressEvent,
    CarbonMetricsEvent,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
    RunCompletedEvent,
    ServiceEventBase,
)


class EventStreamViewer:
    """Streams events to console in JSON or human-readable format

    This class acts as a state listener that outputs events as they arrive.
    It can output in JSON format (one JSON object per line for tooling) or
    in a human-readable text format.

    Usage:
        viewer = EventStreamViewer(format='json')
        state_provider.add_listener(viewer)
    """

    def __init__(
        self,
        output: TextIO | None = None,
        format: str = "text",
        show_progress: bool = True,
    ):
        """Initialize the event stream viewer

        Args:
            output: Output stream (defaults to sys.stdout)
            format: Output format - 'json' for JSON-Lines, 'text' for human-readable
            show_progress: If True, show progress events (can be noisy)
        """
        self.output = output or sys.stdout
        self.format = format
        self.show_progress = show_progress

    def __call__(self, event: EventBase) -> None:
        """Handle incoming event (StateListener interface)"""
        # Skip progress events if not showing them
        if not self.show_progress and isinstance(event, JobProgressEvent):
            return

        if self.format == "json":
            self._output_json(event)
        else:
            self._output_text(event)

    def _output_json(self, event: EventBase) -> None:
        """Output event as JSON-Lines format"""
        print(event.to_json(), file=self.output, flush=True)

    def _output_text(self, event: EventBase) -> None:
        """Output event in human-readable format"""
        timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        event_type = event.event_type

        # Format based on event type
        message = self._format_event(event)
        if message:
            print(
                f"[{timestamp}] {event_type}: {message}", file=self.output, flush=True
            )

    def _format_event(self, event: EventBase) -> str | None:
        """Format event details for human-readable output"""
        match event:
            case JobSubmittedEvent():
                deps = (
                    f" (depends: {len(event.depends_on)})" if event.depends_on else ""
                )
                return f"job={event.job_id[:12]}... task={event.task_id}{deps}"

            case JobStateChangedEvent():
                extra = ""
                if event.exit_code is not None:
                    extra = f" exit_code={event.exit_code}"
                if event.failure_reason:
                    extra += f" reason={event.failure_reason}"
                if event.retry_count > 0:
                    extra += f" retry={event.retry_count}"
                return f"job={event.job_id[:12]}... state={event.state}{extra}"

            case JobProgressEvent():
                desc = f" ({event.desc})" if event.desc else ""
                return f"job={event.job_id[:12]}... level={event.level} {event.progress:.1%}{desc}"

            case CarbonMetricsEvent():
                final = " [FINAL]" if event.is_final else ""
                return (
                    f"job={event.job_id[:12]}... "
                    f"CO2={event.co2_kg:.4f}kg energy={event.energy_kwh:.4f}kWh{final}"
                )

            case ExperimentUpdatedEvent():
                return f"experiment={event.experiment_id}"

            case RunUpdatedEvent():
                return f"experiment={event.experiment_id} run={event.run_id}"

            case RunCompletedEvent():
                return (
                    f"experiment={event.experiment_id} run={event.run_id} "
                    f"status={event.status}"
                )

            case ServiceEventBase():
                state = (
                    getattr(event, "state", "") if hasattr(event, "state") else "added"
                )
                return (
                    f"service={event.service_id} experiment={event.experiment_id} "
                    f"state={state}"
                )

            case _:
                # Generic fallback
                return str(event)


def run_event_viewer(
    state_provider,
    format: str = "text",
    show_progress: bool = True,
) -> None:
    """Run event viewer as the main UI

    This function blocks and streams events until interrupted.

    Args:
        state_provider: StateProvider instance to watch
        format: Output format ('json' or 'text')
        show_progress: Whether to show progress events
    """
    from termcolor import cprint

    viewer = EventStreamViewer(format=format, show_progress=show_progress)
    state_provider.add_listener(viewer)

    format_desc = "JSON" if format == "json" else "text"
    cprint(f"Streaming events ({format_desc} format). Press Ctrl+C to stop.", "yellow")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        state_provider.remove_listener(viewer)
