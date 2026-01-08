"""Services list widget for the TUI"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider


class ServicesList(Vertical):
    """Widget displaying services for selected experiment

    Services are retrieved from StateProvider.get_services() which
    abstracts away whether services are live (from scheduler) or recreated
    from database state_dict. The UI treats all services uniformly.
    """

    BINDINGS = [
        Binding("s", "start_service", "Start"),
        Binding("x", "stop_service", "Stop"),
        Binding("u", "copy_url", "Copy URL", show=False),
    ]

    # State icons for display
    STATE_ICONS = {
        "STOPPED": "⏹",
        "STARTING": "⏳",
        "RUNNING": "▶",
        "STOPPING": "⏳",
    }

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.current_experiment: Optional[str] = None
        self._services: dict = {}  # service_id -> Service object

    def compose(self) -> ComposeResult:
        yield DataTable(id="services-table", cursor_type="row")

    def on_mount(self) -> None:
        """Set up the services table"""
        table = self.query_one("#services-table", DataTable)
        table.add_columns("ID", "Description", "State", "URL")
        table.cursor_type = "row"

    def set_experiment(self, experiment_id: Optional[str]) -> None:
        """Set the current experiment and refresh services"""
        self.current_experiment = experiment_id
        self.refresh_services()

    def refresh_services(self) -> None:
        """Refresh the services list from state provider"""
        table = self.query_one("#services-table", DataTable)
        table.clear()
        self._services = {}

        if not self.current_experiment:
            return

        # Get services from state provider (handles live vs DB automatically)
        services = self.state_provider.get_services(self.current_experiment)
        self.log.info(
            f"refresh_services got {len(services)} services: {[(s.id, id(s), getattr(s, 'url', None)) for s in services]}"
        )

        for service in services:
            service_id = service.id
            self._services[service_id] = service

            state_name = service.state.name if hasattr(service, "state") else "UNKNOWN"
            state_icon = self.STATE_ICONS.get(state_name, "?")
            url = getattr(service, "url", None) or "-"
            description = (
                service.description() if hasattr(service, "description") else ""
            )

            table.add_row(
                service_id,
                description,
                f"{state_icon} {state_name}",
                url,
                key=service_id,
            )

    def _get_selected_service(self):
        """Get the currently selected Service object"""
        table = self.query_one("#services-table", DataTable)
        if table.cursor_row is not None and table.row_count > 0:
            row_key = list(table.rows.keys())[table.cursor_row]
            if row_key:
                service_id = str(row_key.value)
                return self._services.get(service_id)
        return None

    def action_start_service(self) -> None:
        """Start the selected service"""
        service = self._get_selected_service()
        if not service:
            return

        self.log.info(
            f"Starting service {service.id} (type={type(service).__name__}, "
            f"has_get_url={hasattr(service, 'get_url')}, is_live={self.state_provider.is_live})"
        )

        try:
            if hasattr(service, "get_url"):
                url = service.get_url()
                self.log.info(f"Service started, url={url}, service.url={service.url}")
                self.notify(f"Service started: {url}", severity="information")
            else:
                # MockService - service state loaded from file but not the actual service
                self.notify(
                    "Service not available (loaded from saved state)",
                    severity="warning",
                )
            self.refresh_services()
        except Exception as e:
            self.notify(f"Failed to start service: {e}", severity="error")

    def action_stop_service(self) -> None:
        """Stop the selected service"""
        service = self._get_selected_service()
        if not service:
            return

        from experimaestro.scheduler.services import ServiceState

        if service.state == ServiceState.STOPPED:
            self.notify("Service is not running", severity="warning")
            return

        try:
            if hasattr(service, "stop"):
                service.stop()
                self.notify(f"Service stopped: {service.id}", severity="information")
            else:
                self.notify("Service does not support stopping", severity="warning")
            self.refresh_services()
        except Exception as e:
            self.notify(f"Failed to stop service: {e}", severity="error")

    def action_copy_url(self) -> None:
        """Copy the service URL to clipboard"""
        service = self._get_selected_service()
        if not service:
            return

        url = getattr(service, "url", None)
        if url:
            try:
                import pyperclip

                pyperclip.copy(url)
                self.notify(f"URL copied: {url}", severity="information")
            except Exception as e:
                self.notify(f"Failed to copy: {e}", severity="error")
        else:
            self.notify("Start the service first to get URL", severity="warning")
