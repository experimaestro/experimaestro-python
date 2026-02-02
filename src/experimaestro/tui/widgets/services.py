"""Services list widget for the TUI"""

from typing import Optional
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider


class ServicesList(Vertical):
    """Widget displaying services for selected experiment

    Services are retrieved from StateProvider.get_services() which
    abstracts away whether services are live (from scheduler) or recreated
    from database state_dict. The UI treats all services uniformly.

    For remote monitoring, service syncs are managed globally by GlobalServiceSyncs.
    """

    BINDINGS = [
        Binding("ctrl+s", "start_service", "Start"),
        Binding("ctrl+k", "stop_service", "Stop"),
        Binding("l", "view_logs", "View Logs", priority=True),
        Binding("u", "copy_url", "Copy URL", show=False),
    ]

    # State icons for display
    STATE_ICONS = {
        "STOPPED": "⏹",
        "STARTING": "⏳",
        "RUNNING": "▶",
        "STOPPING": "⏳",
        "ERROR": "⚠",
    }

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.current_experiment: Optional[str] = None
        self._services: dict = {}  # service_id -> Service object

    def compose(self) -> ComposeResult:
        yield Static(
            "Services",
            id="services-header",
            classes="section-title",
        )
        yield Static("Loading services...", id="services-loading", classes="hidden")
        yield DataTable(id="services-table", cursor_type="row")

    def on_mount(self) -> None:
        """Set up the services table"""
        table = self.query_one("#services-table", DataTable)
        table.add_columns("ID", "Description", "State", "Sync", "URL")
        table.cursor_type = "row"

    def set_experiment(self, experiment_id: Optional[str]) -> None:
        """Set the current experiment and refresh services"""
        self.current_experiment = experiment_id

        # Clear and show loading for remote
        if self.state_provider.is_remote:
            table = self.query_one("#services-table", DataTable)
            table.clear()
            self._services = {}
            self.query_one("#services-loading", Static).remove_class("hidden")

        # Load in background
        self._load_services(experiment_id)

    @work(thread=True, exclusive=True, group="services_load")
    def _load_services(self, experiment_id: Optional[str]) -> None:
        """Load services in background thread"""
        if not experiment_id:
            self.app.call_from_thread(self._on_services_loaded, [])
            return

        services = self.state_provider.get_services(experiment_id)
        self.app.call_from_thread(self._on_services_loaded, services)

    def _on_services_loaded(self, services: list) -> None:
        """Handle loaded services on main thread"""
        self.query_one("#services-loading", Static).add_class("hidden")
        self._refresh_services_with_data(services)

    def refresh_services(self) -> None:
        """Refresh the services display from current services

        Reloads services from state provider to pick up newly added services.
        This is called when ServiceAddedEvent is received.
        """
        if not self.current_experiment:
            return

        # Reload services from state provider to pick up new services
        self._load_services(self.current_experiment)

    def _refresh_services_with_data(self, services: list) -> None:
        """Refresh the services display with provided data (initial load)

        Stores services in self._services and refreshes display.
        """
        import logging

        logger = logging.getLogger("xpm.tui.services")

        self._services = {}
        for service in services:
            self._services[service.id] = service

        logger.debug(
            f"refresh_services got {len(services)} services: "
            f"{[(s.id, getattr(s, 'url', None)) for s in services]}"
        )

        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the table display from current self._services

        Called on initial load and when sync status changes.
        """
        table = self.query_one("#services-table", DataTable)
        table.clear()

        for service_id, service in self._services.items():
            state_name = service.state.name if hasattr(service, "state") else "UNKNOWN"
            state_icon = self.STATE_ICONS.get(state_name, "?")
            description = (
                service.description() if hasattr(service, "description") else ""
            )

            # Get sync status from service (SSHLocalService provides actual status)
            sync_status = service.sync_status or "-"

            # Show error in URL column if there's one, otherwise show URL
            error = service.error
            if error:
                url_or_error = f"⚠ {error}"
            else:
                url_or_error = getattr(service, "url", None) or "-"

            table.add_row(
                service_id,
                description,
                f"{state_icon} {state_name}",
                sync_status,
                url_or_error,
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

    def _refresh_all_services(self) -> None:
        """Refresh both this widget and the global services widget"""
        self.refresh_services()
        # Also refresh global services to update tab title
        try:
            from experimaestro.tui.widgets.global_services import GlobalServiceSyncs

            global_services = self.app.query_one(GlobalServiceSyncs)
            global_services.refresh_services()
        except Exception:
            pass

    def action_start_service(self) -> None:
        """Start the selected service"""
        service = self._get_selected_service()
        if not service:
            return

        # Set STARTING state immediately and refresh UI
        service.set_starting()
        self.notify("Starting service...", severity="information")
        self._refresh_all_services()

        # Set up callback for sync status changes (for SSH remote services)
        # The callback refreshes both widgets when sync status changes
        if hasattr(service, "set_status_change_callback"):
            service.set_status_change_callback(
                lambda: self.app.call_from_thread(self._refresh_all_services)
            )

        # Start service in background thread so UI can update
        self._start_service_worker(service)

    @work(thread=True, exclusive=True, group="service_start")
    def _start_service_worker(self, service) -> None:
        """Start service in background thread"""
        import logging

        logger = logging.getLogger("xpm.tui.services")

        # Convert MockService to live Service (on-demand, cached)
        live_service = service.to_service()

        logger.info(
            f"Starting service {live_service.id} (type={type(live_service).__name__}, "
            f"has_get_url={hasattr(live_service, 'get_url')}, is_live={self.state_provider.is_live})"
        )

        try:
            if hasattr(live_service, "get_url"):
                url = live_service.get_url()
                logger.info(
                    f"Service started, url={url}, service.url={live_service.url}"
                )
                self.app.call_from_thread(
                    self.notify, f"Service started: {url}", severity="information"
                )
            else:
                # Service recreation failed - error was set by to_service()
                error_msg = service.error or "Service cannot be started"
                self.app.call_from_thread(self.notify, error_msg, severity="warning")
            self.app.call_from_thread(self._refresh_all_services)
        except Exception as e:
            # Set error on service so it shows in URL column
            error_msg = str(e)
            service.set_error(error_msg)
            self.app.call_from_thread(self._refresh_all_services)
            self.app.call_from_thread(
                self.notify, f"Failed to start service: {e}", severity="error"
            )

    def action_stop_service(self) -> None:
        """Stop the selected service"""
        service = self._get_selected_service()
        if not service:
            return

        # Convert MockService to live Service (on-demand, cached)
        service = service.to_service()

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
            self._refresh_all_services()
        except Exception as e:
            self.notify(f"Failed to stop service: {e}", severity="error")

    def action_copy_url(self) -> None:
        """Copy the service URL to clipboard"""
        service = self._get_selected_service()
        if not service:
            return

        url = getattr(service, "url", None)
        if url:
            from experimaestro.tui.clipboard import copy

            if copy(url):
                self.notify(f"URL copied: {url}", severity="information")
            else:
                self.notify("Failed to copy URL", severity="error")
        else:
            self.notify("Start the service first to get URL", severity="warning")

    def action_view_logs(self) -> None:
        """View service logs"""
        service = self._get_selected_service()
        if not service:
            return

        # Convert to live service to get log paths
        live_service = service.to_service()

        if not live_service.stdout or not live_service.stderr:
            self.notify("Service logs not available", severity="warning")
            return

        if not live_service.stdout.exists() and not live_service.stderr.exists():
            self.notify("No log files found", severity="warning")
            return

        from experimaestro.tui.log_viewer import create_service_log_viewer

        sync_func = None
        if self.state_provider.is_remote:
            sync_func = self.state_provider.sync_path

        viewer = create_service_log_viewer(live_service, sync_func)
        self.app.push_screen(viewer)
