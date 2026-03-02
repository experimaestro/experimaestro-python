"""Global services widget - shows all running services across experiments"""

import logging

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.tui.global_services")


class GlobalServiceSyncs(Vertical):
    """Widget displaying all running services across all experiments

    Shows services from all experiments with their state and URL.
    Sync status is provided by the service's sync_status property
    (e.g., SSHLocalService provides actual sync state).
    """

    BINDINGS = [
        Binding("ctrl+k", "stop_service", "Stop Service"),
        Binding("l", "view_logs", "View Logs", priority=True),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        # service_key -> Service object for quick access
        self._services: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Static("Running Services", classes="section-title")
        yield Static("Loading services...", id="services-loading")
        yield DataTable(id="global-services-table", cursor_type="row")

    def on_mount(self) -> None:
        """Set up the table"""
        table = self.query_one("#global-services-table", DataTable)
        table.add_columns("Experiment", "Service", "State", "Sync", "URL")
        table.cursor_type = "row"
        self.log.info(
            f"GlobalServiceSyncs mounted, state_provider={type(self.state_provider).__name__}"
        )
        # Initial refresh
        self.refresh_services()

    def refresh_services(self) -> None:
        """Refresh the services list from state provider (background)"""
        try:
            self.query_one("#global-services-table", DataTable)
        except Exception:
            return
        # Show loading indicator
        try:
            self.query_one("#services-loading", Static).remove_class("hidden")
        except Exception:
            pass
        self._load_services()

    @work(thread=True, exclusive=True, group="services_load")
    def _load_services(self) -> None:
        """Load services in background thread"""
        from experimaestro.scheduler.services import ServiceState

        try:
            all_services = self.state_provider.get_services()
            running_services = [
                s
                for s in all_services
                if hasattr(s, "state") and s.state != ServiceState.STOPPED
            ]
            self.log.info(
                f"GlobalServiceSyncs._load_services: got {len(running_services)} "
                f"active services (out of {len(all_services)} total)"
            )
        except Exception as e:
            logger.warning(f"Failed to load services: {e}")
            running_services = []

        self.app.call_from_thread(self._on_services_loaded, running_services)

    def _on_services_loaded(self, running_services: list) -> None:
        """Handle loaded services on main thread"""
        # Hide loading indicator
        try:
            self.query_one("#services-loading", Static).add_class("hidden")
        except Exception:
            pass

        try:
            table = self.query_one("#global-services-table", DataTable)
        except Exception:
            return

        # Guard: ensure columns have been added
        if len(table.columns) == 0:
            return

        table.clear()
        self._services.clear()

        for service in running_services:
            service_id = service.id
            state = service.state if hasattr(service, "state") else None
            state_name = state.name if state else "UNKNOWN"

            # Format experiment display: "exp_id (YYYY-MM-DD HH:MM)"
            exp_id = service.experiment_id or "-"
            run_id = service.run_id
            if run_id and run_id != "dry-run" and len(run_id) >= 13:
                # Parse YYYYMMDD_HHMMSS format
                try:
                    timestamp = (
                        f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]} "
                        f"{run_id[9:11]}:{run_id[11:13]}"
                    )
                    exp_display = f"{exp_id} ({timestamp})"
                except (IndexError, ValueError):
                    exp_display = f"{exp_id} ({run_id})" if run_id else exp_id
            else:
                exp_display = f"{exp_id} ({run_id})" if run_id else exp_id

            self.log.info(
                f"  Service: {service_id}, state={state_name}, exp={exp_display}"
            )

            # Get description
            description = ""
            if hasattr(service, "description"):
                try:
                    description = service.description()
                except Exception:
                    description = service_id

            # Get sync status from service (SSHLocalService provides actual status)
            sync_status = service.sync_status or "-"

            # Show error in URL column if there's one, otherwise show URL
            error = service.error
            if error:
                url_or_error = f"⚠ {error}"
            else:
                url_or_error = getattr(service, "url", None) or "-"

            # State icon
            state_icons = {
                "RUNNING": "▶",
                "STOPPED": "⏹",
                "STARTING": "⏳",
                "STOPPING": "⏳",
                "ERROR": "⚠",
            }
            state_icon = state_icons.get(state_name, "?")

            service_key = f"{exp_id}:{service_id}"
            table.add_row(
                exp_display,
                description or service_id,
                f"{state_icon} {state_name}",
                sync_status,
                url_or_error,
                key=service_key,
            )

            # Store service for quick access
            self._services[service_key] = service

        # Update tab title
        self._update_tab_title()

    def _update_tab_title(self) -> None:
        """Update the Services tab title with count"""
        try:
            self.app.update_services_tab_title()
        except Exception:
            pass

    @property
    def running_service_count(self) -> int:
        """Number of running services"""
        from experimaestro.scheduler.services import ServiceState

        try:
            all_services = self.state_provider.get_services()
            return sum(
                1
                for s in all_services
                if hasattr(s, "state") and s.state == ServiceState.RUNNING
            )
        except Exception:
            return 0

    def _get_selected_service(self):
        """Get the currently selected Service object"""
        table = self.query_one("#global-services-table", DataTable)
        if table.cursor_row is not None and table.row_count > 0:
            row_key = list(table.rows.keys())[table.cursor_row]
            if row_key:
                service_key = str(row_key.value)
                return self._services.get(service_key)
        return None

    def action_stop_service(self) -> None:
        """Stop the selected service"""
        from experimaestro.scheduler.services import ServiceState

        service = self._get_selected_service()
        if not service:
            self.notify("No service selected", severity="warning")
            return

        # Convert MockService to live Service (on-demand, cached)
        service = service.to_service()

        if service.state != ServiceState.RUNNING:
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

    def action_view_logs(self) -> None:
        """View service logs"""
        service = self._get_selected_service()
        if not service:
            return

        # Convert to live service to get log paths
        live_service = service.to_service()

        if not live_service.stdout and not live_service.stderr:
            self.notify("Service logs not available", severity="warning")
            return

        stdout_exists = live_service.stdout and live_service.stdout.exists()
        stderr_exists = live_service.stderr and live_service.stderr.exists()
        if not stdout_exists and not stderr_exists:
            self.notify("No log files found", severity="warning")
            return

        from experimaestro.tui.log_viewer import create_service_log_viewer

        sync_func = None
        if self.state_provider.is_remote:
            sync_func = self.state_provider.sync_path

        viewer = create_service_log_viewer(live_service, sync_func)
        self.app.push_screen(viewer)
