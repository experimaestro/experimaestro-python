"""Services list widget for the TUI"""

from pathlib import Path
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

    def _get_global_services(self):
        """Get the global services sync widget"""
        from experimaestro.tui.widgets.global_services import GlobalServiceSyncs

        try:
            return self.app.query_one(GlobalServiceSyncs)
        except Exception:
            return None

    def _start_synchronizer_for_service(self, service) -> None:
        """Register a service with the global sync manager"""
        import logging

        logger = logging.getLogger("xpm.tui.services")
        service_id = service.id

        if not self.state_provider.is_remote:
            return

        if not self.current_experiment:
            return

        # Check if service has paths in state_dict that need syncing
        state_dict = getattr(service, "_state_dict_data", None)
        if state_dict is None and hasattr(service, "state_dict"):
            try:
                state_dict = service.state_dict()
            except Exception:
                logger.debug(f"Service {service_id}: state_dict() failed")
                return

        if not state_dict:
            logger.info(f"Service {service_id}: no state_dict")
            return

        # Find paths in state_dict
        paths_to_sync = self._extract_paths(state_dict)
        if not paths_to_sync:
            logger.info(f"Service {service_id}: no paths in state_dict: {state_dict}")
            return

        logger.info(f"Service {service_id}: found paths to sync: {paths_to_sync}")

        # Get service description and URL
        description = (
            service.description() if hasattr(service, "description") else service_id
        )
        url = getattr(service, "url", None)

        # Register with global sync manager
        global_services = self._get_global_services()
        if global_services:
            global_services.add_service_sync(
                experiment_id=self.current_experiment,
                service_id=service_id,
                description=description,
                remote_path=paths_to_sync[0],
                url=url,
            )

    def _extract_paths(self, state_dict: dict) -> list[str]:
        """Extract path strings from a service state_dict"""
        from pathlib import PosixPath, WindowsPath

        paths = []

        def find_paths(d):
            if isinstance(d, (Path, PosixPath, WindowsPath)):
                # Direct Path object
                paths.append(str(d))
            elif isinstance(d, dict):
                if "__path__" in d:
                    # Serialized path format
                    paths.append(d["__path__"])
                else:
                    for v in d.values():
                        find_paths(v)
            elif isinstance(d, (list, tuple)):
                for item in d:
                    find_paths(item)

        find_paths(state_dict)
        return paths

    def refresh_services(self) -> None:
        """Refresh the services list from state provider

        For remote providers, this runs in background. For local, it's synchronous.
        """
        if not self.current_experiment:
            return

        if self.state_provider.is_remote:
            self._load_services(self.current_experiment)
        else:
            services = self.state_provider.get_services(self.current_experiment)
            self._refresh_services_with_data(services)

    def _refresh_services_with_data(self, services: list) -> None:
        """Refresh the services display with provided data"""
        import logging

        logger = logging.getLogger("xpm.tui.services")

        table = self.query_one("#services-table", DataTable)
        table.clear()
        self._services = {}

        global_services = self._get_global_services()

        logger.debug(
            f"refresh_services got {len(services)} services: "
            f"{[(s.id, getattr(s, 'url', None)) for s in services]}"
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

            # Get sync status from global services
            sync_status = "-"
            if global_services and self.current_experiment:
                status = global_services.get_sync_status(
                    self.current_experiment, service_id
                )
                if status:
                    sync_status = status

            table.add_row(
                service_id,
                description,
                f"{state_icon} {state_name}",
                sync_status,
                url,
                key=service_id,
            )

            # Start synchronizer for running services with paths (remote only)
            if state_name == "RUNNING":
                self._start_synchronizer_for_service(service)
            elif (
                state_name == "STOPPED" and global_services and self.current_experiment
            ):
                # Stop sync when service is explicitly stopped
                global_services.stop_service_sync(self.current_experiment, service_id)

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
        import logging

        logger = logging.getLogger("xpm.tui.services")

        service = self._get_selected_service()
        if not service:
            return

        logger.info(
            f"Starting service {service.id} (type={type(service).__name__}, "
            f"has_get_url={hasattr(service, 'get_url')}, is_live={self.state_provider.is_live})"
        )

        try:
            if hasattr(service, "get_url"):
                url = service.get_url()
                logger.info(f"Service started, url={url}, service.url={service.url}")
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
