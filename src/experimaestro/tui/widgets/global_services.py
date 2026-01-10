"""Global services widget - shows all running services across experiments"""

import logging
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.tui.global_services")


class GlobalServiceSyncs(Vertical):
    """Widget displaying all running services across all experiments

    Shows services from all experiments with their state and URL.
    For remote monitoring, also tracks file synchronization status.
    """

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        # service_key -> {synchronizer, ...} for remote file syncs
        self._syncs: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield Static("Running Services", classes="section-title")
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
        """Refresh the services list from state provider"""

        try:
            table = self.query_one("#global-services-table", DataTable)
        except Exception:
            return

        # Guard: ensure columns have been added
        if len(table.columns) == 0:
            return

        table.clear()

        try:
            # Get all services from state provider
            all_services = self.state_provider.get_services()
            self.log.info(
                f"GlobalServiceSyncs.refresh_services: got {len(all_services)} services"
            )

            for service in all_services:
                service_id = service.id
                state = service.state if hasattr(service, "state") else None
                state_name = state.name if state else "UNKNOWN"
                exp_id = getattr(service, "_experiment_id", None) or "-"
                self.log.info(
                    f"  Service: {service_id}, state={state_name}, exp={exp_id}"
                )

                # Get description
                description = ""
                if hasattr(service, "description"):
                    try:
                        description = service.description()
                    except Exception:
                        description = service_id

                # Get URL
                url = getattr(service, "url", None) or "-"

                # Get sync status for remote monitoring
                sync_status = "-"
                service_key = f"{exp_id}:{service_id}"
                if service_key in self._syncs:
                    sync_info = self._syncs[service_key]
                    synchronizer = sync_info.get("synchronizer")
                    if synchronizer:
                        if synchronizer.syncing:
                            sync_status = "⟳ Syncing"
                        else:
                            sync_status = f"✓ {synchronizer.interval:.0f}s"

                # State icon
                state_icons = {
                    "RUNNING": "▶",
                    "STOPPED": "⏹",
                    "STARTING": "⏳",
                    "STOPPING": "⏳",
                }
                state_icon = state_icons.get(state_name, "?")

                table.add_row(
                    exp_id,
                    description or service_id,
                    f"{state_icon} {state_name}",
                    sync_status,
                    url,
                    key=service_key,
                )

        except Exception as e:
            logger.warning(f"Failed to refresh global services: {e}")

        # Update tab title
        self._update_tab_title()

    def add_service_sync(
        self,
        experiment_id: str,
        service_id: str,
        description: str,
        remote_path: str,
        url: Optional[str] = None,
    ) -> None:
        """Add a new service sync (called from ServicesList for remote monitoring)"""
        from experimaestro.scheduler.remote.adaptive_sync import AdaptiveSynchronizer

        service_key = f"{experiment_id}:{service_id}"

        # Don't restart if already syncing
        if service_key in self._syncs:
            return

        if not self.state_provider.is_remote:
            return

        sync_name = f"service:{description}"

        synchronizer = AdaptiveSynchronizer(
            sync_func=self.state_provider.sync_path,
            remote_path=remote_path,
            name=sync_name,
            on_sync_start=lambda sk=service_key: self.app.call_from_thread(
                self._on_sync_start, sk
            ),
            on_sync_complete=lambda p, sk=service_key: self.app.call_from_thread(
                self._on_sync_complete, sk, p
            ),
        )

        self._syncs[service_key] = {
            "synchronizer": synchronizer,
            "experiment_id": experiment_id,
            "service_id": service_id,
            "description": description,
            "remote_path": remote_path,
            "url": url or "-",
        }

        synchronizer.start()
        logger.info(f"Started global sync for {service_key}: {remote_path}")

        # Refresh to show sync status
        self.refresh_services()

    def stop_service_sync(self, experiment_id: str, service_id: str) -> None:
        """Stop a service sync (called when service is STOPPED)"""
        service_key = f"{experiment_id}:{service_id}"

        if service_key in self._syncs:
            self._syncs[service_key]["synchronizer"].stop()
            del self._syncs[service_key]
            logger.info(f"Stopped global sync for {service_key}")
            self.refresh_services()

    def _update_tab_title(self) -> None:
        """Update the Services tab title with count"""
        try:
            self.app.update_services_tab_title()
        except Exception:
            pass

    def has_sync(self, experiment_id: str, service_id: str) -> bool:
        """Check if a sync exists for this service"""
        return f"{experiment_id}:{service_id}" in self._syncs

    def get_sync_status(self, experiment_id: str, service_id: str) -> Optional[str]:
        """Get sync status string for display"""
        service_key = f"{experiment_id}:{service_id}"
        if service_key not in self._syncs:
            return None

        sync_info = self._syncs[service_key]
        synchronizer = sync_info["synchronizer"]

        if synchronizer.syncing:
            return "⟳"
        else:
            return f"✓ {synchronizer.interval:.0f}s"

    def _on_sync_start(self, service_key: str) -> None:
        """Handle sync start"""
        self.refresh_services()

    def _on_sync_complete(self, service_key: str, local_path: Path) -> None:
        """Handle sync complete"""
        self.refresh_services()

    def on_unmount(self) -> None:
        """Stop all syncs when app closes"""
        for service_key, info in list(self._syncs.items()):
            if "synchronizer" in info:
                info["synchronizer"].stop()
        self._syncs.clear()

    @property
    def sync_count(self) -> int:
        """Number of active syncs (for backward compatibility)"""
        return len(self._syncs)

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
