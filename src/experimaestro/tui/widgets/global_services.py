"""Global services sync widget - tracks all running service syncs across experiments"""

import logging
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.tui.global_services")


class GlobalServiceSyncs(Vertical):
    """Widget displaying all active service syncs across all experiments

    This widget maintains service synchronizers globally - they persist
    even when navigating between experiments. Syncs only stop when:
    - The service state becomes STOPPED
    - The app is closed
    """

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        # service_key -> {synchronizer, experiment_id, service_id, description}
        self._syncs: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield Static("Active Service Syncs", classes="section-title")
        yield DataTable(id="global-services-table", cursor_type="row")

    def on_mount(self) -> None:
        """Set up the table"""
        table = self.query_one("#global-services-table", DataTable)
        table.add_columns("Experiment", "Service", "Status", "Interval", "URL")
        table.cursor_type = "row"

    def add_service_sync(
        self,
        experiment_id: str,
        service_id: str,
        description: str,
        remote_path: str,
        url: Optional[str] = None,
    ) -> None:
        """Add a new service sync (called from ServicesList)"""
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

        # Update table and tab title
        self._refresh_table()
        self._update_tab_title()

    def stop_service_sync(self, experiment_id: str, service_id: str) -> None:
        """Stop a service sync (called when service is STOPPED)"""
        service_key = f"{experiment_id}:{service_id}"

        if service_key in self._syncs:
            self._syncs[service_key]["synchronizer"].stop()
            del self._syncs[service_key]
            logger.info(f"Stopped global sync for {service_key}")
            self._refresh_table()
            self._update_tab_title()

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
        self._refresh_table()

    def _on_sync_complete(self, service_key: str, local_path: Path) -> None:
        """Handle sync complete"""
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table display"""
        try:
            table = self.query_one("#global-services-table", DataTable)
            table.clear()

            for service_key, info in self._syncs.items():
                synchronizer = info["synchronizer"]

                if synchronizer.syncing:
                    status = "⟳ Syncing"
                else:
                    status = "✓ Idle"

                interval = f"{synchronizer.interval:.0f}s"

                table.add_row(
                    info["experiment_id"],
                    info["description"],
                    status,
                    interval,
                    info.get("url", "-"),
                    key=service_key,
                )
        except Exception as e:
            logger.warning(f"Failed to refresh global services table: {e}")

    def on_unmount(self) -> None:
        """Stop all syncs when app closes"""
        for service_key, info in list(self._syncs.items()):
            info["synchronizer"].stop()
        self._syncs.clear()

    @property
    def sync_count(self) -> int:
        """Number of active syncs"""
        return len(self._syncs)
