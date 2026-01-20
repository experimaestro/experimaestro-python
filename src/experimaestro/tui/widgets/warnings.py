"""Warnings widget - shows all unresolved warnings"""

import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.tui.warnings")


class WarningsTab(Vertical):
    """Widget displaying all unresolved warnings

    Shows warnings that need user attention with their description and available actions.
    Users can select a warning and execute an action on it.
    """

    BINDINGS = [
        Binding("enter", "handle_warning", "Handle Warning"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        # warning_key -> WarningEvent object for quick access
        self._warnings: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Static("Unresolved Warnings", classes="section-title")
        yield DataTable(id="warnings-table", cursor_type="row")

    def on_mount(self) -> None:
        """Set up the table"""
        table = self.query_one("#warnings-table", DataTable)
        table.add_columns("Experiment", "Type", "Description", "Actions")
        table.cursor_type = "row"
        self.log.info(
            f"WarningsTab mounted, state_provider={type(self.state_provider).__name__}"
        )
        # Initial refresh
        self.refresh_warnings()

    def refresh_warnings(self) -> None:
        """Refresh the warnings list from state provider"""
        try:
            table = self.query_one("#warnings-table", DataTable)
        except Exception:
            return

        # Guard: ensure columns have been added
        if len(table.columns) == 0:
            return

        table.clear()

        try:
            # Get all unresolved warnings from state provider
            warnings = self.state_provider.get_unresolved_warnings()
            self.log.info(f"WarningsTab.refresh_warnings: got {len(warnings)} warnings")

            # Clear warnings dict before repopulating
            self._warnings.clear()

            for warning in warnings:
                warning_key = warning.warning_key
                experiment_id = warning.experiment_id or "-"
                run_id = warning.run_id

                # Format experiment display
                if run_id and run_id != "dry-run" and len(run_id) >= 13:
                    # Parse YYYYMMDD_HHMMSS format
                    try:
                        timestamp = (
                            f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]} "
                            f"{run_id[9:11]}:{run_id[11:13]}"
                        )
                        exp_display = f"{experiment_id} ({timestamp})"
                    except (IndexError, ValueError):
                        exp_display = (
                            f"{experiment_id} ({run_id})" if run_id else experiment_id
                        )
                else:
                    exp_display = (
                        f"{experiment_id} ({run_id})" if run_id else experiment_id
                    )

                # Extract warning type from context (if available)
                warning_type = warning.context.get("title", "Warning")

                # Truncate description for table display
                description = warning.description.split("\n")[0]  # First line only
                if len(description) > 60:
                    description = description[:57] + "..."

                # Count available actions
                action_count = len(warning.actions)
                actions_display = f"{action_count} action(s)"

                # Severity icon
                severity_icons = {
                    "info": "ℹ",
                    "warning": "⚠",
                    "error": "⛔",
                }
                severity_icon = severity_icons.get(warning.severity, "?")

                table.add_row(
                    exp_display,
                    f"{severity_icon} {warning_type}",
                    description,
                    actions_display,
                    key=warning_key,
                )

                # Store warning for quick access
                self._warnings[warning_key] = warning

        except Exception as e:
            logger.warning(f"Failed to refresh warnings: {e}")

        # Update tab title
        self._update_tab_title()

    def _update_tab_title(self) -> None:
        """Update the Warnings tab title with count"""
        try:
            self.app.update_warnings_tab_title()
        except Exception:
            pass

    @property
    def warning_count(self) -> int:
        """Number of unresolved warnings"""
        try:
            return len(self.state_provider.get_unresolved_warnings())
        except Exception:
            return 0

    def _get_selected_warning(self):
        """Get the currently selected WarningEvent object"""
        table = self.query_one("#warnings-table", DataTable)
        if table.cursor_row is not None and table.row_count > 0:
            row_key = list(table.rows.keys())[table.cursor_row]
            if row_key:
                warning_key = str(row_key.value)
                return self._warnings.get(warning_key)
        return None

    def action_handle_warning(self) -> None:
        """Handle the selected warning by showing action dialog"""
        from experimaestro.tui.dialogs import WarningDialog

        warning = self._get_selected_warning()
        if not warning:
            self.notify("No warning selected", severity="warning")
            return

        # Extract title from context
        title = warning.context.get("title", "Warning")

        def handle_response(action_key: str | None) -> None:
            if action_key is None:
                # User dismissed the warning
                return

            # Execute the action via state provider
            try:
                self.state_provider.execute_warning_action(
                    warning_key=warning.warning_key,
                    action_key=action_key,
                    experiment_id=warning.experiment_id,
                    run_id=warning.run_id,
                )
                self.notify(
                    f"Action '{action_key}' completed successfully",
                    severity="information",
                    timeout=5,
                )
                # Refresh warnings list to remove resolved warning
                self.refresh_warnings()
            except Exception as e:
                self.log(f"Failed to execute action '{action_key}': {e}")
                self.notify(
                    f"Action '{action_key}' failed: {e}",
                    severity="error",
                    timeout=10,
                )

        # Show warning dialog
        self.app.push_screen(
            WarningDialog(
                warning_key=warning.warning_key,
                title=title,
                description=warning.description,
                actions=warning.actions,
                severity=warning.severity,
            ),
            handle_response,
        )

    def action_refresh(self) -> None:
        """Manually refresh the warnings list"""
        self.refresh_warnings()
