"""Runs list widget for the TUI"""

from datetime import datetime
from typing import Optional
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tui.utils import format_duration
from experimaestro.tui.messages import RunSelected


class RunsList(Widget):
    """Widget displaying runs for selected experiment"""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=False),
        Binding("enter", "select_run", "Select", show=False),
    ]

    visible: reactive[bool] = reactive(False)

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.experiment_id: Optional[str] = None
        self.current_run_id: Optional[str] = None
        self.runs = []

    def compose(self) -> ComposeResult:
        with Vertical(id="runs-container"):
            yield Static("", id="runs-title")
            yield DataTable(id="runs-table", cursor_type="row")

    def on_mount(self) -> None:
        """Initialize the runs table"""
        table = self.query_one("#runs-table", DataTable)
        table.add_column("Run ID", key="run_id")
        table.add_column("Status", key="status", width=12)
        table.add_column("Host", key="host")
        table.add_column("Jobs", key="jobs", width=10)
        table.add_column("Started", key="started")
        table.add_column("Duration", key="duration", width=12)

    def watch_visible(self, visible: bool) -> None:
        """Show/hide the runs list"""
        if visible:
            self.display = True
            self.remove_class("hidden")
        else:
            self.display = False
            self.add_class("hidden")

    def set_experiment(self, experiment_id: str, current_run_id: Optional[str]) -> None:
        """Set the experiment and refresh runs"""
        self.experiment_id = experiment_id
        self.current_run_id = current_run_id
        self.query_one("#runs-title", Static).update(
            f"[bold]Runs for {experiment_id}[/bold]"
        )
        self.refresh_runs()
        self.visible = True
        # Focus the runs table
        self.query_one("#runs-table", DataTable).focus()

    def refresh_runs(self) -> None:
        """Refresh the runs list"""
        table = self.query_one("#runs-table", DataTable)
        table.clear()

        if not self.experiment_id:
            return

        self.runs = self.state_provider.get_experiment_runs(self.experiment_id)

        for run in self.runs:
            # Format status with icon
            if run.status == "active":
                status = "▶ Active"
            elif run.status == "completed":
                status = "✓ Done"
            elif run.status == "failed":
                status = "❌ Failed"
            else:
                status = run.status or "-"

            # Mark current run
            run_id_display = run.run_id
            if run.run_id == self.current_run_id:
                run_id_display = f"★ {run.run_id}"

            # Format jobs
            jobs_text = f"{run.finished_jobs}/{run.total_jobs}"
            if run.failed_jobs > 0:
                jobs_text += f" ({run.failed_jobs}✗)"

            # Format hostname
            hostname = run.hostname or "-"

            # Format started time (datetime object)
            started = "-"
            if run.started_at:
                started = run.started_at.strftime("%Y-%m-%d %H:%M")

            # Calculate duration
            duration = "-"
            if run.started_at:
                if run.ended_at:
                    elapsed = (run.ended_at - run.started_at).total_seconds()
                else:
                    elapsed = (datetime.now() - run.started_at).total_seconds()
                duration = format_duration(elapsed)

            table.add_row(
                run_id_display,
                status,
                hostname,
                jobs_text,
                started,
                duration,
                key=run.run_id,
            )

    def _get_selected_run_id(self) -> Optional[str]:
        """Get the run_id from the currently selected row"""
        table = self.query_one("#runs-table", DataTable)
        if table.cursor_row is None:
            return None
        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            return str(row_key.value)
        return None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle run selection"""
        if event.row_key and self.experiment_id:
            run_id = str(event.row_key.value)
            is_current = run_id == self.current_run_id
            self.post_message(RunSelected(self.experiment_id, run_id, is_current))
            self.visible = False

    def action_select_run(self) -> None:
        """Select the highlighted run"""
        run_id = self._get_selected_run_id()
        if run_id and self.experiment_id:
            is_current = run_id == self.current_run_id
            self.post_message(RunSelected(self.experiment_id, run_id, is_current))
            self.visible = False

    def action_go_back(self) -> None:
        """Hide the runs list"""
        self.visible = False
        # Return focus to experiments table
        try:
            self.app.query_one("#experiments-table", DataTable).focus()
        except Exception:
            pass
