"""Experiments list widget for the TUI"""

from datetime import datetime
from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import DataTable, Label
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tui.utils import format_duration
from experimaestro.tui.messages import (
    ExperimentSelected,
    ExperimentDeselected,
    DeleteExperimentRequest,
    KillExperimentRequest,
    ShowRunsRequest,
)


class ExperimentsList(Widget):
    """Widget displaying list of experiments"""

    BINDINGS = [
        Binding("d", "show_runs", "Runs"),
        Binding("ctrl+d", "delete_experiment", "Delete", show=False),
        Binding("k", "kill_experiment", "Kill", show=False),
        Binding("S", "sort_by_status", "Sort ⚑", show=False),
        Binding("D", "sort_by_date", "Sort Date", show=False),
    ]

    current_experiment: reactive[Optional[str]] = reactive(None)
    collapsed: reactive[bool] = reactive(False)

    # Track current sort state
    _sort_column: Optional[str] = None
    _sort_reverse: bool = False

    # Status sort order (for sorting by status)
    STATUS_ORDER = {
        "failed": 0,  # Failed experiments first (need attention)
        "running": 1,  # Running experiments
        "done": 2,  # Completed experiments
        "empty": 3,  # Empty experiments last
    }

    # Column key to display name mapping
    COLUMN_LABELS = {
        "id": "ID",
        "run": "Run",
        "runs": "#",
        "host": "Host",
        "jobs": "Jobs",
        "status": "Status",
        "started": "Started",
        "duration": "Duration",
    }

    # Columns that support sorting (column key -> sort column name)
    SORTABLE_COLUMNS = {
        "status": "status",
        "started": "started",
    }

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.experiments = []

    def _get_selected_experiment_id(self) -> Optional[str]:
        """Get the experiment ID from the currently selected row"""
        table = self.query_one("#experiments-table", DataTable)
        if table.cursor_row is None:
            return None
        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            return str(row_key.value)
        return None

    def action_delete_experiment(self) -> None:
        """Request to delete the selected experiment"""
        exp_id = self._get_selected_experiment_id()
        if exp_id:
            self.post_message(DeleteExperimentRequest(exp_id))

    def action_kill_experiment(self) -> None:
        """Request to kill all running jobs in the selected experiment"""
        exp_id = self._get_selected_experiment_id()
        if exp_id:
            self.post_message(KillExperimentRequest(exp_id))

    def action_show_runs(self) -> None:
        """Show runs for the selected experiment"""
        exp_id = self._get_selected_experiment_id()
        if exp_id:
            # Get current run_id for this experiment
            exp_info = next(
                (exp for exp in self.experiments if exp.experiment_id == exp_id),
                None,
            )
            current_run_id = (
                getattr(exp_info, "current_run_id", None) if exp_info else None
            )
            self.post_message(ShowRunsRequest(exp_id, current_run_id))

    def action_sort_by_status(self) -> None:
        """Sort experiments by status"""
        if self._sort_column == "status":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "status"
            self._sort_reverse = False
        self._update_column_headers()
        self.refresh_experiments()
        order = "desc" if self._sort_reverse else "asc"
        self.notify(f"Sorted by status ({order})", severity="information")

    def action_sort_by_date(self) -> None:
        """Sort experiments by start date"""
        if self._sort_column == "started":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "started"
            self._sort_reverse = True  # Default to newest first for date
        self._update_column_headers()
        self.refresh_experiments()
        order = "newest first" if self._sort_reverse else "oldest first"
        self.notify(f"Sorted by date ({order})", severity="information")

    def _get_status_category(self, exp) -> str:
        """Get status category for an experiment (for sorting)"""
        failed = exp.failed_jobs
        total = exp.total_jobs
        finished = exp.finished_jobs

        if failed > 0:
            return "failed"
        elif finished == total and total > 0:
            return "done"
        elif finished < total:
            return "running"
        else:
            return "empty"

    def _get_status_sort_key(self, exp):
        """Get sort key for an experiment based on status"""
        status_category = self._get_status_category(exp)
        status_order = self.STATUS_ORDER.get(status_category, 99)
        # Secondary sort by failed count (more failures first)
        failed_count = exp.failed_jobs if status_category == "failed" else 0
        return (status_order, -failed_count)

    def _update_column_headers(self) -> None:
        """Update column headers with sort indicators"""
        table = self.query_one("#experiments-table", DataTable)
        for column in table.columns.values():
            col_key = str(column.key.value) if column.key else None
            if col_key and col_key in self.COLUMN_LABELS:
                label = self.COLUMN_LABELS[col_key]
                sort_col = self.SORTABLE_COLUMNS.get(col_key)
                if sort_col and self._sort_column == sort_col:
                    # Add sort indicator
                    indicator = "▼" if self._sort_reverse else "▲"
                    new_label = f"{label} {indicator}"
                else:
                    new_label = label
                column.label = new_label

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Handle column header click for sorting"""
        col_key = str(event.column_key.value) if event.column_key else None
        if col_key and col_key in self.SORTABLE_COLUMNS:
            sort_col = self.SORTABLE_COLUMNS[col_key]
            if self._sort_column == sort_col:
                self._sort_reverse = not self._sort_reverse
            else:
                self._sort_column = sort_col
                # Default to reverse for date (newest first)
                self._sort_reverse = sort_col == "started"
            self._update_column_headers()
            self.refresh_experiments()

    def compose(self) -> ComposeResult:
        # Collapsed header (hidden initially)
        with Horizontal(id="collapsed-header", classes="hidden"):
            yield Label("", id="collapsed-experiment-info")

        # Full experiments table
        with Container(id="experiments-table-container"):
            yield Label("Experiments", classes="section-title")
            yield DataTable(id="experiments-table", cursor_type="row")

    def on_mount(self) -> None:
        """Initialize the experiments table"""
        # Start expanded
        self.add_class("expanded")

        table = self.query_one("#experiments-table", DataTable)
        table.add_column("ID", key="id")
        table.add_column("Run", key="run")
        table.add_column("#", key="runs", width=3)
        table.add_column("Host", key="host")
        table.add_column("Jobs", key="jobs")
        table.add_column("Status", key="status")
        table.add_column("Started", key="started")
        table.add_column("Duration", key="duration")
        self.refresh_experiments()

        # If there's only one experiment, automatically select it
        if len(self.experiments) == 1:
            exp = self.experiments[0]
            exp_id = exp.experiment_id
            run_id = getattr(exp, "current_run_id", None)
            self.current_experiment = exp_id
            self.collapse_to_experiment(exp_id)
            self.post_message(ExperimentSelected(exp_id, run_id))

    def refresh_experiments(self) -> None:  # noqa: C901
        """Refresh the experiments list from state provider"""
        # Guard: ensure the table is mounted before querying
        try:
            table = self.query_one("#experiments-table", DataTable)
        except Exception:
            # Widget not yet fully composed, will be called again from on_mount
            return

        # Guard: ensure columns have been added (on_mount may not have run yet)
        if len(table.columns) == 0:
            return

        self.log.info(
            f"State provider: {type(self.state_provider).__name__}, is_live={self.state_provider.is_live}"
        )

        try:
            self.experiments = self.state_provider.get_experiments()
            self.log.debug(
                f"Refreshing experiments: found {len(self.experiments)} experiments"
            )
        except Exception as e:
            self.log.error(f"ERROR refreshing experiments: {e}")
            import traceback

            self.log.error(traceback.format_exc())
            self.experiments = []
            return

        # Sort experiments based on selected column
        experiments_sorted = list(self.experiments)
        if self._sort_column == "status":
            experiments_sorted.sort(
                key=self._get_status_sort_key,
                reverse=self._sort_reverse,
            )
        elif self._sort_column == "started":
            # Sort by started time (experiments without start time go to end)
            experiments_sorted.sort(
                key=lambda e: e.started_at or datetime.min,
                reverse=self._sort_reverse,
            )
        # Default: no sorting, use order from state provider

        # Get existing row keys
        existing_keys = set(table.rows.keys())
        current_exp_ids = set()

        # Check if we need to rebuild (sort order may have changed)
        current_order = [e.experiment_id for e in experiments_sorted]
        existing_order = [str(k.value) for k in table.rows.keys()]
        needs_rebuild = current_order != existing_order

        # Build row data for all experiments
        rows_data = {}
        for exp in experiments_sorted:
            exp_id = exp.experiment_id
            current_exp_ids.add(exp_id)
            total = exp.total_jobs
            finished = exp.finished_jobs
            failed = exp.failed_jobs

            # Determine status
            if failed > 0:
                status = f"❌ {failed} failed"
            elif finished == total and total > 0:
                status = "✓ Done"
            elif finished < total:
                status = f"▶ {finished}/{total}"
            else:
                status = "Empty"

            jobs_text = f"{finished}/{total}"

            # Format started time
            if exp.started_at:
                started = exp.started_at.strftime("%Y-%m-%d %H:%M")
            else:
                started = "-"

            # Calculate duration
            duration = "-"
            if exp.started_at:
                if exp.ended_at:
                    elapsed = (exp.ended_at - exp.started_at).total_seconds()
                else:
                    # Still running - show elapsed time
                    elapsed = (datetime.now() - exp.started_at).total_seconds()
                # Format duration
                duration = format_duration(elapsed)

            # Get hostname (may be None for older experiments)
            hostname = getattr(exp, "hostname", None) or "-"

            # Get run_id
            run_id = getattr(exp, "current_run_id", None) or "-"

            # Get runs count for this experiment (only for offline monitoring)
            runs_count = "-"
            if not self.state_provider.is_live:
                try:
                    runs = self.state_provider.get_experiment_runs(exp_id)
                    runs_count = str(len(runs))
                except Exception as e:
                    self.log.error(f"Error getting runs for {exp_id}: {e}")
                    import traceback

                    self.log.error(traceback.format_exc())

            rows_data[exp_id] = (
                exp_id,
                run_id,
                runs_count,
                hostname,
                jobs_text,
                status,
                started,
                duration,
            )

        if needs_rebuild:
            # Full rebuild needed - save selection, clear, rebuild
            selected_key = None
            if table.cursor_row is not None and table.row_count > 0:
                try:
                    row_keys = list(table.rows.keys())
                    if table.cursor_row < len(row_keys):
                        selected_key = str(row_keys[table.cursor_row].value)
                except (IndexError, KeyError):
                    pass

            table.clear()
            new_cursor_row = None
            for idx, exp in enumerate(experiments_sorted):
                exp_id = exp.experiment_id
                table.add_row(*rows_data[exp_id], key=exp_id)
                if selected_key == exp_id:
                    new_cursor_row = idx

            if new_cursor_row is not None and table.row_count > 0:
                table.move_cursor(row=new_cursor_row)
        else:
            # Update cells in place (no reordering needed)
            for exp_id, row_data in rows_data.items():
                if exp_id in existing_keys:
                    (
                        _,
                        run_id,
                        runs_count,
                        hostname,
                        jobs_text,
                        status,
                        started,
                        duration,
                    ) = row_data
                    table.update_cell(exp_id, "id", exp_id, update_width=True)
                    table.update_cell(exp_id, "run", run_id, update_width=True)
                    table.update_cell(exp_id, "runs", runs_count, update_width=True)
                    table.update_cell(exp_id, "host", hostname, update_width=True)
                    table.update_cell(exp_id, "jobs", jobs_text, update_width=True)
                    table.update_cell(exp_id, "status", status, update_width=True)
                    table.update_cell(exp_id, "started", started, update_width=True)
                    table.update_cell(exp_id, "duration", duration, update_width=True)
                else:
                    table.add_row(*row_data, key=exp_id)

            # Remove rows for experiments that no longer exist
            for old_exp_id in existing_keys - current_exp_ids:
                table.remove_row(old_exp_id)

        # Update collapsed header if viewing an experiment
        if self.collapsed and self.current_experiment:
            self._update_collapsed_header(self.current_experiment)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle experiment selection"""
        if event.row_key:
            exp_id = str(event.row_key.value)
            self.current_experiment = exp_id
            self.collapse_to_experiment(self.current_experiment)
            # Find run_id from experiments list
            exp_info = next(
                (exp for exp in self.experiments if exp.experiment_id == exp_id),
                None,
            )
            run_id = getattr(exp_info, "current_run_id", None) if exp_info else None
            self.post_message(ExperimentSelected(exp_id, run_id))

    def _update_collapsed_header(self, experiment_id: str) -> None:
        """Update the collapsed experiment header with current stats"""
        exp_info = next(
            (exp for exp in self.experiments if exp.experiment_id == experiment_id),
            None,
        )
        if not exp_info:
            return

        total = exp_info.total_jobs
        finished = exp_info.finished_jobs
        failed = exp_info.failed_jobs
        run_id = getattr(exp_info, "current_run_id", None)

        if failed > 0:
            status = f"❌ {failed} failed"
        elif finished == total and total > 0:
            status = "✓ Done"
        elif finished < total:
            status = f"▶ {finished}/{total}"
        else:
            status = "Empty"

        collapsed_label = self.query_one("#collapsed-experiment-info", Label)
        run_text = f" [{run_id}]" if run_id else ""
        collapsed_label.update(
            f"📊 {experiment_id}{run_text} - {status} (click to go back)"
        )

    def collapse_to_experiment(self, experiment_id: str) -> None:
        """Collapse the experiments list to show only the selected experiment"""
        self._update_collapsed_header(experiment_id)

        # Hide table, show collapsed header
        self.query_one("#experiments-table-container").add_class("hidden")
        self.query_one("#collapsed-header").remove_class("hidden")
        self.collapsed = True
        self.remove_class("expanded")

    def expand_experiments(self) -> None:
        """Expand back to full experiments list"""
        # Show table, hide collapsed header
        self.query_one("#collapsed-header").add_class("hidden")
        self.query_one("#experiments-table-container").remove_class("hidden")
        self.collapsed = False
        self.current_experiment = None
        self.add_class("expanded")

        # Focus the experiments table
        table = self.query_one("#experiments-table", DataTable)
        table.focus()

    def on_click(self) -> None:
        """Handle clicks on the widget"""
        if self.collapsed:
            self.expand_experiments()
            self.post_message(ExperimentDeselected())
