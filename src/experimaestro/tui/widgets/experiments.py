"""Experiments list widget for the TUI"""

from datetime import datetime
import time as time_module
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
)


class ExperimentsList(Widget):
    """Widget displaying list of experiments"""

    BINDINGS = [
        Binding("d", "delete_experiment", "Delete", show=False),
        Binding("k", "kill_experiment", "Kill", show=False),
    ]

    current_experiment: reactive[Optional[str]] = reactive(None)
    collapsed: reactive[bool] = reactive(False)

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

    def refresh_experiments(self) -> None:
        """Refresh the experiments list from state provider"""
        table = self.query_one("#experiments-table", DataTable)

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

        # Get existing row keys
        existing_keys = set(table.rows.keys())
        current_exp_ids = set()

        for exp in self.experiments:
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
                started = datetime.fromtimestamp(exp.started_at).strftime(
                    "%Y-%m-%d %H:%M"
                )
            else:
                started = "-"

            # Calculate duration
            duration = "-"
            if exp.started_at:
                if exp.ended_at:
                    elapsed = exp.ended_at - exp.started_at
                else:
                    # Still running - show elapsed time
                    elapsed = time_module.time() - exp.started_at
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

            # Update existing row or add new one
            if exp_id in existing_keys:
                table.update_cell(exp_id, "id", exp_id, update_width=True)
                table.update_cell(exp_id, "run", run_id, update_width=True)
                table.update_cell(exp_id, "runs", runs_count, update_width=True)
                table.update_cell(exp_id, "host", hostname, update_width=True)
                table.update_cell(exp_id, "jobs", jobs_text, update_width=True)
                table.update_cell(exp_id, "status", status, update_width=True)
                table.update_cell(exp_id, "started", started, update_width=True)
                table.update_cell(exp_id, "duration", duration, update_width=True)
            else:
                table.add_row(
                    exp_id,
                    run_id,
                    runs_count,
                    hostname,
                    jobs_text,
                    status,
                    started,
                    duration,
                    key=exp_id,
                )

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

    def expand_experiments(self) -> None:
        """Expand back to full experiments list"""
        # Show table, hide collapsed header
        self.query_one("#collapsed-header").add_class("hidden")
        self.query_one("#experiments-table-container").remove_class("hidden")
        self.collapsed = False
        self.current_experiment = None

        # Focus the experiments table
        table = self.query_one("#experiments-table", DataTable)
        table.focus()

    def on_click(self) -> None:
        """Handle clicks on the widget"""
        if self.collapsed:
            self.expand_experiments()
            self.post_message(ExperimentDeselected())
