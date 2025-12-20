"""Main Textual TUI application for experiment monitoring"""

import logging
from pathlib import Path
from typing import Optional
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Widget
from textual.widgets import (
    Header,
    Footer,
    DataTable,
    Label,
    TabbedContent,
    TabPane,
    RichLog,
)
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from textual import events, work
from experimaestro.scheduler.state_provider import StateProvider


class CaptureLog(RichLog):
    """Custom RichLog that handles Print events and logging"""

    def on_mount(self) -> None:
        """Enable print capturing when widget is mounted"""
        # Capture print statements
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        """Handle print events from captured stdout/stderr"""
        self.write(event.text)


class ExperimentsList(Widget):
    """Widget displaying list of experiments"""

    current_experiment: reactive[Optional[str]] = reactive(None)
    collapsed: reactive[bool] = reactive(False)

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.experiments = []

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
        table.add_column("Jobs", key="jobs")
        table.add_column("Status", key="status")
        self.refresh_experiments()

        # If there's only one experiment, automatically select it
        if len(self.experiments) == 1:
            exp_id = self.experiments[0]["experiment_id"]
            self.current_experiment = exp_id
            self.collapse_to_experiment(exp_id)
            self.post_message(ExperimentSelected(exp_id))

    def refresh_experiments(self) -> None:
        """Refresh the experiments list from state provider"""
        table = self.query_one("#experiments-table", DataTable)

        self.experiments = self.state_provider.get_experiments()
        self.log("Refreshing experiments", len(self.experiments))

        # Get existing row keys
        existing_keys = set(table.rows.keys())
        current_exp_ids = set()

        for exp in self.experiments:
            exp_id = exp["experiment_id"]
            current_exp_ids.add(exp_id)
            total = exp["total_jobs"]
            finished = exp["finished_jobs"]
            failed = exp["failed_jobs"]

            # Determine status
            if failed > 0:
                status = f"❌ {failed} failed"
            elif finished == total and total > 0:
                status = "✓ Done"
            elif finished < total:
                status = f"⏳ {finished}/{total}"
            else:
                status = "Empty"

            jobs_text = f"{finished}/{total}"

            # Update existing row or add new one
            if exp_id in existing_keys:
                table.update_cell(exp_id, "id", exp_id, update_width=True)
                table.update_cell(exp_id, "jobs", jobs_text, update_width=True)
                table.update_cell(exp_id, "status", status, update_width=True)
            else:
                table.add_row(exp_id, jobs_text, status, key=exp_id)

        # Remove rows for experiments that no longer exist
        for old_exp_id in existing_keys - current_exp_ids:
            table.remove_row(old_exp_id)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle experiment selection"""
        if event.row_key:
            self.current_experiment = str(event.row_key.value)
            self.collapse_to_experiment(self.current_experiment)
            self.post_message(ExperimentSelected(str(event.row_key.value)))

    def collapse_to_experiment(self, experiment_id: str) -> None:
        """Collapse the experiments list to show only the selected experiment"""
        # Find experiment info
        exp_info = next(
            (exp for exp in self.experiments if exp["experiment_id"] == experiment_id),
            None,
        )
        if not exp_info:
            return

        # Update collapsed header
        total = exp_info["total_jobs"]
        finished = exp_info["finished_jobs"]
        failed = exp_info["failed_jobs"]

        if failed > 0:
            status = f"❌ {failed} failed"
        elif finished == total and total > 0:
            status = "✓ Done"
        elif finished < total:
            status = f"⏳ {finished}/{total}"
        else:
            status = "Empty"

        collapsed_label = self.query_one("#collapsed-experiment-info", Label)
        collapsed_label.update(f"📊 {experiment_id} - {status} (click to expand)")

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


class ExperimentSelected(Message):
    """Message sent when an experiment is selected"""

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id


class ExperimentDeselected(Message):
    """Message sent when an experiment is deselected"""

    pass


class JobsTable(Widget):
    """Widget displaying jobs for selected experiment"""

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.current_experiment: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Label("Jobs", classes="section-title")
        yield DataTable(id="jobs-table", cursor_type="row")

    def on_mount(self) -> None:
        """Initialize the jobs table"""
        table = self.query_one("#jobs-table", DataTable)
        table.add_column("Job ID", key="job_id", width=8)
        table.add_column("Task", key="task")
        table.add_column("Status", key="status")
        table.add_column("Progress", key="progress")
        table.add_column("Submitted", key="submitted")
        table.add_column("Duration", key="duration")
        table.cursor_type = "row"

    def set_experiment(self, experiment_id: Optional[str]) -> None:
        """Set the current experiment and refresh jobs"""
        self.current_experiment = experiment_id
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        """Refresh the jobs list from state provider"""
        table = self.query_one("#jobs-table", DataTable)

        if not self.current_experiment:
            return

        jobs = self.state_provider.get_jobs(self.current_experiment)
        self.log(f"Refreshing jobs for {self.current_experiment}: {len(jobs)} jobs")

        # Get existing row keys
        existing_keys = set(table.rows.keys())
        current_job_ids = set()

        for job in jobs:
            job_id = job["jobId"]
            current_job_ids.add(job_id)
            task_id = job["taskId"]
            status = job["status"]

            # Format status with icon
            if status == "done":
                status_text = "✓ Done"
            elif status == "failed":
                status_text = "❌ Failed"
            elif status == "running":
                status_text = "⏳ Running"
            else:
                status_text = status

            # Format progress
            progress_list = job.get("progress", [])
            if progress_list:
                # Get the last progress entry
                last_progress = progress_list[-1]
                progress_pct = last_progress.get("progress", 0) * 100
                progress_text = f"{progress_pct:.0f}%"
            else:
                progress_text = "-"

            # Format timestamps
            submitted = job.get("submitted", "-")
            if submitted and submitted != "-":
                submitted = submitted.split("T")[0]  # Just the date

            # Calculate duration
            start = job.get("start")
            end = job.get("end")
            if start and end and start != "-" and end != "-":
                # Simple duration display (would need proper parsing for accurate calculation)
                duration = "completed"
            elif start and start != "-":
                duration = "running"
            else:
                duration = "-"

            row_data = (
                job_id,
                task_id,
                status_text,
                progress_text,
                submitted,
                duration,
            )

            # Update existing row or add new one
            if job_id in existing_keys:
                table.update_cell(job_id, "job_id", job_id, update_width=True)
                table.update_cell(job_id, "task", task_id, update_width=True)
                table.update_cell(job_id, "status", status_text, update_width=True)
                table.update_cell(job_id, "progress", progress_text, update_width=True)
                table.update_cell(job_id, "submitted", submitted, update_width=True)
                table.update_cell(job_id, "duration", duration, update_width=True)
                self.log(f"Updated job row: {job_id}")
            else:
                table.add_row(*row_data, key=job_id)
                self.log(f"Added job row: {job_id}")

        # Remove rows for jobs that no longer exist
        for old_job_id in existing_keys - current_job_ids:
            table.remove_row(old_job_id)

        self.log(f"Jobs table now has {table.row_count} rows")


class ExperimentTUI(App):
    """Textual TUI for monitoring experiments"""

    CSS = """
    #main-container {
        width: 100%;
        height: 100%;
    }

    ExperimentsList {
        width: 100%;
        height: auto;
    }

    Monitor {
        height: 100%;
    }

    #experiments-table-container {
        width: 100%;
        height: auto;
    }

    #collapsed-header {
        background: $boost;
        padding: 1;
        text-style: bold;
        border: solid green;
        height: auto;
        width: 100%;
    }

    #collapsed-header:hover {
        background: $primary;
    }

    #collapsed-experiment-info {
        width: 100%;
        height: auto;
    }

    #jobs-container {
        width: 100%;
        height: 1fr;
        border: solid blue;
        padding: 1;
    }

    JobsTable {
        width: 100%;
        height: 100%;
    }

    #jobs-table {
        height: 100%;
    }

    .hidden {
        display: none;
    }

    .section-title {
        background: $boost;
        padding: 1;
        text-style: bold;
    }

    #experiments-table {
        height: auto;
        min-height: 5;
    }

    DataTable {
        height: 100%;
    }

    RichLog {
        height: 100%;
        border: solid cyan;
    }

    TabbedContent {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("escape", "back_to_experiments", "Back to Experiments", priority=True),
    ]

    def __init__(
        self,
        workdir: Optional[Path] = None,
        watch: bool = True,
        state_provider: Optional[StateProvider] = None,
        show_logs: bool = False,
    ):
        """Initialize the TUI

        Args:
            workdir: Workspace directory (required if state_provider not provided)
            watch: Enable filesystem watching for workspace mode
            state_provider: Pre-initialized state provider (for active experiments)
            show_logs: Whether to show the logs tab (for active experiments)
        """
        super().__init__()
        self.workdir = workdir
        self.watch = watch
        self.show_logs = show_logs

        # Initialize state provider before compose
        if state_provider:
            self.state_provider = state_provider
        else:
            from experimaestro.scheduler.state_provider import WorkspaceStateProvider

            self.state_provider = WorkspaceStateProvider(self.workdir, watch=self.watch)

    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()

        if self.show_logs:
            # Tabbed layout with logs
            with TabbedContent():
                with TabPane("Monitor", id="monitor-tab"):
                    yield from self._compose_monitor_view()
                with TabPane("Logs", id="logs-tab"):
                    yield CaptureLog(id="logs", wrap=True, highlight=True, markup=True)
        else:
            # Simple layout without logs
            with Vertical(id="main-container"):
                yield from self._compose_monitor_view()

        yield Footer()

    def _compose_monitor_view(self):
        """Compose the monitor view with experiments and jobs"""
        yield ExperimentsList(self.state_provider)
        # Jobs view (hidden initially)
        with Vertical(id="jobs-container", classes="hidden"):
            yield JobsTable(self.state_provider)

    def on_mount(self) -> None:
        """Initialize the application"""
        # Resets logging
        logging.basicConfig(level=logging.INFO, force=True)

        # Get the widgets
        experiments_list = self.query_one(ExperimentsList)
        experiments_list.refresh_experiments()

        # Start auto-refresh
        self.start_auto_refresh()

    def on_experiment_selected(self, message: ExperimentSelected) -> None:
        """Handle experiment selection - show jobs view"""
        self.log(f"Experiment selected: {message.experiment_id}")
        jobs_table_widget = self.query_one(JobsTable)
        jobs_table_widget.set_experiment(message.experiment_id)

        # Show jobs container
        jobs_container = self.query_one("#jobs-container")
        self.log(f"Jobs container before: hidden={jobs_container.has_class('hidden')}")
        jobs_container.remove_class("hidden")
        self.log(f"Jobs container after: hidden={jobs_container.has_class('hidden')}")

        # Focus the jobs table
        jobs_table = self.query_one("#jobs-table", DataTable)
        jobs_table.focus()

    def on_experiment_deselected(self, message: ExperimentDeselected) -> None:
        """Handle experiment deselection - hide jobs view"""
        jobs_container = self.query_one("#jobs-container")
        jobs_container.add_class("hidden")

    def action_refresh(self) -> None:
        """Manually refresh the data"""
        experiments_list = self.query_one(ExperimentsList)
        jobs_table = self.query_one(JobsTable)

        experiments_list.refresh_experiments()
        jobs_table.refresh_jobs()

    def action_back_to_experiments(self) -> None:
        """Switch back to experiments view from jobs view"""
        experiments_list = self.query_one(ExperimentsList)
        if experiments_list.collapsed:
            experiments_list.expand_experiments()
            jobs_container = self.query_one("#jobs-container")
            jobs_container.add_class("hidden")

    def action_switch_focus(self) -> None:
        """Switch focus between experiments and jobs tables"""
        focused = self.focused
        if focused:
            experiments_table = self.query_one("#experiments-table", DataTable)
            jobs_table = self.query_one("#jobs-table", DataTable)

            if focused == experiments_table:
                jobs_table.focus()
            else:
                experiments_table.focus()

    @work(thread=True)
    async def start_auto_refresh(self) -> None:
        """Auto-refresh the display every 5 seconds"""
        import asyncio

        while True:
            await asyncio.sleep(5)
            self.call_from_thread(self.action_refresh)

    def on_unmount(self) -> None:
        """Clean up when closing"""
        if self.state_provider:
            self.state_provider.close()
