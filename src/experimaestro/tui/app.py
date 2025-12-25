"""Main Textual TUI application for experiment monitoring"""

import logging
from pathlib import Path
from typing import Optional
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    DataTable,
    Label,
    TabbedContent,
    TabPane,
    RichLog,
    Button,
    Static,
)
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen
from textual import events
from experimaestro.scheduler.state_provider import (
    WorkspaceStateProvider,
    StateEvent,
    StateEventType,
)
from experimaestro.tui.log_viewer import LogViewerScreen


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal screen for quit confirmation"""

    CSS = """
    QuitConfirmScreen {
        align: center middle;
    }

    #quit-dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #quit-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #quit-message {
        margin-bottom: 1;
    }

    #quit-warning {
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
    }

    #quit-buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #quit-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, has_active_experiment: bool = False):
        super().__init__()
        self.has_active_experiment = has_active_experiment

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-dialog"):
            yield Static("Quit Experimaestro?", id="quit-title")

            if self.has_active_experiment:
                yield Static(
                    "⚠️  The experiment is still in progress.\n"
                    "Quitting will prevent new jobs from being launched.",
                    id="quit-warning",
                )
            else:
                yield Static("Are you sure you want to quit?", id="quit-message")

            with Horizontal(id="quit-buttons"):
                yield Button("Quit", variant="error", id="quit-yes")
                yield Button("Cancel", variant="primary", id="quit-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


def get_status_text(status: str):
    if status == "done":
        status_text = "✓ Done"
    elif status == "error":
        status_text = "❌ Failed"
    elif status == "running":
        status_text = "⏳ Running"
    elif status == "unscheduled":
        status_text = "📆 Unscheduled"
    else:
        status_text = status

    return status_text


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

    def __init__(self, state_provider: WorkspaceStateProvider) -> None:
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
            exp_id = self.experiments[0].experiment_id
            self.current_experiment = exp_id
            self.collapse_to_experiment(exp_id)
            self.post_message(ExperimentSelected(exp_id))

    def refresh_experiments(self) -> None:
        """Refresh the experiments list from state provider"""
        table = self.query_one("#experiments-table", DataTable)

        try:
            self.experiments = self.state_provider.get_experiments()
            self.log(
                f"Refreshing experiments: found {len(self.experiments)} experiments"
            )
        except Exception as e:
            self.log(f"ERROR refreshing experiments: {e}")
            import traceback

            self.log(traceback.format_exc())
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
            (exp for exp in self.experiments if exp.experiment_id == experiment_id),
            None,
        )
        if not exp_info:
            return

        # Update collapsed header
        total = exp_info.total_jobs
        finished = exp_info.finished_jobs
        failed = exp_info.failed_jobs

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


class JobSelected(Message):
    """Message sent when a job is selected"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class JobDeselected(Message):
    """Message sent when returning from job detail view"""

    pass


class ViewJobLogs(Message):
    """Message sent when user wants to view job logs"""

    def __init__(self, job_path: str, task_id: str) -> None:
        super().__init__()
        self.job_path = job_path
        self.task_id = task_id


class JobDetailView(Widget):
    """Widget displaying detailed job information"""

    BINDINGS = [
        Binding("l", "view_logs", "View Logs", priority=True),
    ]

    def __init__(self, state_provider: WorkspaceStateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.current_job_id: Optional[str] = None
        self.current_experiment_id: Optional[str] = None
        self.job_data: Optional[dict] = None

    def compose(self) -> ComposeResult:
        yield Label("Job Details", classes="section-title")
        with Vertical(id="job-detail-content"):
            yield Label("", id="job-id-label")
            yield Label("", id="job-task-label")
            yield Label("", id="job-status-label")
            yield Label("", id="job-path-label")
            yield Label("", id="job-times-label")
            yield Label("Tags:", classes="subsection-title")
            yield Label("", id="job-tags-label")
            yield Label("Progress:", classes="subsection-title")
            yield Label("", id="job-progress-label")
            yield Label("", id="job-logs-hint")

    def action_view_logs(self) -> None:
        """View job logs with toolong"""
        if self.job_data and self.job_data.path and self.job_data.task_id:
            self.post_message(
                ViewJobLogs(str(self.job_data.path), self.job_data.task_id)
            )

    def set_job(self, job_id: str, experiment_id: str) -> None:
        """Set the job to display"""
        self.current_job_id = job_id
        self.current_experiment_id = experiment_id
        self.refresh_job_detail()

    def refresh_job_detail(self) -> None:
        """Refresh job details from state provider"""
        if not self.current_job_id or not self.current_experiment_id:
            return

        job = self.state_provider.get_job(
            self.current_job_id, self.current_experiment_id
        )
        if not job:
            self.log(f"Job not found: {self.current_job_id}")
            return

        self.job_data = job

        # Update labels
        self.query_one("#job-id-label", Label).update(f"Job ID: {job.identifier}")
        self.query_one("#job-task-label", Label).update(f"Task: {job.task_id}")

        # Format status
        status_text = get_status_text(job.state.name if job.state else "unknown")

        self.query_one("#job-status-label", Label).update(f"Status: {status_text}")

        # Path (from locator)
        locator = job.locator or "-"
        self.query_one("#job-path-label", Label).update(f"Locator: {locator}")

        # Times - format timestamps
        from datetime import datetime
        import time as time_module

        def format_time(ts):
            if ts:
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            return "-"

        def format_duration(seconds: float) -> str:
            if seconds < 0:
                return "-"
            seconds = int(seconds)
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                return f"{seconds // 60}m {seconds % 60}s"
            elif seconds < 86400:
                return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
            else:
                return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"

        submitted = format_time(job.submittime)
        start = format_time(job.starttime)
        end = format_time(job.endtime)

        # Calculate duration
        duration = "-"
        if job.starttime:
            if job.endtime:
                duration = format_duration(job.endtime - job.starttime)
            else:
                duration = (
                    format_duration(time_module.time() - job.starttime) + " (running)"
                )

        times_text = f"Submitted: {submitted} | Start: {start} | End: {end} | Duration: {duration}"
        self.query_one("#job-times-label", Label).update(times_text)

        # Tags - job.tags is now a dict
        tags = job.tags
        if tags:
            tags_text = ", ".join(f"{k}={v}" for k, v in tags.items())
        else:
            tags_text = "(no tags)"
        self.query_one("#job-tags-label", Label).update(tags_text)

        # Progress
        progress_list = job.progress or []
        if progress_list:
            progress_lines = []
            for p in progress_list:
                level = p.get("level", 0)
                pct = p.get("progress", 0) * 100
                desc = p.get("desc", "")
                indent = "  " * level
                progress_lines.append(f"{indent}{pct:.1f}% {desc}")
            progress_text = "\n".join(progress_lines) if progress_lines else "-"
        else:
            progress_text = "-"
        self.query_one("#job-progress-label", Label).update(progress_text)

        # Log files hint - log files are named after the last part of the task ID
        job_path = job.path
        task_id = job.task_id
        if job_path and task_id:
            # Extract the last component of the task ID (e.g., "evaluate" from "mnist_xp.learn.evaluate")
            task_name = task_id.split(".")[-1]
            stdout_path = job_path / f"{task_name}.out"
            stderr_path = job_path / f"{task_name}.err"
            logs_exist = stdout_path.exists() or stderr_path.exists()
            if logs_exist:
                self.query_one("#job-logs-hint", Label).update(
                    "[bold cyan]Press 'l' to view logs[/bold cyan]"
                )
            else:
                self.query_one("#job-logs-hint", Label).update("(no log files found)")
        else:
            self.query_one("#job-logs-hint", Label).update("")


class JobsTable(Widget):
    """Widget displaying jobs for selected experiment"""

    def __init__(self, state_provider: WorkspaceStateProvider) -> None:
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

        # Sort jobs by submission time (oldest first)
        # Jobs without submittime go to the end
        jobs.sort(key=lambda j: j.submittime or float("inf"))

        # Get existing row keys
        existing_keys = set(table.rows.keys())
        current_job_ids = set()

        for job in jobs:
            job_id = job.identifier
            current_job_ids.add(job_id)
            task_id = job.task_id
            status = job.state.name if job.state else "unknown"

            # Format status with icon
            status_text = get_status_text(status)

            # Format progress - only show for running jobs
            if job.state and job.state.finished():
                # Don't show progress for finished jobs
                progress_text = "-"
            else:
                progress_list = job.progress or []
                if progress_list:
                    # Get the last progress entry
                    last_progress = progress_list[-1]
                    progress_pct = last_progress.get("progress", 0) * 100
                    progress_text = f"{progress_pct:.0f}%"
                else:
                    progress_text = "-"

            # Format timestamps
            from datetime import datetime
            import time as time_module

            submitted = "-"
            if job.submittime:
                submitted = datetime.fromtimestamp(job.submittime).strftime(
                    "%Y-%m-%d %H:%M"
                )

            # Calculate duration
            start = job.starttime
            end = job.endtime
            duration = "-"
            if start:
                if end:
                    # Job finished - show total duration
                    elapsed = end - start
                else:
                    # Job still running - show elapsed time so far
                    elapsed = time_module.time() - start
                # Format duration as human-readable
                duration = self._format_duration(elapsed)

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

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string"""
        if seconds < 0:
            return "-"

        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle job selection"""
        if event.row_key and self.current_experiment:
            job_id = str(event.row_key.value)
            self.post_message(JobSelected(job_id, self.current_experiment))


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

    #job-detail-container {
        width: 100%;
        height: 1fr;
        border: solid magenta;
        padding: 1;
    }

    JobDetailView {
        width: 100%;
        height: 100%;
    }

    #job-detail-content {
        padding: 1;
    }

    #job-detail-content Label {
        margin-bottom: 1;
    }

    .subsection-title {
        background: $surface;
        padding: 0 1;
        text-style: italic;
        margin-top: 1;
    }

    #job-logs-hint {
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "go_back", "Back"),
        Binding("l", "view_logs", "View Logs"),
    ]

    def __init__(
        self,
        workdir: Optional[Path] = None,
        watch: bool = True,
        state_provider: Optional[WorkspaceStateProvider] = None,
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
        self._listener_registered = False

        # Initialize state provider before compose
        if state_provider:
            self.state_provider = state_provider
            self.owns_provider = False  # Don't close external provider
            self._has_active_experiment = True  # External provider = active experiment
        else:
            from experimaestro.scheduler.state_provider import WorkspaceStateProvider

            # Get singleton provider instance for this workspace
            self.state_provider = WorkspaceStateProvider.get_instance(
                self.workdir,
                read_only=False,
                sync_on_start=True,
                sync_interval_minutes=5,
            )
            self.owns_provider = False  # Provider is singleton, don't close
            self._has_active_experiment = False  # Just viewing, no active experiment

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
        """Compose the monitor view with experiments, jobs, and job details"""
        yield ExperimentsList(self.state_provider)
        # Jobs view (hidden initially)
        with Vertical(id="jobs-container", classes="hidden"):
            yield JobsTable(self.state_provider)
        # Job detail view (hidden initially)
        with Vertical(id="job-detail-container", classes="hidden"):
            yield JobDetailView(self.state_provider)

    def on_mount(self) -> None:
        """Initialize the application"""
        # Resets logging
        logging.basicConfig(level=logging.INFO, force=True)

        # Get the widgets
        experiments_list = self.query_one(ExperimentsList)
        experiments_list.refresh_experiments()

        # Register as listener for push notifications from state provider
        if self.state_provider:
            self.state_provider.add_listener(self._on_state_event)
            self._listener_registered = True
            self.log("Registered state listener for push notifications")

    def _on_state_event(self, event: StateEvent) -> None:
        """Handle state change events from the state provider

        This is called from the state provider's thread, so we use
        call_from_thread to safely update the UI.
        """
        self.call_from_thread(self._handle_state_event, event)

    def _handle_state_event(self, event: StateEvent) -> None:
        """Process state event on the main thread"""
        self.log(f"State event received: {event.event_type.name}")

        if event.event_type == StateEventType.EXPERIMENT_UPDATED:
            # Refresh experiments list
            experiments_list = self.query_one(ExperimentsList)
            experiments_list.refresh_experiments()

        elif event.event_type == StateEventType.JOB_UPDATED:
            # Refresh jobs table if we're viewing the affected experiment
            jobs_table = self.query_one(JobsTable)
            event_exp_id = event.data.get("experimentId")

            if jobs_table.current_experiment == event_exp_id:
                jobs_table.refresh_jobs()

            # Also refresh job detail if we're viewing the affected job
            job_detail_container = self.query_one("#job-detail-container")
            if not job_detail_container.has_class("hidden"):
                job_detail_view = self.query_one(JobDetailView)
                event_job_id = event.data.get("jobId")
                if job_detail_view.current_job_id == event_job_id:
                    job_detail_view.refresh_job_detail()

            # Also update the experiment stats in the experiments list
            experiments_list = self.query_one(ExperimentsList)
            experiments_list.refresh_experiments()

        elif event.event_type == StateEventType.RUN_UPDATED:
            # Refresh experiments list to show updated run info
            experiments_list = self.query_one(ExperimentsList)
            experiments_list.refresh_experiments()

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
        # Also hide job detail if visible
        job_detail_container = self.query_one("#job-detail-container")
        job_detail_container.add_class("hidden")

    def on_job_selected(self, message: JobSelected) -> None:
        """Handle job selection - show job detail view"""
        self.log(f"Job selected: {message.job_id} from {message.experiment_id}")

        # Hide jobs table, show job detail
        jobs_container = self.query_one("#jobs-container")
        jobs_container.add_class("hidden")

        job_detail_container = self.query_one("#job-detail-container")
        job_detail_container.remove_class("hidden")

        # Set the job to display
        job_detail_view = self.query_one(JobDetailView)
        job_detail_view.set_job(message.job_id, message.experiment_id)

    def on_job_deselected(self, message: JobDeselected) -> None:
        """Handle job deselection - go back to jobs view"""
        # Hide job detail, show jobs table
        job_detail_container = self.query_one("#job-detail-container")
        job_detail_container.add_class("hidden")

        jobs_container = self.query_one("#jobs-container")
        jobs_container.remove_class("hidden")

        # Focus the jobs table
        jobs_table = self.query_one("#jobs-table", DataTable)
        jobs_table.focus()

    def action_refresh(self) -> None:
        """Manually refresh the data"""
        experiments_list = self.query_one(ExperimentsList)
        jobs_table = self.query_one(JobsTable)

        experiments_list.refresh_experiments()
        jobs_table.refresh_jobs()

        # Also refresh job detail if visible
        job_detail_container = self.query_one("#job-detail-container")
        if not job_detail_container.has_class("hidden"):
            job_detail_view = self.query_one(JobDetailView)
            job_detail_view.refresh_job_detail()

    def action_go_back(self) -> None:
        """Go back one level in the navigation hierarchy"""
        # Check if job detail is visible -> go back to jobs
        job_detail_container = self.query_one("#job-detail-container")
        if not job_detail_container.has_class("hidden"):
            self.post_message(JobDeselected())
            return

        # Check if jobs list is visible -> go back to experiments
        jobs_container = self.query_one("#jobs-container")
        if not jobs_container.has_class("hidden"):
            experiments_list = self.query_one(ExperimentsList)
            if experiments_list.collapsed:
                experiments_list.expand_experiments()
                jobs_container.add_class("hidden")

    def action_view_logs(self) -> None:
        """View logs for the current job (if job detail is visible)"""
        job_detail_container = self.query_one("#job-detail-container")
        if not job_detail_container.has_class("hidden"):
            job_detail_view = self.query_one(JobDetailView)
            job_detail_view.action_view_logs()

    def on_view_job_logs(self, message: ViewJobLogs) -> None:
        """Handle request to view job logs - push LogViewerScreen"""
        job_path = Path(message.job_path)
        # Log files are named after the last part of the task ID
        task_name = message.task_id.split(".")[-1]
        stdout_path = job_path / f"{task_name}.out"
        stderr_path = job_path / f"{task_name}.err"

        # Collect existing log files
        log_files = []
        if stdout_path.exists():
            log_files.append(str(stdout_path))
        if stderr_path.exists():
            log_files.append(str(stderr_path))

        if not log_files:
            self.notify(
                f"No log files found: {task_name}.out/.err in {job_path}",
                severity="warning",
            )
            return

        # Push the log viewer screen
        job_id = job_path.name
        self.push_screen(LogViewerScreen(log_files, job_id))

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

    def action_quit(self) -> None:
        """Show quit confirmation dialog"""

        def handle_quit_response(confirmed: bool) -> None:
            if confirmed:
                self.exit()

        self.push_screen(
            QuitConfirmScreen(has_active_experiment=self._has_active_experiment),
            handle_quit_response,
        )

    def on_unmount(self) -> None:
        """Clean up when closing"""
        # Unregister listener
        if self._listener_registered and self.state_provider:
            self.state_provider.remove_listener(self._on_state_event)
            self._listener_registered = False
            self.log("Unregistered state listener")

        # Only close state provider if we own it (not external/active experiment)
        if self.state_provider and self.owns_provider:
            self.state_provider.close()
