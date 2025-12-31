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
    Input,
)
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual import events
from rich.text import Text
from experimaestro.scheduler.state_provider import (
    WorkspaceStateProvider,
    StateEvent,
    StateEventType,
)
from experimaestro.tui.log_viewer import LogViewerScreen


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
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


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal screen for quit confirmation"""

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


class DeleteConfirmScreen(ModalScreen[bool]):
    """Modal screen for delete confirmation"""

    def __init__(
        self, item_type: str, item_name: str, warning: Optional[str] = None
    ) -> None:
        super().__init__()
        self.item_type = item_type
        self.item_name = item_name
        self.warning = warning

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-dialog"):
            yield Static(f"Delete {self.item_type}?", id="delete-title")
            yield Static(
                f"This will permanently delete: {self.item_name}", id="delete-message"
            )

            if self.warning:
                yield Static(f"Warning: {self.warning}", id="delete-warning")

            with Horizontal(id="delete-buttons"):
                yield Button("Delete", variant="error", id="delete-yes")
                yield Button("Cancel", variant="primary", id="delete-no")

    def on_mount(self) -> None:
        """Focus cancel button by default"""
        self.query_one("#delete-no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


class KillConfirmScreen(ModalScreen[bool]):
    """Modal screen for kill confirmation"""

    def __init__(self, item_type: str, item_name: str) -> None:
        super().__init__()
        self.item_type = item_type
        self.item_name = item_name

    def compose(self) -> ComposeResult:
        with Vertical(id="kill-dialog"):
            yield Static(f"Kill {self.item_type}?", id="kill-title")
            yield Static(f"This will terminate: {self.item_name}", id="kill-message")

            with Horizontal(id="kill-buttons"):
                yield Button("Kill", variant="warning", id="kill-yes")
                yield Button("Cancel", variant="primary", id="kill-no")

    def on_mount(self) -> None:
        """Focus cancel button by default"""
        self.query_one("#kill-no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "kill-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


def get_status_icon(status: str):
    if status == "done":
        return "✓"
    elif status == "error":
        return "❌"
    elif status == "running":
        return "▶"
    elif status == "waiting":
        return "⌛"  # Waiting for dependencies
    else:
        # phantom, unscheduled or unknown
        return "👻"


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

    BINDINGS = [
        Binding("d", "delete_experiment", "Delete", show=False),
        Binding("k", "kill_experiment", "Kill", show=False),
    ]

    current_experiment: reactive[Optional[str]] = reactive(None)
    collapsed: reactive[bool] = reactive(False)

    def __init__(self, state_provider: WorkspaceStateProvider) -> None:
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
        table.add_column("Jobs", key="jobs")
        table.add_column("Status", key="status")
        table.add_column("Started", key="started")
        table.add_column("Duration", key="duration")
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

        from datetime import datetime
        import time as time_module

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

            # Update existing row or add new one
            if exp_id in existing_keys:
                table.update_cell(exp_id, "id", exp_id, update_width=True)
                table.update_cell(exp_id, "jobs", jobs_text, update_width=True)
                table.update_cell(exp_id, "status", status, update_width=True)
                table.update_cell(exp_id, "started", started, update_width=True)
                table.update_cell(exp_id, "duration", duration, update_width=True)
            else:
                table.add_row(exp_id, jobs_text, status, started, duration, key=exp_id)

        # Remove rows for experiments that no longer exist
        for old_exp_id in existing_keys - current_exp_ids:
            table.remove_row(old_exp_id)

        # Update collapsed header if viewing an experiment
        if self.collapsed and self.current_experiment:
            self._update_collapsed_header(self.current_experiment)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle experiment selection"""
        if event.row_key:
            self.current_experiment = str(event.row_key.value)
            self.collapse_to_experiment(self.current_experiment)
            self.post_message(ExperimentSelected(str(event.row_key.value)))

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

        if failed > 0:
            status = f"❌ {failed} failed"
        elif finished == total and total > 0:
            status = "✓ Done"
        elif finished < total:
            status = f"▶ {finished}/{total}"
        else:
            status = "Empty"

        collapsed_label = self.query_one("#collapsed-experiment-info", Label)
        collapsed_label.update(f"📊 {experiment_id} - {status} (click to go back)")

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


class ViewJobLogsRequest(Message):
    """Message sent when user requests to view logs from jobs table"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class DeleteJobRequest(Message):
    """Message sent when user requests to delete a job"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class DeleteExperimentRequest(Message):
    """Message sent when user requests to delete an experiment"""

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id


class KillJobRequest(Message):
    """Message sent when user requests to kill a running job"""

    def __init__(self, job_id: str, experiment_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.experiment_id = experiment_id


class KillExperimentRequest(Message):
    """Message sent when user requests to kill all running jobs in an experiment"""

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id


class FilterChanged(Message):
    """Message sent when search filter changes"""

    def __init__(self, filter_fn) -> None:
        super().__init__()
        self.filter_fn = filter_fn


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

        # Format status with icon and name
        status_name = job.state.name if job.state else "unknown"
        status_icon = get_status_icon(status_name)
        status_text = f"{status_icon} {status_name}"

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


class SearchBar(Widget):
    """Search bar widget with filter hints for filtering jobs"""

    visible: reactive[bool] = reactive(False)
    _keep_filter: bool = False  # Flag to keep filter when hiding
    _query_valid: bool = False  # Track if current query is valid

    def __init__(self) -> None:
        super().__init__()
        self.filter_fn = None
        self.active_query = ""  # Store the active query text

    def compose(self) -> ComposeResult:
        # Active filter indicator (shown when filter active but bar hidden)
        yield Static("", id="active-filter")
        # Search input container
        with Vertical(id="search-container"):
            yield Input(
                placeholder="Filter: @state = 'done', @name ~ 'pattern', tag = 'value'",
                id="search-input",
            )
            yield Static(
                "Syntax: @state = 'done' | @name ~ 'regex' | tag = 'value' | and/or",
                id="search-hints",
            )
            yield Static("", id="search-error")

    def on_mount(self) -> None:
        """Initialize visibility state"""
        # Start with everything hidden
        self.display = False
        self.query_one("#search-container").display = False
        self.query_one("#active-filter").display = False
        self.query_one("#search-error").display = False

    def watch_visible(self, visible: bool) -> None:
        """Show/hide search bar"""
        search_container = self.query_one("#search-container")
        active_filter = self.query_one("#active-filter")
        error_widget = self.query_one("#search-error")

        if visible:
            self.display = True
            search_container.display = True
            active_filter.display = False
            self.query_one("#search-input", Input).focus()
        else:
            if not self._keep_filter:
                self.query_one("#search-input", Input).value = ""
                self.filter_fn = None
                self.active_query = ""
                self._query_valid = False
            self._keep_filter = False

            # Show/hide based on whether filter is active
            if self.filter_fn is not None:
                # Filter active - show indicator, hide input
                self.display = True
                search_container.display = False
                error_widget.display = False
                active_filter.update(
                    f"Filter: {self.active_query} (/ to edit, c to clear)"
                )
                active_filter.display = True
            else:
                # No filter - hide everything including this widget
                self.display = False
                search_container.display = False
                active_filter.display = False
                error_widget.display = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Parse filter expression when input changes"""
        query = event.value.strip()
        input_widget = self.query_one("#search-input", Input)
        error_widget = self.query_one("#search-error", Static)

        if not query:
            self.filter_fn = None
            self._query_valid = False
            self.post_message(FilterChanged(None))
            input_widget.remove_class("error")
            input_widget.remove_class("valid")
            error_widget.display = False
            return

        try:
            from experimaestro.cli.filter import createFilter

            self.filter_fn = createFilter(query)
            self._query_valid = True
            self.active_query = query
            self.post_message(FilterChanged(self.filter_fn))
            input_widget.remove_class("error")
            input_widget.add_class("valid")
            error_widget.display = False
        except Exception as e:
            self.filter_fn = None
            self._query_valid = False
            self.post_message(FilterChanged(None))
            input_widget.remove_class("valid")
            input_widget.add_class("error")
            error_widget.update(f"Invalid query: {str(e)[:50]}")
            error_widget.display = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Apply filter and hide search bar (only if query is valid)"""
        if self._query_valid and self.filter_fn is not None:
            # Set flag to keep filter when hiding
            self._keep_filter = True
            self.visible = False
            # Post message to focus jobs table
            self.post_message(SearchApplied())
        # If invalid, do nothing (keep input focused for correction)


class SearchApplied(Message):
    """Message sent when search filter is applied via Enter"""

    pass


class JobsTable(Vertical):
    """Widget displaying jobs for selected experiment"""

    BINDINGS = [
        Binding("d", "delete_job", "Delete", show=False),
        Binding("k", "kill_job", "Kill", show=False),
        Binding("l", "view_logs", "Logs"),
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("/", "toggle_search", "Search"),
        Binding("c", "clear_filter", "Clear", show=False),
        Binding("S", "sort_by_status", "Sort ⚑", show=False),
        Binding("T", "sort_by_task", "Sort Task", show=False),
        Binding("D", "sort_by_submitted", "Sort Date", show=False),
        Binding("escape", "clear_search", show=False, priority=True),
    ]

    # Track current sort state
    _sort_column: Optional[str] = None
    _sort_reverse: bool = False
    _needs_rebuild: bool = True  # Start with rebuild needed

    def __init__(self, state_provider: WorkspaceStateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.filter_fn = None
        self.current_experiment: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield SearchBar()
        yield Label("Jobs", classes="section-title")
        yield DataTable(id="jobs-table", cursor_type="row")

    def action_toggle_search(self) -> None:
        """Toggle search bar visibility"""
        search_bar = self.query_one(SearchBar)
        search_bar.visible = not search_bar.visible

    def action_clear_filter(self) -> None:
        """Clear the active filter"""
        if self.filter_fn is not None:
            search_bar = self.query_one(SearchBar)
            search_bar.query_one("#search-input", Input).value = ""
            search_bar.filter_fn = None
            search_bar.active_query = ""
            search_bar._query_valid = False
            # Hide the SearchBar completely
            search_bar.display = False
            search_bar.query_one("#search-container").display = False
            search_bar.query_one("#active-filter").display = False
            search_bar.query_one("#search-error").display = False
            self.filter_fn = None
            self.refresh_jobs()
            self.notify("Filter cleared", severity="information")

    def action_sort_by_status(self) -> None:
        """Sort jobs by status"""
        if self._sort_column == "status":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "status"
            self._sort_reverse = False
        self._needs_rebuild = True
        self._update_column_headers()
        self.refresh_jobs()
        order = "desc" if self._sort_reverse else "asc"
        self.notify(f"Sorted by status ({order})", severity="information")

    def action_sort_by_task(self) -> None:
        """Sort jobs by task"""
        if self._sort_column == "task":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "task"
            self._sort_reverse = False
        self._needs_rebuild = True
        self._update_column_headers()
        self.refresh_jobs()
        order = "desc" if self._sort_reverse else "asc"
        self.notify(f"Sorted by task ({order})", severity="information")

    def action_sort_by_submitted(self) -> None:
        """Sort jobs by submission time"""
        if self._sort_column == "submitted":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "submitted"
            self._sort_reverse = False
        self._needs_rebuild = True
        self._update_column_headers()
        self.refresh_jobs()
        order = "newest first" if self._sort_reverse else "oldest first"
        self.notify(f"Sorted by date ({order})", severity="information")

    def action_clear_search(self) -> None:
        """Handle escape: hide search bar if visible, or go back"""
        search_bar = self.query_one(SearchBar)
        if search_bar.visible:
            # Search bar visible - hide it and clear filter
            search_bar.visible = False
            self.filter_fn = None
            self.refresh_jobs()
            # Focus the jobs table
            self.query_one("#jobs-table", DataTable).focus()
        else:
            # Search bar hidden - go back (keep filter)
            self.app.action_go_back()

    def on_filter_changed(self, message: FilterChanged) -> None:
        """Apply new filter"""
        self.filter_fn = message.filter_fn
        self.refresh_jobs()

    def on_search_applied(self, message: SearchApplied) -> None:
        """Focus jobs table when search is applied"""
        self.query_one("#jobs-table", DataTable).focus()

    def _get_selected_job_id(self) -> Optional[str]:
        """Get the job ID from the currently selected row"""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None:
            return None
        row_key = table.get_row_at(table.cursor_row)
        if row_key:
            # The first column is job_id
            return str(table.get_row_at(table.cursor_row)[0])
        return None

    def action_delete_job(self) -> None:
        """Request to delete the selected job"""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        # Get job ID from the row key
        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            self.post_message(DeleteJobRequest(job_id, self.current_experiment))

    def action_kill_job(self) -> None:
        """Request to kill the selected job"""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            self.post_message(KillJobRequest(job_id, self.current_experiment))

    def action_view_logs(self) -> None:
        """Request to view logs for the selected job"""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            self.post_message(ViewJobLogsRequest(job_id, self.current_experiment))

    def action_copy_path(self) -> None:
        """Copy the job folder path to clipboard"""
        import pyperclip

        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            job = self.state_provider.get_job(job_id, self.current_experiment)
            if job and job.path:
                try:
                    pyperclip.copy(str(job.path))
                    self.notify(f"Path copied: {job.path}", severity="information")
                except Exception as e:
                    self.notify(f"Failed to copy: {e}", severity="error")
            else:
                self.notify("No path available for this job", severity="warning")

    # Status sort order (for sorting by status)
    STATUS_ORDER = {
        "running": 0,
        "waiting": 1,
        "error": 2,
        "done": 3,
        "unscheduled": 4,
        "phantom": 5,
    }

    # Column key to display name mapping
    COLUMN_LABELS = {
        "job_id": "ID",
        "task": "Task",
        "status": "⚑",
        "tags": "Tags",
        "submitted": "Submitted",
        "duration": "Duration",
    }

    # Columns that support sorting (column key -> sort column name)
    SORTABLE_COLUMNS = {
        "status": "status",
        "task": "task",
        "submitted": "submitted",
    }

    def on_mount(self) -> None:
        """Initialize the jobs table"""
        table = self.query_one("#jobs-table", DataTable)
        table.add_column("ID", key="job_id")
        table.add_column("Task", key="task")
        table.add_column("⚑", key="status", width=6)
        table.add_column("Tags", key="tags")
        table.add_column("Submitted", key="submitted")
        table.add_column("Duration", key="duration")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def _update_column_headers(self) -> None:
        """Update column headers with sort indicators"""
        table = self.query_one("#jobs-table", DataTable)
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
                self._sort_reverse = False
            self._needs_rebuild = True
            self._update_column_headers()
            self.refresh_jobs()

    def set_experiment(self, experiment_id: Optional[str]) -> None:
        """Set the current experiment and refresh jobs"""
        self.current_experiment = experiment_id
        self.refresh_jobs()

    def refresh_jobs(self) -> None:  # noqa: C901
        """Refresh the jobs list from state provider"""
        table = self.query_one("#jobs-table", DataTable)

        if not self.current_experiment:
            return

        jobs = self.state_provider.get_jobs(self.current_experiment)
        self.log(f"Refreshing jobs for {self.current_experiment}: {len(jobs)} jobs")

        # Apply filter if set
        if self.filter_fn:
            jobs = [j for j in jobs if self.filter_fn(j)]
            self.log(f"After filter: {len(jobs)} jobs")

        # Sort jobs based on selected column
        if self._sort_column == "status":
            # Sort by status priority
            jobs.sort(
                key=lambda j: self.STATUS_ORDER.get(
                    j.state.name if j.state else "unknown", 99
                ),
                reverse=self._sort_reverse,
            )
        elif self._sort_column == "task":
            # Sort by task name
            jobs.sort(
                key=lambda j: j.task_id or "",
                reverse=self._sort_reverse,
            )
        else:
            # Default: sort by submission time (oldest first by default)
            # Jobs without submittime go to the end
            jobs.sort(
                key=lambda j: j.submittime or float("inf"),
                reverse=self._sort_reverse,
            )

        # Check if we need to rebuild (new/removed jobs, or status changed when sorting by status)
        from datetime import datetime
        import time as time_module

        existing_keys = {str(k.value) for k in table.rows.keys()}
        current_job_ids = {job.identifier for job in jobs}

        # Check if job set changed
        jobs_changed = existing_keys != current_job_ids

        # Check if status changed when sorting by status
        status_changed = False
        if self._sort_column == "status" and not jobs_changed:
            current_statuses = {
                job.identifier: (job.state.name if job.state else "unknown")
                for job in jobs
            }
            if (
                hasattr(self, "_last_statuses")
                and self._last_statuses != current_statuses
            ):
                status_changed = True
            self._last_statuses = current_statuses

        needs_rebuild = self._needs_rebuild or jobs_changed or status_changed
        self._needs_rebuild = False

        # Build row data for all jobs
        rows_data = {}
        for job in jobs:
            job_id = job.identifier
            task_id = job.task_id
            status = job.state.name if job.state else "unknown"

            # Format status with icon (and progress % if running)
            if status == "running":
                progress_list = job.progress or []
                if progress_list:
                    last_progress = progress_list[-1]
                    progress_pct = last_progress.get("progress", 0) * 100
                    status_text = f"▶ {progress_pct:.0f}%"
                else:
                    status_text = "▶"
            else:
                status_text = get_status_icon(status)

            # Format tags - show all tags on single line
            tags = job.tags
            if tags:
                tags_text = Text()
                for i, (k, v) in enumerate(tags.items()):
                    if i > 0:
                        tags_text.append(", ")
                    tags_text.append(f"{k}", style="bold")
                    tags_text.append(f"={v}")
            else:
                tags_text = Text("-")

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
                    elapsed = end - start
                else:
                    elapsed = time_module.time() - start
                duration = self._format_duration(elapsed)

            job_id_short = job_id[:7]
            rows_data[job_id] = (
                job_id_short,
                task_id,
                status_text,
                tags_text,
                submitted,
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
            for idx, job in enumerate(jobs):
                job_id = job.identifier
                table.add_row(*rows_data[job_id], key=job_id)
                if selected_key == job_id:
                    new_cursor_row = idx

            if new_cursor_row is not None and table.row_count > 0:
                table.move_cursor(row=new_cursor_row)
        else:
            # Just update cells in place - no reordering needed
            for job_id, row_data in rows_data.items():
                job_id_short, task_id, status_text, tags_text, submitted, duration = (
                    row_data
                )
                table.update_cell(job_id, "job_id", job_id_short, update_width=True)
                table.update_cell(job_id, "task", task_id, update_width=True)
                table.update_cell(job_id, "status", status_text, update_width=True)
                table.update_cell(job_id, "tags", tags_text, update_width=True)
                table.update_cell(job_id, "submitted", submitted, update_width=True)
                table.update_cell(job_id, "duration", duration, update_width=True)

        self.log(f"Jobs table now has {table.row_count} rows (rebuild={needs_rebuild})")

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


class SizeCalculated(Message):
    """Message sent when a folder size has been calculated"""

    def __init__(self, job_id: str, size: str, size_bytes: int) -> None:
        super().__init__()
        self.job_id = job_id
        self.size = size
        self.size_bytes = size_bytes


class OrphanJobsScreen(Screen):
    """Screen for viewing and managing orphan jobs"""

    BINDINGS = [
        Binding("d", "delete_selected", "Delete"),
        Binding("D", "delete_all", "Delete All", key_display="D"),
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("T", "sort_by_task", "Sort Task", show=False),
        Binding("Z", "sort_by_size", "Sort Size", show=False),
    ]

    _size_cache: dict = {}  # Class-level cache (formatted strings)
    _size_bytes_cache: dict = {}  # Class-level cache (raw bytes for sorting)

    def __init__(self, state_provider: WorkspaceStateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.orphan_jobs = []
        self._pending_jobs = []  # Jobs waiting for size calculation
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="orphan-container"):
            yield Static("Orphan Jobs", id="orphan-title")
            yield Static("", id="orphan-stats")
            yield DataTable(id="orphan-table", cursor_type="row")
            yield Static("", id="orphan-job-info")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the orphan jobs table"""
        table = self.query_one("#orphan-table", DataTable)
        table.add_column("⚑", key="status", width=3)
        table.add_column("Job ID", key="job_id", width=10)
        table.add_column("Task", key="task")
        table.add_column("Size", key="size", width=10)
        self.refresh_orphans()

    def action_sort_by_task(self) -> None:
        """Sort by task name"""
        if self._sort_column == "task":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "task"
            self._sort_reverse = False
        self._rebuild_table()
        order = "desc" if self._sort_reverse else "asc"
        self.notify(f"Sorted by task ({order})", severity="information")

    def action_sort_by_size(self) -> None:
        """Sort by size"""
        if self._sort_column == "size":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "size"
            self._sort_reverse = True  # Default: largest first
        self._rebuild_table()
        order = "largest first" if self._sort_reverse else "smallest first"
        self.notify(f"Sorted by size ({order})", severity="information")

    def _get_sorted_jobs(self):
        """Return jobs sorted by current sort column"""
        jobs = self.orphan_jobs[:]
        if self._sort_column == "task":
            jobs.sort(key=lambda j: j.task_id or "", reverse=self._sort_reverse)
        elif self._sort_column == "size":
            # Sort by raw bytes, jobs not in cache go to end
            jobs.sort(
                key=lambda j: self._size_bytes_cache.get(j.identifier, -1),
                reverse=self._sort_reverse,
            )
        return jobs

    def _rebuild_table(self) -> None:
        """Rebuild the table with current sort order"""
        table = self.query_one("#orphan-table", DataTable)
        table.clear()

        for job in self._get_sorted_jobs():
            status_icon = get_status_icon(job.state.name if job.state else "unknown")
            if job.identifier in self._size_cache:
                size_text = self._size_cache[job.identifier]
            else:
                size_text = "waiting"
            table.add_row(
                status_icon,
                job.identifier[:7],
                job.task_id,
                size_text,
                key=job.identifier,
            )

    def refresh_orphans(self) -> None:
        """Refresh the orphan jobs list"""
        # Only include orphan jobs that have an existing folder
        all_orphans = self.state_provider.get_orphan_jobs()
        self.orphan_jobs = [j for j in all_orphans if j.path and j.path.exists()]

        # Update stats
        stats = self.query_one("#orphan-stats", Static)
        stats.update(f"Found {len(self.orphan_jobs)} orphan jobs")

        # Collect jobs needing size calculation
        self._pending_jobs = [
            j for j in self.orphan_jobs if j.identifier not in self._size_cache
        ]

        # Rebuild table
        self._rebuild_table()

        # Start calculating sizes
        if self._pending_jobs:
            self._calculate_next_size()

    def _calculate_next_size(self) -> None:
        """Calculate size for the next pending job using a worker"""
        if not self._pending_jobs:
            return

        job = self._pending_jobs.pop(0)
        # Update to "calc..."
        self._update_size_cell(job.identifier, "calc...")
        # Run calculation in worker thread
        self.run_worker(
            self._calc_size_worker(job.identifier, job.path),
            thread=True,
        )

    async def _calc_size_worker(self, job_id: str, path):
        """Worker to calculate folder size"""
        size_bytes = self._get_folder_size(path)
        size_str = self._format_size(size_bytes)
        self._size_cache[job_id] = size_str
        self._size_bytes_cache[job_id] = size_bytes
        self.post_message(SizeCalculated(job_id, size_str, size_bytes))

    def on_size_calculated(self, message: SizeCalculated) -> None:
        """Handle size calculation completion"""
        self._size_bytes_cache[message.job_id] = message.size_bytes
        self._update_size_cell(message.job_id, message.size)
        # Calculate next one
        self._calculate_next_size()

    @staticmethod
    def _get_folder_size(path) -> int:
        """Calculate total size of a folder"""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except (OSError, PermissionError):
            pass
        return total

    @staticmethod
    def _format_size(size: int) -> str:
        """Format size in human-readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def _update_size_cell(self, job_id: str, value: str = None) -> None:
        """Update the size cell for a job"""
        try:
            table = self.query_one("#orphan-table", DataTable)
            size_text = (
                value if value is not None else self._size_cache.get(job_id, "-")
            )
            table.update_cell(job_id, "size", size_text)
        except Exception:
            pass  # Table may have changed

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show job details when a row is selected"""
        self._update_job_info()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show job details when cursor moves"""
        self._update_job_info()

    def _update_job_info(self) -> None:
        """Update the job info display"""
        table = self.query_one("#orphan-table", DataTable)
        info = self.query_one("#orphan-job-info", Static)

        if table.cursor_row is None:
            info.update("")
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            job = next((j for j in self.orphan_jobs if j.identifier == job_id), None)
            if job and job.path:
                size = self._size_cache.get(job.identifier, "calculating...")
                info.update(f"Path: {job.path}  |  Size: {size}")
            else:
                info.update("")

    def action_copy_path(self) -> None:
        """Copy the job folder path to clipboard"""
        import pyperclip

        table = self.query_one("#orphan-table", DataTable)
        if table.cursor_row is None:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            job = next((j for j in self.orphan_jobs if j.identifier == job_id), None)
            if job and job.path:
                try:
                    pyperclip.copy(str(job.path))
                    self.notify("Path copied", severity="information")
                except Exception as e:
                    self.notify(f"Failed to copy: {e}", severity="error")

    def action_delete_selected(self) -> None:
        """Delete the selected orphan job"""
        table = self.query_one("#orphan-table", DataTable)
        if table.cursor_row is None:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            job = next((j for j in self.orphan_jobs if j.identifier == job_id), None)
            if job:
                self._delete_job(job)

    def _delete_job(self, job) -> None:
        """Delete a single orphan job with confirmation"""

        def handle_delete(confirmed: bool) -> None:
            if confirmed:
                success, msg = self.state_provider.delete_job_safely(job)
                if success:
                    self.notify(msg, severity="information")
                    self.refresh_orphans()
                else:
                    self.notify(msg, severity="error")

        self.app.push_screen(
            DeleteConfirmScreen("orphan job", job.identifier),
            handle_delete,
        )

    def action_delete_all(self) -> None:
        """Delete all orphan jobs"""
        if not self.orphan_jobs:
            self.notify("No orphan jobs to delete", severity="warning")
            return

        # Filter out running jobs
        deletable_jobs = [j for j in self.orphan_jobs if not j.state.running()]

        if not deletable_jobs:
            self.notify("All orphan jobs are running", severity="warning")
            return

        def handle_delete_all(confirmed: bool) -> None:
            if confirmed:
                deleted = 0
                for job in deletable_jobs:
                    success, _ = self.state_provider.delete_job_safely(
                        job, cascade_orphans=False
                    )
                    if success:
                        deleted += 1

                # Clean up orphan partials once at the end
                self.state_provider.cleanup_orphan_partials(perform=True)

                self.notify(f"Deleted {deleted} orphan jobs", severity="information")
                self.refresh_orphans()

        self.app.push_screen(
            DeleteConfirmScreen(
                "all orphan jobs",
                f"{len(deletable_jobs)} jobs",
                "This action cannot be undone",
            ),
            handle_delete_all,
        )

    def action_refresh(self) -> None:
        """Refresh the orphan jobs list"""
        self.refresh_orphans()

    def action_go_back(self) -> None:
        """Go back to main screen"""
        self.dismiss()


class HelpScreen(ModalScreen[None]):
    """Modal screen showing keyboard shortcuts"""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("?", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll

        help_text = """
[bold]Keyboard Shortcuts[/bold]

[bold cyan]Navigation[/bold cyan]
  q         Quit application
  Esc       Go back / Close dialog
  r         Refresh data
  ?         Show this help

[bold cyan]Experiments[/bold cyan]
  Enter     Select experiment
  d         Delete experiment
  k         Kill all running jobs

[bold cyan]Jobs[/bold cyan]
  l         View job logs
  d         Delete job
  k         Kill running job
  /         Open search filter
  c         Clear search filter
  S         Sort by status
  T         Sort by task
  D         Sort by date
  f         Copy folder path

[bold cyan]Search Filter[/bold cyan]
  Enter     Apply filter
  Esc       Close and clear filter

[bold cyan]Orphan Jobs[/bold cyan]
  o         Show orphan jobs
  T         Sort by task
  Z         Sort by size
  d         Delete selected
  D         Delete all
  f         Copy folder path
"""
        with Vertical(id="help-dialog"):
            yield Static("Experimaestro Help", id="help-title")
            with VerticalScroll(id="help-scroll"):
                yield Static(help_text, id="help-content")
            yield Button("Close", id="help-close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


class ExperimaestroUI(App):
    """Textual TUI for monitoring experiments"""

    TITLE = "Experimaestro UI"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "show_help", "Help"),
        Binding("escape", "go_back", "Back", show=False),
        Binding("l", "view_logs", "Logs", show=False),
        Binding("o", "show_orphans", "Orphans", show=False),
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

    def action_show_orphans(self) -> None:
        """Show orphan jobs screen"""
        self.push_screen(OrphanJobsScreen(self.state_provider))

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

    def on_view_job_logs_request(self, message: ViewJobLogsRequest) -> None:
        """Handle log viewing request from jobs table"""
        job = self.state_provider.get_job(message.job_id, message.experiment_id)
        if not job or not job.path or not job.task_id:
            self.notify("Cannot find job logs", severity="warning")
            return
        self.post_message(ViewJobLogs(str(job.path), job.task_id))

    def on_delete_job_request(self, message: DeleteJobRequest) -> None:
        """Handle job deletion request"""
        job = self.state_provider.get_job(message.job_id, message.experiment_id)
        if not job:
            self.notify("Job not found", severity="error")
            return

        if job.state.running():
            self.notify("Cannot delete a running job", severity="warning")
            return

        # Save cursor position to restore after delete
        jobs_table = self.query_one(JobsTable)
        table = jobs_table.query_one("#jobs-table", DataTable)
        cursor_row = table.cursor_row

        def handle_delete_response(confirmed: bool) -> None:
            if confirmed:
                success, msg = self.state_provider.delete_job_safely(job)
                if success:
                    self.notify(msg, severity="information")
                    self.action_refresh()
                    # Move cursor to previous row (or first if was at top)
                    if cursor_row is not None and table.row_count > 0:
                        new_row = min(cursor_row, table.row_count - 1)
                        if new_row > 0 and cursor_row > 0:
                            new_row = cursor_row - 1
                        table.move_cursor(row=new_row)
                else:
                    self.notify(msg, severity="error")

        self.push_screen(
            DeleteConfirmScreen("job", job.identifier),
            handle_delete_response,
        )

    def on_delete_experiment_request(self, message: DeleteExperimentRequest) -> None:
        """Handle experiment deletion request"""
        jobs = self.state_provider.get_jobs(message.experiment_id)
        running_jobs = [j for j in jobs if j.state.running()]

        if running_jobs:
            self.notify(
                f"Cannot delete: {len(running_jobs)} jobs are running",
                severity="warning",
            )
            return

        warning = (
            f"{len(jobs)} jobs will remain (not deleted by default)" if jobs else None
        )

        def handle_delete_response(confirmed: bool) -> None:
            if confirmed:
                success, msg = self.state_provider.delete_experiment(
                    message.experiment_id, delete_jobs=False
                )
                if success:
                    self.notify(msg, severity="information")
                    # Go back to experiments list
                    experiments_list = self.query_one(ExperimentsList)
                    experiments_list.expand_experiments()
                    self.post_message(ExperimentDeselected())
                    self.action_refresh()
                else:
                    self.notify(msg, severity="error")

        self.push_screen(
            DeleteConfirmScreen("experiment", message.experiment_id, warning),
            handle_delete_response,
        )

    def on_kill_job_request(self, message: KillJobRequest) -> None:
        """Handle job kill request"""
        job = self.state_provider.get_job(message.job_id, message.experiment_id)
        if not job:
            self.notify("Job not found", severity="error")
            return

        if not job.state.running():
            self.notify("Job is not running", severity="warning")
            return

        def handle_kill_response(confirmed: bool) -> None:
            if confirmed:
                success = self.state_provider.kill_job(job, perform=True)
                if success:
                    self.notify(f"Job {job.identifier} killed", severity="information")
                    self.action_refresh()
                else:
                    self.notify("Failed to kill job", severity="error")

        self.push_screen(
            KillConfirmScreen("job", job.identifier),
            handle_kill_response,
        )

    def on_kill_experiment_request(self, message: KillExperimentRequest) -> None:
        """Handle experiment kill request (kill all running jobs)"""
        jobs = self.state_provider.get_jobs(message.experiment_id)
        running_jobs = [j for j in jobs if j.state.running()]

        if not running_jobs:
            self.notify("No running jobs in experiment", severity="warning")
            return

        def handle_kill_response(confirmed: bool) -> None:
            if confirmed:
                killed = 0
                for job in running_jobs:
                    if self.state_provider.kill_job(job, perform=True):
                        killed += 1
                self.notify(
                    f"Killed {killed} of {len(running_jobs)} running jobs",
                    severity="information",
                )
                self.action_refresh()

        self.push_screen(
            KillConfirmScreen("experiment", f"{len(running_jobs)} running jobs"),
            handle_kill_response,
        )

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

    def action_show_help(self) -> None:
        """Show help screen with keyboard shortcuts"""
        self.push_screen(HelpScreen())

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
