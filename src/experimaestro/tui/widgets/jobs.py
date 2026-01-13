"""Jobs-related widgets for the TUI"""

from datetime import datetime
from typing import Optional
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Label, Input, Static
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tui.utils import format_duration, get_status_icon
from experimaestro.tui.messages import (
    JobSelected,
    ViewJobLogs,
    ViewJobLogsRequest,
    DeleteJobRequest,
    KillJobRequest,
    FilterChanged,
    SearchApplied,
)


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


class JobDetailView(Widget):
    """Widget displaying detailed job information"""

    BINDINGS = [
        Binding("l", "view_logs", "View Logs", priority=True),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.current_job_id: Optional[str] = None
        self.current_experiment_id: Optional[str] = None
        self.job_data: Optional[dict] = None
        self.tags_map: dict[str, dict[str, str]] = {}  # job_id -> {tag_key: tag_value}
        self.dependencies_map: dict[
            str, list[str]
        ] = {}  # job_id -> [depends_on_job_ids]

    def compose(self) -> ComposeResult:
        yield Label("Job Details", classes="section-title")
        yield Static(
            "Loading job details...", id="job-detail-loading", classes="hidden"
        )
        with VerticalScroll(id="job-detail-content"):
            yield Label("", id="job-id-label")
            yield Label("", id="job-task-label")
            yield Label("", id="job-status-label")
            yield Label("", id="job-path-label")
            yield Label("", id="job-times-label")
            yield Label("Process:", classes="subsection-title")
            yield Label("", id="job-process-label")
            yield Label("Tags:", classes="subsection-title")
            yield Label("", id="job-tags-label")
            yield Label("Dependencies:", classes="subsection-title")
            yield Label("", id="job-dependencies-label")
            yield Label("Progress:", classes="subsection-title")
            yield Label("", id="job-progress-label")
            yield Label("", id="job-logs-hint")

    def action_view_logs(self) -> None:
        """View job logs with toolong"""
        if self.job_data and self.job_data.path and self.job_data.task_id:
            self.post_message(
                ViewJobLogs(
                    str(self.job_data.path), self.job_data.task_id, self.job_data.state
                )
            )

    def _get_process_info(self, job) -> str:
        """Get process information for a job using the state provider"""
        pinfo = self.state_provider.get_process_info(job)

        if pinfo is None:
            if job.state and job.state.finished():
                return "(process completed)"
            return "(no process info)"

        # Build process info string
        parts = [f"PID: [bold]{pinfo.pid}[/bold]", f"Type: {pinfo.type}"]

        if pinfo.running:
            if pinfo.cpu_percent is not None:
                parts.append(f"CPU: {pinfo.cpu_percent:.1f}%")
            if pinfo.memory_mb is not None:
                parts.append(f"Mem: {pinfo.memory_mb:.1f}MB")
            if pinfo.num_threads is not None:
                parts.append(f"Threads: {pinfo.num_threads}")
        elif job.state and job.state.running():
            parts.append("[dim](process not found)[/dim]")

        return " | ".join(parts)

    def set_job(self, job_id: str, experiment_id: str) -> None:
        """Set the job to display"""
        self.current_job_id = job_id
        self.current_experiment_id = experiment_id

        # Show loading for remote
        if self.state_provider.is_remote:
            self.query_one("#job-detail-loading", Static).remove_class("hidden")

        # Load in background
        self._load_job_detail(job_id, experiment_id)

    @work(thread=True, exclusive=True, group="job_detail_load")
    def _load_job_detail(self, job_id: str, experiment_id: str) -> None:
        """Load job details in background thread"""
        # Load tags and dependencies if needed
        tags_map = self.state_provider.get_tags_map(experiment_id)
        deps_map = self.state_provider.get_dependencies_map(experiment_id)
        job = self.state_provider.get_job(job_id, experiment_id)

        self.app.call_from_thread(self._on_job_loaded, job, tags_map, deps_map)

    def _on_job_loaded(self, job, tags_map: dict, deps_map: dict) -> None:
        """Handle loaded job on main thread"""
        self.query_one("#job-detail-loading", Static).add_class("hidden")
        self.tags_map = tags_map
        self.dependencies_map = deps_map

        if not job:
            self.log(f"Job not found: {self.current_job_id}")
            return

        self._update_job_display(job)

    def refresh_job_detail(self) -> None:
        """Refresh job details from state provider"""
        if not self.current_job_id or not self.current_experiment_id:
            return

        if self.state_provider.is_remote:
            self._load_job_detail(self.current_job_id, self.current_experiment_id)
        else:
            job = self.state_provider.get_job(
                self.current_job_id, self.current_experiment_id
            )
            if job:
                self._update_job_display(job)

    def _update_job_display(self, job) -> None:
        """Update the display with job data"""
        self.job_data = job

        # Update labels
        self.query_one("#job-id-label", Label).update(f"Job ID: {job.identifier}")
        self.query_one("#job-task-label", Label).update(f"Task: {job.task_id}")

        # Format status with icon and name
        status_name = job.state.name if job.state else "unknown"
        failure_reason = getattr(job, "failure_reason", None)
        transient = getattr(job, "transient", None)
        status_icon = get_status_icon(status_name, failure_reason, transient)
        status_text = f"{status_icon} {status_name}"
        if failure_reason:
            status_text += f" ({failure_reason.name})"

        self.query_one("#job-status-label", Label).update(f"Status: {status_text}")

        # Path (from locator)
        locator = job.locator or "-"
        self.query_one("#job-path-label", Label).update(f"Locator: {locator}")

        # Times - format timestamps (now datetime objects)
        def format_time(ts):
            if ts:
                return ts.strftime("%Y-%m-%d %H:%M:%S")
            return "-"

        submitted = format_time(job.submittime)
        start = format_time(job.starttime)
        end = format_time(job.endtime)

        # Calculate duration (starttime/endtime are now datetime objects)
        duration = "-"
        if job.starttime:
            if job.endtime:
                duration = format_duration(
                    (job.endtime - job.starttime).total_seconds()
                )
            else:
                duration = (
                    format_duration((datetime.now() - job.starttime).total_seconds())
                    + " (running)"
                )

        times_text = f"Submitted: {submitted} | Start: {start} | End: {end} | Duration: {duration}"
        self.query_one("#job-times-label", Label).update(times_text)

        # Process information
        process_text = self._get_process_info(job)
        self.query_one("#job-process-label", Label).update(process_text)

        # Tags are stored in JobTagModel, accessed via tags_map
        tags = self.tags_map.get(job.identifier, {})
        if tags:
            tags_text = ", ".join(f"{k}={v}" for k, v in tags.items())
        else:
            tags_text = "(no tags)"
        self.query_one("#job-tags-label", Label).update(tags_text)

        # Dependencies are stored in JobDependenciesModel, accessed via dependencies_map
        depends_on = self.dependencies_map.get(job.identifier, [])
        if depends_on:
            # Try to get task IDs for the dependency jobs
            dep_texts = []
            for dep_job_id in depends_on:
                dep_job = self.state_provider.get_job(
                    dep_job_id, self.current_experiment_id
                )
                if dep_job:
                    dep_task_name = dep_job.task_id.split(".")[-1]
                    dep_texts.append(f"{dep_task_name} ({dep_job_id[:8]}...)")
                else:
                    dep_texts.append(f"{dep_job_id[:8]}...")
            dependencies_text = ", ".join(dep_texts)
        else:
            dependencies_text = "(no dependencies)"
        self.query_one("#job-dependencies-label", Label).update(dependencies_text)

        # Progress
        progress_list = job.progress or []
        if progress_list:
            progress_lines = []
            for p in progress_list:
                level = p.level
                pct = p.progress * 100
                desc = p.desc or ""
                indent = "  " * level

                # Create visual progress bar (20 chars wide)
                bar_width = 20
                filled = int(p.progress * bar_width)
                remaining = bar_width - filled

                # Use Unicode block characters with colors
                filled_bar = "█" * filled
                remaining_bar = "░" * remaining

                # Color based on progress level
                if pct >= 100:
                    bar_color = "green"
                elif pct >= 50:
                    bar_color = "cyan"
                else:
                    bar_color = "yellow"

                # Format: [bar] percentage description
                bar_text = f"[{bar_color}]{filled_bar}[/][dim]{remaining_bar}[/]"
                pct_text = f"[bold]{pct:5.1f}%[/bold]"
                desc_text = f" [italic]{desc}[/]" if desc else ""

                progress_lines.append(f"{indent}{bar_text} {pct_text}{desc_text}")
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


class JobsTable(Vertical):
    """Widget displaying jobs for selected experiment"""

    BINDINGS = [
        Binding("ctrl+d", "delete_job", "Delete", show=False),
        Binding("k", "kill_job", "Kill", show=False),
        Binding("l", "view_logs", "Logs", key_display="l"),
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("/", "toggle_search", "Search"),
        Binding("c", "clear_filter", "Clear", show=False),
        Binding("r", "refresh_live", "Refresh"),
        Binding("S", "sort_by_status", "Sort ⚑", show=False),
        Binding("T", "sort_by_task", "Sort Task", show=False),
        Binding("D", "sort_by_submitted", "Sort Date", show=False),
        Binding("escape", "clear_search", show=False, priority=True),
    ]

    # Track current sort state
    _sort_column: Optional[str] = None
    _sort_reverse: bool = False
    _needs_rebuild: bool = True  # Start with rebuild needed

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.filter_fn = None
        self.current_experiment: Optional[str] = None
        self.current_run_id: Optional[str] = None
        self.is_past_run: bool = False
        self.tags_map: dict[str, dict[str, str]] = {}  # job_id -> {tag_key: tag_value}
        self.dependencies_map: dict[
            str, list[str]
        ] = {}  # job_id -> [depends_on_job_ids]

    def compose(self) -> ComposeResult:
        yield Static("", id="past-run-banner", classes="hidden")
        yield Static("Loading jobs...", id="jobs-loading", classes="hidden")
        yield SearchBar()
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

    def action_refresh_live(self) -> None:
        """Refresh the jobs table"""
        self.refresh_jobs()
        self.notify("Jobs refreshed", severity="information")

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
            if job.path:
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

    # Failure reason sort order (within error status)
    # More actionable failures first
    FAILURE_ORDER = {
        "TIMEOUT": 0,  # Might just need retry
        "MEMORY": 1,  # Might need resource adjustment
        "DEPENDENCY": 2,  # Need to fix upstream job first
        "FAILED": 3,  # Generic failure
    }

    @classmethod
    def _get_status_sort_key(cls, job):
        """Get sort key for a job based on status and failure reason.

        Returns tuple (status_order, failure_order) for proper sorting.
        """
        state_name = job.state.name if job.state else "unknown"
        status_order = cls.STATUS_ORDER.get(state_name, 99)

        # For error jobs, also sort by failure reason
        if state_name == "error":
            failure_reason = getattr(job, "failure_reason", None)
            if failure_reason:
                failure_order = cls.FAILURE_ORDER.get(failure_reason.name, 99)
            else:
                failure_order = 99  # Unknown failure at end
        else:
            failure_order = 0

        return (status_order, failure_order)

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

    def set_experiment(
        self,
        experiment_id: Optional[str],
        run_id: Optional[str] = None,
        is_past_run: bool = False,
    ) -> None:
        """Set the current experiment and refresh jobs

        Args:
            experiment_id: The experiment ID to show jobs for
            run_id: The specific run ID (optional)
            is_past_run: Whether this is a past (non-current) run
        """
        self.current_experiment = experiment_id
        self.current_run_id = run_id
        self.is_past_run = is_past_run

        # Update the past run banner
        banner = self.query_one("#past-run-banner", Static)
        if is_past_run and run_id:
            banner.update(f"[bold yellow]Viewing past run: {run_id}[/bold yellow]")
            banner.remove_class("hidden")
        else:
            banner.add_class("hidden")

        # Clear table and show loading for remote providers
        if self.state_provider.is_remote:
            table = self.query_one("#jobs-table", DataTable)
            table.clear()
            self.query_one("#jobs-loading", Static).remove_class("hidden")

        # Load data in background worker
        self._load_experiment_data(experiment_id, run_id)

    @work(thread=True, exclusive=True, group="jobs_load")
    def _load_experiment_data(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> None:
        """Load experiment data in background thread"""
        if not experiment_id:
            self.tags_map = {}
            self.dependencies_map = {}
            self.app.call_from_thread(self._on_data_loaded, [])
            return

        # Fetch data (this is the slow part for remote)
        tags_map = self.state_provider.get_tags_map(experiment_id, run_id)
        dependencies_map = self.state_provider.get_dependencies_map(
            experiment_id, run_id
        )
        jobs = self.state_provider.get_jobs(experiment_id, run_id=run_id)

        # Update on main thread
        self.app.call_from_thread(
            self._on_data_loaded, jobs, tags_map, dependencies_map
        )

    def _on_data_loaded(
        self,
        jobs: list,
        tags_map: dict = None,
        dependencies_map: dict = None,
    ) -> None:
        """Handle loaded data on main thread"""
        # Hide loading indicator
        self.query_one("#jobs-loading", Static).add_class("hidden")

        # Update maps
        if tags_map is not None:
            self.tags_map = tags_map
        if dependencies_map is not None:
            self.dependencies_map = dependencies_map

        # Refresh display with loaded jobs
        self._refresh_jobs_with_data(jobs)

    def refresh_jobs(self) -> None:
        """Refresh the jobs list from state provider

        For remote providers, this runs in background. For local, it's synchronous.
        """
        if not self.current_experiment:
            return

        if self.state_provider.is_remote:
            # Use background worker for remote
            self._load_experiment_data(self.current_experiment, self.current_run_id)
        else:
            # Synchronous for local (fast)
            jobs = self.state_provider.get_jobs(
                self.current_experiment, run_id=self.current_run_id
            )
            self._refresh_jobs_with_data(jobs)

    def _refresh_jobs_with_data(self, jobs: list) -> None:  # noqa: C901
        """Refresh the jobs display with provided job data"""
        table = self.query_one("#jobs-table", DataTable)

        self.log.debug(
            f"Refreshing jobs for {self.current_experiment}/{self.current_run_id}: {len(jobs)} jobs"
        )

        # Apply filter if set
        if self.filter_fn:
            jobs = [j for j in jobs if self.filter_fn(j)]
            self.log.debug(f"After filter: {len(jobs)} jobs")

        # Sort jobs based on selected column
        if self._sort_column == "status":
            # Sort by status priority, then by failure reason for errors
            jobs.sort(
                key=self._get_status_sort_key,
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
                key=lambda j: j.submittime or datetime.max,
                reverse=self._sort_reverse,
            )

        # Check if we need to rebuild (new/removed jobs, or status changed when sorting by status)
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
                    # We only report main progress here (level 0)
                    last_progress = progress_list[0]
                    progress_pct = last_progress.progress * 100
                    status_text = f"▶ {progress_pct:.0f}%"
                else:
                    status_text = "▶"
            else:
                failure_reason = getattr(job, "failure_reason", None)
                transient = getattr(job, "transient", None)
                status_text = get_status_icon(status, failure_reason, transient)

            # Tags are stored in JobTagModel, accessed via tags_map
            job_tags = self.tags_map.get(job.identifier, {})
            if job_tags:
                tags_text = Text()
                for i, (k, v) in enumerate(job_tags.items()):
                    if i > 0:
                        tags_text.append(", ")
                    tags_text.append(f"{k}", style="bold")
                    tags_text.append(f"={v}")
            else:
                tags_text = Text("-")

            submitted = "-"
            if job.submittime:
                submitted = job.submittime.strftime("%Y-%m-%d %H:%M")

            # Calculate duration (starttime/endtime are now datetime objects)
            start = job.starttime
            end = job.endtime
            duration = "-"
            if start:
                if end:
                    elapsed = (end - start).total_seconds()
                else:
                    elapsed = (datetime.now() - start).total_seconds()
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
            for idx, (job_id, row_data) in enumerate(rows_data.items()):
                table.add_row(*row_data, key=job_id)
                if selected_key == job_id:
                    new_cursor_row = idx

            if new_cursor_row is not None and table.row_count > 0:
                table.move_cursor(row=new_cursor_row)
        else:
            # Just update cells in place - no reordering needed
            for job_id, row_data in rows_data.items():
                (
                    job_id_short,
                    task_id,
                    status_text,
                    tags_text,
                    submitted,
                    duration,
                ) = row_data
                table.update_cell(job_id, "job_id", job_id_short, update_width=True)
                table.update_cell(job_id, "task", task_id, update_width=True)
                table.update_cell(job_id, "status", status_text, update_width=True)
                table.update_cell(job_id, "tags", tags_text, update_width=True)
                table.update_cell(job_id, "submitted", submitted, update_width=True)
                table.update_cell(job_id, "duration", duration, update_width=True)

        self.log.debug(
            f"Jobs table now has {table.row_count} rows (rebuild={needs_rebuild})"
        )

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
