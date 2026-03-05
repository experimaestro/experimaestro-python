"""Jobs-related widgets for the TUI"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Label, Input, Static, Tree
from textual.widgets.tree import TreeNode
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text

from experimaestro.scheduler.interfaces import JobState
from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tui.utils import format_duration
from experimaestro.tui.messages import (
    JobSelected,
    JobHighlighted,
    ViewJobLogs,
    ViewJobLogsRequest,
    DeleteJobRequest,
    KillJobRequest,
    FilterChanged,
    SearchApplied,
)
from experimaestro.carbon.utils import (
    format_co2_kg,
    format_energy_kwh,
    format_power,
)

# Progress bar using Unicode block characters
_PROGRESS_BLOCKS = " ▏▎▍▌▋▊▉█"


def _format_progress_bar(fraction: float, width: int = 10) -> Text:
    """Render a compact progress bar with percentage using Unicode blocks"""
    fraction = max(0.0, min(1.0, fraction))
    pct = int(fraction * 100)
    filled_float = fraction * width
    full = int(filled_float)
    remainder = filled_float - full

    bar = "█" * full
    if full < width:
        bar += _PROGRESS_BLOCKS[int(remainder * 8)]
        bar += " " * (width - full - 1)

    text = Text()
    text.append(bar, style="bold green")
    text.append(f" {pct:3d}%", style="green")
    return text


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
        self.current_task_id: Optional[str] = None
        self.current_experiment_id: Optional[str] = None
        self.job_data: Optional[dict] = None
        self.tags_map: dict[str, dict[str, str]] = {}  # job_id -> {tag_key: tag_value}
        self.dependencies_map: dict[
            str, list[str]
        ] = {}  # job_id -> [depends_on_job_ids]
        # Mapping from job_id to task_id for dependencies lookup
        self.job_task_id_map: dict[str, str] = {}

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
            yield Label("Carbon Impact:", classes="subsection-title")
            yield Label("", id="job-carbon-label")
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

    def _get_carbon_info(self, job) -> str:
        """Get carbon impact information for a job"""
        carbon_metrics = getattr(job, "carbon_metrics", None)

        if carbon_metrics is None:
            if job.state and job.state.finished():
                return "(no carbon data)"
            return "(carbon tracking not started)"

        # Build carbon info string with detailed breakdown
        parts = []

        # CO2 emissions (primary metric)
        co2_str = format_co2_kg(carbon_metrics.co2_kg)
        parts.append(f"[bold green]CO2: {co2_str}[/bold green]")

        # Energy consumption
        energy_str = format_energy_kwh(carbon_metrics.energy_kwh)
        parts.append(f"Energy: {energy_str}")

        # Power breakdown (if available)
        power_parts = []
        if carbon_metrics.cpu_power_w > 0:
            power_parts.append(f"CPU: {format_power(carbon_metrics.cpu_power_w)}")
        if carbon_metrics.gpu_power_w > 0:
            power_parts.append(f"GPU: {format_power(carbon_metrics.gpu_power_w)}")
        if carbon_metrics.ram_power_w > 0:
            power_parts.append(f"RAM: {format_power(carbon_metrics.ram_power_w)}")
        if power_parts:
            parts.append(f"Avg Power: {', '.join(power_parts)}")

        # Region
        if carbon_metrics.region:
            parts.append(f"Region: {carbon_metrics.region}")

        # Status indicator
        if carbon_metrics.is_final:
            parts.append("[dim](final)[/dim]")
        elif job.state and job.state.running():
            parts.append("[cyan](live)[/cyan]")

        return " | ".join(parts)

    def set_job(self, job_id: str, task_id: str, experiment_id: str) -> None:
        """Set the job to display"""
        self.current_job_id = job_id
        self.current_task_id = task_id
        self.current_experiment_id = experiment_id

        # Show loading for remote
        if self.state_provider.is_remote:
            self.query_one("#job-detail-loading", Static).remove_class("hidden")

        # Load in background
        self._load_job_detail(job_id, task_id, experiment_id)

    @work(thread=True, exclusive=True, group="job_detail_load")
    def _load_job_detail(self, job_id: str, task_id: str, experiment_id: str) -> None:
        """Load job details in background thread"""
        # Load tags and dependencies if needed
        tags_map = self.state_provider.get_tags_map(experiment_id)
        deps_map = self.state_provider.get_dependencies_map(experiment_id)
        experiment_job_info = self.state_provider.get_experiment_job_info(experiment_id)

        # Load all jobs for the experiment to build job_id -> task_id mapping
        jobs = self.state_provider.get_jobs(experiment_id)
        job_task_map = {j.identifier: j.task_id or "" for j in jobs}

        job = self.state_provider.get_job(task_id, job_id)

        self.app.call_from_thread(
            self._on_job_loaded,
            job,
            tags_map,
            deps_map,
            job_task_map,
            experiment_job_info,
        )

    def _on_job_loaded(
        self,
        job,
        tags_map: dict,
        deps_map: dict,
        job_task_map: dict,
        experiment_job_info: dict = None,
    ) -> None:
        """Handle loaded job on main thread"""
        self.query_one("#job-detail-loading", Static).add_class("hidden")
        self.tags_map = tags_map
        self.dependencies_map = deps_map
        self.job_task_id_map = job_task_map
        if experiment_job_info is not None:
            self.experiment_job_info = experiment_job_info

        if not job:
            self.log(f"Job not found: {self.current_job_id}")
            return

        self._update_job_display(job)

    def refresh_job_detail(self) -> None:
        """Refresh job details from state provider (always background)"""
        if not self.current_job_id or not self.current_task_id:
            return

        self._load_job_detail(
            self.current_job_id,
            self.current_task_id,
            self.current_experiment_id or "",
        )

    def _update_job_display(self, job) -> None:
        """Update the display with job data"""
        self.job_data = job

        # Update labels
        self.query_one("#job-id-label", Label).update(f"Job ID: {job.identifier}")
        self.query_one("#job-task-label", Label).update(f"Task: {job.task_id}")

        # Format status with icon and name
        status_name = job.state.name if job.state else "unknown"
        status_icon = job.state.icon if job.state else "❓"
        status_text = f"{status_icon} {status_name}"
        failure_reason = job.state.failure_reason if job.state else None
        if failure_reason:
            status_text += f" ({failure_reason.name})"

        self.query_one("#job-status-label", Label).update(f"Status: {status_text}")

        # Path - show remote path for remote providers, local path otherwise
        display_path = self.state_provider.get_display_path(job) if job.path else "-"
        self.query_one("#job-path-label", Label).update(f"Path: {display_path}")

        # Times - format timestamps (now datetime objects)
        def format_time(ts):
            if ts:
                return ts.strftime("%Y-%m-%d %H:%M:%S")
            return "-"

        # Get submittime from experiment job info
        job_info = getattr(self, "experiment_job_info", {}).get(job.identifier)
        submittime_dt = (
            datetime.fromtimestamp(job_info.timestamp)
            if job_info and job_info.timestamp
            else None
        )
        submitted = format_time(submittime_dt)
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

        # Show last state check if available from the process
        if job._process is not None:
            last_check = job._process.last_state_check
            if last_check is not None:
                ago = int((datetime.now() - last_check).total_seconds())
                times_text += f" | Last check: {ago}s ago"

        self.query_one("#job-times-label", Label).update(times_text)

        # Process information
        process_text = self._get_process_info(job)
        self.query_one("#job-process-label", Label).update(process_text)

        # Carbon impact information
        carbon_text = self._get_carbon_info(job)
        self.query_one("#job-carbon-label", Label).update(carbon_text)

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
            # Try to get task IDs for the dependency jobs from our mapping
            dep_texts = []
            for dep_job_id in depends_on:
                dep_task_id = self.job_task_id_map.get(dep_job_id, "")
                if dep_task_id:
                    dep_task_name = dep_task_id.split(".")[-1]
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


class GroupLevel(Enum):
    """Grouping levels for the tree view"""

    STATUS = "status"
    TASK_ID = "task_id"
    TAG = "tag"


# Default grouping order
DEFAULT_GROUP_ORDER = [GroupLevel.STATUS, GroupLevel.TASK_ID, GroupLevel.TAG]

# Display names for group levels
GROUP_LEVEL_NAMES = {
    GroupLevel.STATUS: "Status",
    GroupLevel.TASK_ID: "Task",
    GroupLevel.TAG: "Tag",
}


@dataclass
class TreeJobData:
    """Data attached to a job leaf node in the tree"""

    job_id: str
    task_id: str
    status_text: str
    tags: dict[str, str]


@dataclass
class TreeGroupData:
    """Data attached to a group node in the tree"""

    level: GroupLevel
    value: str
    job_count: int


class JobsTreeView(Widget):
    """Tree-based job view with configurable grouping hierarchy"""

    BINDINGS = [
        Binding("g", "cycle_grouping", "Cycle Group"),
        Binding("1", "toggle_level('status')", "Tog Status", show=False),
        Binding("2", "toggle_level('task_id')", "Tog Task", show=False),
        Binding("3", "toggle_level('tag')", "Tog Tag", show=False),
        Binding("e", "expand_all", "Expand All", show=False),
        Binding("E", "collapse_all", "Collapse All", show=False),
        Binding("enter", "select_node", "Select", show=False),
        Binding("l", "view_logs", "Logs", key_display="l"),
        Binding("ctrl+d", "delete_job", "Delete", show=False),
        Binding("ctrl+k", "kill_job", "Kill", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.grouping_order: list[GroupLevel] = list(DEFAULT_GROUP_ORDER)
        self._disabled_levels: set[GroupLevel] = set()
        self._jobs: list = []
        self._tags_map: dict[str, dict[str, str]] = {}
        self._experiment_obj = None
        self._experiment_job_info: dict = {}
        self._task_id_map: dict[str, str] = {}
        self.current_experiment: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="tree-grouping-indicator")
        yield Tree("Jobs", id="jobs-tree")

    def on_mount(self) -> None:
        tree = self.query_one("#jobs-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3
        self._update_grouping_indicator()

    def _active_levels(self) -> list[GroupLevel]:
        """Return the grouping order with disabled levels removed"""
        return [lvl for lvl in self.grouping_order if lvl not in self._disabled_levels]

    def _update_grouping_indicator(self) -> None:
        parts = []
        for lvl in self.grouping_order:
            name = GROUP_LEVEL_NAMES[lvl]
            idx = list(GroupLevel).index(lvl) + 1
            if lvl in self._disabled_levels:
                part = f"[dim]{idx}[/dim]:[on red]{name}[/on red]"
            else:
                part = f"[dim]{idx}[/dim]:[on green]{name}[/on green]"
            parts.append(part)
        separator = " [bold]>[/bold] "
        indicator = self.query_one("#tree-grouping-indicator", Static)
        indicator.update(
            f"[bold]Grouping:[/bold] {separator.join(parts)}  [dim](g cycle)[/dim]"
        )

    def action_cycle_grouping(self) -> None:
        """Rotate the grouping order: move first level to end"""
        self.grouping_order = self.grouping_order[1:] + self.grouping_order[:1]
        self._update_grouping_indicator()
        self.rebuild_tree()
        active = self._active_levels()
        order_str = " > ".join(GROUP_LEVEL_NAMES[lvl] for lvl in active)
        self.notify(f"Grouping: {order_str}", severity="information")

    def action_toggle_level(self, level_value: str) -> None:
        """Toggle a grouping level on/off"""
        level = GroupLevel(level_value)
        if level in self._disabled_levels:
            self._disabled_levels.discard(level)
            self.notify(
                f"{GROUP_LEVEL_NAMES[level]} grouping enabled",
                severity="information",
            )
        else:
            # Don't allow disabling all levels
            active = self._active_levels()
            if len(active) <= 1 and level in active:
                self.notify("Cannot disable all grouping levels", severity="warning")
                return
            self._disabled_levels.add(level)
            self.notify(
                f"{GROUP_LEVEL_NAMES[level]} grouping disabled",
                severity="information",
            )
        self._update_grouping_indicator()
        self.rebuild_tree()

    def _expand_recursive(self, node: TreeNode, expand: bool) -> None:
        """Expand or collapse all group nodes recursively"""
        for child in node.children:
            if isinstance(child.data, TreeGroupData):
                if expand:
                    child.expand()
                else:
                    child.collapse()
                self._expand_recursive(child, expand)

    def action_expand_all(self) -> None:
        """Expand all tree nodes"""
        tree = self.query_one("#jobs-tree", Tree)
        self._expand_recursive(tree.root, True)

    def action_collapse_all(self) -> None:
        """Collapse all tree nodes"""
        tree = self.query_one("#jobs-tree", Tree)
        self._expand_recursive(tree.root, False)

    def set_data(
        self,
        jobs: list,
        tags_map: dict[str, dict[str, str]],
        experiment_obj,
        experiment_job_info: dict,
        task_id_map: dict[str, str],
        current_experiment: str | None,
    ) -> None:
        """Update the tree with new job data"""
        self._jobs = jobs
        self._tags_map = tags_map
        self._experiment_obj = experiment_obj
        self._experiment_job_info = experiment_job_info
        self._task_id_map = task_id_map
        self.current_experiment = current_experiment
        self.rebuild_tree()

    def _collect_expanded_paths(self, node: TreeNode, path: tuple = ()) -> set[tuple]:
        """Collect paths of expanded group nodes"""
        expanded = set()
        for child in node.children:
            if isinstance(child.data, TreeGroupData):
                child_path = path + (child.data.value,)
                if child.is_expanded:
                    expanded.add(child_path)
                expanded |= self._collect_expanded_paths(child, child_path)
        return expanded

    def _restore_expanded(
        self, node: TreeNode, expanded: set[tuple], path: tuple = ()
    ) -> None:
        """Restore expanded state of group nodes"""
        for child in node.children:
            if isinstance(child.data, TreeGroupData):
                child_path = path + (child.data.value,)
                if child_path in expanded:
                    child.expand()
                self._restore_expanded(child, expanded, child_path)

    def rebuild_tree(self) -> None:
        """Rebuild the tree from current data, preserving expanded state"""
        tree = self.query_one("#jobs-tree", Tree)

        # Save expanded state before clearing
        expanded_paths = self._collect_expanded_paths(tree.root)
        first_build = not bool(tree.root.children)

        tree.clear()

        if not self._jobs:
            tree.root.add_leaf("[dim]No jobs[/dim]")
            return

        # Build enriched job records for grouping
        enriched = []
        for job in self._jobs:
            resolved_state, _ = StateProvider.get_resolved_state(
                job, self._experiment_obj
            )
            status_name = resolved_state.name if resolved_state else "unknown"
            status_icon = resolved_state.icon if resolved_state else "?"
            tags = self._tags_map.get(job.identifier, {})

            # Progress (for running jobs)
            progress_text = ""
            progress_fraction = -1.0
            if resolved_state and resolved_state == JobState.RUNNING:
                progress_list = job.progress or []
                if progress_list:
                    progress_fraction = progress_list[0].progress
                    progress_text = f"{progress_fraction * 100:.0f}%"

            # Duration
            start = job.starttime
            end = job.endtime
            duration = ""
            if start:
                if end:
                    elapsed = (end - start).total_seconds()
                else:
                    elapsed = (datetime.now() - start).total_seconds()
                duration = format_duration(elapsed)

            # Submitted time
            submitted = ""
            job_info = self._experiment_job_info.get(job.identifier)
            if job_info and job_info.timestamp:
                submitted = datetime.fromtimestamp(job_info.timestamp).strftime(
                    "%m-%d %H:%M"
                )

            # Carbon
            carbon_metrics = getattr(job, "carbon_metrics", None)
            co2_text = format_co2_kg(carbon_metrics.co2_kg) if carbon_metrics else ""

            enriched.append(
                {
                    "job": job,
                    "status_name": status_name,
                    "status_icon": status_icon,
                    "task_id": job.task_id or "",
                    "tags": tags,
                    "progress": progress_text,
                    "progress_fraction": progress_fraction,
                    "duration": duration,
                    "submitted": submitted,
                    "co2": co2_text,
                }
            )

        self._active_group_levels = self._active_levels()
        self._build_group(tree.root, enriched, level_index=0)

        if first_build:
            # First build: expand all top-level nodes
            for child in tree.root.children:
                child.expand()
        else:
            # Restore previous expanded state
            self._restore_expanded(tree.root, expanded_paths)

    def _build_group(
        self,
        parent: TreeNode,
        jobs: list[dict],
        level_index: int,
    ) -> None:
        """Recursively build tree groups, skipping non-discriminating levels"""
        if not jobs:
            return

        # If we've exhausted all grouping levels, add job leaves
        if level_index >= len(self._active_group_levels):
            self._add_job_leaves(parent, jobs)
            return

        level = self._active_group_levels[level_index]

        if level == GroupLevel.TAG:
            self._build_tag_group(parent, jobs, level_index)
        else:
            groups = self._group_by_level(jobs, level)

            # Skip this level if it doesn't discriminate
            if len(groups) <= 1:
                self._build_group(parent, jobs, level_index + 1)
                return

            for group_key, group_jobs in sorted(groups.items()):
                label = self._format_group_label(level, group_key, len(group_jobs))
                node = parent.add(
                    label, data=TreeGroupData(level, group_key, len(group_jobs))
                )
                self._build_group(node, group_jobs, level_index + 1)

    def _build_tag_group(
        self,
        parent: TreeNode,
        jobs: list[dict],
        level_index: int,
    ) -> None:
        """Handle TAG grouping: find the most discriminating tag key and split on it.

        If no tag discriminates, skip to next level."""
        # Collect all tag keys across these jobs
        tag_keys: Counter[str] = Counter()
        for j in jobs:
            for k in j["tags"]:
                tag_keys[k] += 1

        if not tag_keys:
            # No tags at all, skip to next level
            self._build_group(parent, jobs, level_index + 1)
            return

        # Find the tag key that produces the most distinct groups
        # (best discrimination), only counting jobs that have the key
        best_key = None
        best_score = 0
        for key, count in tag_keys.most_common():
            values_with_key = {str(j["tags"][key]) for j in jobs if key in j["tags"]}
            if len(values_with_key) > 1 and len(values_with_key) > best_score:
                best_key = key
                best_score = len(values_with_key)

        if best_key is None:
            # No tag discriminates, skip
            self._build_group(parent, jobs, level_index + 1)
            return

        # Split jobs: those with the tag key vs those without
        jobs_with_tag: list[dict] = []
        jobs_without_tag: list[dict] = []
        for j in jobs:
            if best_key in j["tags"]:
                jobs_with_tag.append(j)
            else:
                jobs_without_tag.append(j)

        # Group jobs that have this tag key
        groups: dict[str, list[dict]] = {}
        for j in jobs_with_tag:
            val = str(j["tags"][best_key])
            groups.setdefault(val, []).append(j)

        for group_val, group_jobs in sorted(groups.items()):
            label = Text()
            label.append(f"{best_key}", style="bold")
            label.append(f"={group_val} ({len(group_jobs)})")
            node = parent.add(
                label,
                data=TreeGroupData(
                    GroupLevel.TAG, f"{best_key}={group_val}", len(group_jobs)
                ),
            )
            # Continue with TAG level again (for remaining tag keys)
            # but also allow other levels after
            self._build_tag_subgroup(node, group_jobs, best_key, level_index)

        # Jobs without this tag: continue to next grouping levels
        if jobs_without_tag:
            self._build_tag_subgroup(parent, jobs_without_tag, best_key, level_index)

    def _build_tag_subgroup(
        self,
        parent: TreeNode,
        jobs: list[dict],
        used_key: str,
        original_level_index: int,
    ) -> None:
        """After splitting on a tag key, try remaining tag keys, then continue
        with the rest of the grouping order."""
        # Collect remaining tag keys (exclude already used)
        remaining_keys: Counter[str] = Counter()
        for j in jobs:
            for k in j["tags"]:
                if k != used_key:
                    remaining_keys[k] += 1

        # Find next discriminating tag key (only among jobs that have it)
        best_key = None
        best_score = 0
        for key, count in remaining_keys.most_common():
            values_with_key = {str(j["tags"][key]) for j in jobs if key in j["tags"]}
            if len(values_with_key) > 1 and len(values_with_key) > best_score:
                best_key = key
                best_score = len(values_with_key)

        if best_key is not None:
            # Split jobs: those with the tag key vs those without
            jobs_with_tag = [j for j in jobs if best_key in j["tags"]]
            jobs_without_tag = [j for j in jobs if best_key not in j["tags"]]

            groups: dict[str, list[dict]] = {}
            for j in jobs_with_tag:
                val = str(j["tags"][best_key])
                groups.setdefault(val, []).append(j)

            for group_val, group_jobs in sorted(groups.items()):
                label = Text()
                label.append(f"{best_key}", style="bold")
                label.append(f"={group_val} ({len(group_jobs)})")
                node = parent.add(
                    label,
                    data=TreeGroupData(
                        GroupLevel.TAG, f"{best_key}={group_val}", len(group_jobs)
                    ),
                )
                self._build_tag_subgroup(
                    node, group_jobs, best_key, original_level_index
                )

            # Jobs without this tag: continue without it
            if jobs_without_tag:
                self._build_tag_subgroup(
                    parent, jobs_without_tag, best_key, original_level_index
                )
        else:
            # No more discriminating tags, continue with remaining group levels
            self._build_group(parent, jobs, original_level_index + 1)

    def _group_by_level(
        self, jobs: list[dict], level: GroupLevel
    ) -> dict[str, list[dict]]:
        """Group jobs by a specific level"""
        groups: dict[str, list[dict]] = {}
        for j in jobs:
            if level == GroupLevel.STATUS:
                key = j["status_name"]
            elif level == GroupLevel.TASK_ID:
                key = j["task_id"]
            else:
                key = ""
            groups.setdefault(key, []).append(j)
        return groups

    # Build status name -> icon map from JobState singletons
    _STATUS_ICONS: dict[str, str] = {
        state.name: state.icon
        for state in [
            JobState.DONE,
            JobState.ERROR,
            JobState.RUNNING,
            JobState.SCHEDULED,
            JobState.WAITING,
            JobState.READY,
            JobState.UNSCHEDULED,
            JobState.TRANSIENT,
        ]
    }

    def _format_group_label(self, level: GroupLevel, key: str, count: int) -> Text:
        """Format a group node label"""
        label = Text()
        if level == GroupLevel.STATUS:
            icon = self._STATUS_ICONS.get(key, "❓")
            label.append(f"{icon} {key}", style="bold")
            label.append(f" ({count})")
        elif level == GroupLevel.TASK_ID:
            label.append(f"{key}", style="bold cyan")
            label.append(f" ({count})")
        else:
            label.append(f"{key} ({count})")
        return label

    def _add_job_leaves(self, parent: TreeNode, jobs: list[dict]) -> None:
        """Add job leaf nodes with table-like information"""
        # Collect which group levels are already represented in ancestor nodes
        ancestor_levels = self._ancestor_group_levels(parent)
        # Collect which tag keys are already grouped on
        ancestor_tag_keys: set[str] = set()
        for level, value in ancestor_levels:
            if level == GroupLevel.TAG and "=" in value:
                ancestor_tag_keys.add(value.split("=", 1)[0])

        grouped_by_task = any(lv == GroupLevel.TASK_ID for lv, _ in ancestor_levels)
        grouped_by_status = any(lv == GroupLevel.STATUS for lv, _ in ancestor_levels)

        for j in jobs:
            job = j["job"]
            label = Text()

            # Status icon (only if not already grouped by status)
            if not grouped_by_status:
                label.append(f"{j['status_icon']} ")

            # Progress bar (for running jobs)
            if j["progress_fraction"] >= 0:
                label.append_text(_format_progress_bar(j["progress_fraction"]))
                label.append(" ")

            # Short job ID
            label.append(f"{job.identifier[:7]} ", style="dim")

            # Task ID (only if not already grouped by task)
            if not grouped_by_task:
                label.append(f"{j['task_id']} ", style="cyan")

            # Tags not already used for grouping
            remaining_tags = {
                k: v for k, v in j["tags"].items() if k not in ancestor_tag_keys
            }
            if remaining_tags:
                tag_parts = [f"{k}={v}" for k, v in remaining_tags.items()]
                label.append(", ".join(tag_parts), style="dim")

            # Duration
            if j["duration"]:
                label.append(f" [{j['duration']}]", style="yellow")

            # Submitted
            if j["submitted"]:
                label.append(f" {j['submitted']}", style="dim italic")

            # CO2
            if j["co2"]:
                label.append(f" CO2:{j['co2']}", style="green")

            data = TreeJobData(
                job_id=job.identifier,
                task_id=j["task_id"],
                status_text=j["status_name"],
                tags=j["tags"],
            )
            parent.add_leaf(label, data=data)

    def _ancestor_group_levels(self, node: TreeNode) -> list[tuple[GroupLevel, str]]:
        """Collect (level, value) pairs from ancestor group nodes"""
        result: list[tuple[GroupLevel, str]] = []
        current = node
        while current.parent is not None:
            if isinstance(current.data, TreeGroupData):
                result.append((current.data.level, current.data.value))
            current = current.parent
        return result

    def _get_selected_job_data(self) -> TreeJobData | None:
        """Get TreeJobData from the currently highlighted tree node"""
        tree = self.query_one("#jobs-tree", Tree)
        node = tree.cursor_node
        if node and isinstance(node.data, TreeJobData):
            return node.data
        return None

    def action_select_node(self) -> None:
        """Handle enter on a tree node"""
        tree = self.query_one("#jobs-tree", Tree)
        node = tree.cursor_node
        if node is None:
            return
        if isinstance(node.data, TreeJobData):
            # Job leaf - select it
            self.post_message(
                JobSelected(
                    node.data.job_id, node.data.task_id, self.current_experiment or ""
                )
            )
        elif isinstance(node.data, TreeGroupData):
            # Group node - toggle expand/collapse
            node.toggle()

    def action_view_logs(self) -> None:
        data = self._get_selected_job_data()
        if data and self.current_experiment:
            self.post_message(
                ViewJobLogsRequest(data.job_id, data.task_id, self.current_experiment)
            )

    def action_delete_job(self) -> None:
        data = self._get_selected_job_data()
        if data and self.current_experiment:
            self.post_message(
                DeleteJobRequest(data.job_id, data.task_id, self.current_experiment)
            )

    def action_kill_job(self) -> None:
        data = self._get_selected_job_data()
        if data and self.current_experiment:
            self.post_message(
                KillJobRequest(data.job_id, data.task_id, self.current_experiment)
            )

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update status bar when tree cursor moves"""
        node = event.node
        if isinstance(node.data, TreeJobData):
            self.post_message(
                JobHighlighted(node.data.job_id, f"{node.data.status_text}")
            )


class JobsTable(Vertical):
    """Widget displaying jobs for selected experiment"""

    BINDINGS = [
        Binding("ctrl+d", "delete_job", "Delete", show=False),
        Binding("ctrl+k", "kill_job", "Kill", show=False),
        Binding("l", "view_logs", "Logs", key_display="l"),
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("/", "toggle_search", "Search"),
        Binding("c", "clear_filter", "Clear", show=False),
        Binding("r", "refresh_live", "Refresh"),
        Binding("S", "sort_by_status", "Sort ⚑", show=False),
        Binding("T", "sort_by_task", "Sort Task", show=False),
        Binding("D", "sort_by_submitted", "Sort Date", show=False),
        Binding("t", "toggle_tree_view", "Tree"),
        Binding("escape", "clear_search", show=False, priority=True),
    ]

    # Track current sort state
    _sort_column: Optional[str] = None
    _sort_reverse: bool = False
    _needs_rebuild: bool = True  # Start with rebuild needed
    _tree_mode: bool = False

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
        self.task_id_map: dict[str, str] = {}  # job_id -> task_id
        self.experiment_job_info: dict = {}  # job_id -> ExperimentJobInformation
        self._current_experiment_obj = None  # BaseExperiment object for resolved state

    def compose(self) -> ComposeResult:
        yield Static("", id="past-run-banner", classes="hidden")
        yield Static("Loading jobs...", id="jobs-loading", classes="hidden")
        yield SearchBar()
        yield DataTable(id="jobs-table", cursor_type="row")
        tree_view = JobsTreeView()
        tree_view.display = False
        yield tree_view

    def action_toggle_tree_view(self) -> None:
        """Toggle between table and tree view"""
        self._tree_mode = not self._tree_mode
        table = self.query_one("#jobs-table", DataTable)
        tree_view = self.query_one(JobsTreeView)
        table.display = not self._tree_mode
        tree_view.display = self._tree_mode
        if self._tree_mode:
            self._update_tree_view()
            tree_view.query_one("#jobs-tree", Tree).focus()
            self.notify("Tree view", severity="information")
        else:
            table.focus()
            self.notify("Table view", severity="information")

    def _update_tree_view(self) -> None:
        """Update the tree view with current data"""
        if not self._tree_mode:
            return
        tree_view = self.query_one(JobsTreeView)
        jobs = (
            self.state_provider.get_jobs(
                self.current_experiment, run_id=self.current_run_id
            )
            if self.current_experiment
            else []
        )
        if self.filter_fn:
            jobs = [j for j in jobs if self.filter_fn(j)]
        tree_view.set_data(
            jobs=jobs,
            tags_map=self.tags_map,
            experiment_obj=self._current_experiment_obj,
            experiment_job_info=self.experiment_job_info,
            task_id_map=self.task_id_map,
            current_experiment=self.current_experiment,
        )

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
            task_id = self.task_id_map.get(job_id, "")
            self.post_message(
                DeleteJobRequest(job_id, task_id, self.current_experiment)
            )

    def action_kill_job(self) -> None:
        """Request to kill the selected job"""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            task_id = self.task_id_map.get(job_id, "")
            self.post_message(KillJobRequest(job_id, task_id, self.current_experiment))

    def action_view_logs(self) -> None:
        """Request to view logs for the selected job"""
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            task_id = self.task_id_map.get(job_id, "")
            self.post_message(
                ViewJobLogsRequest(job_id, task_id, self.current_experiment)
            )

    def action_copy_path(self) -> None:
        """Copy the job folder path to clipboard"""
        from experimaestro.tui.clipboard import copy

        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self.current_experiment:
            return

        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            job_id = str(row_key.value)
            task_id = self.task_id_map.get(job_id, "")
            if task_id:
                job = self.state_provider.get_job(task_id, job_id)
                if job and job.path:
                    display_path = self.state_provider.get_display_path(job)
                    if copy(display_path):
                        self.notify(
                            f"Path copied: {display_path}", severity="information"
                        )
                    else:
                        self.notify("Failed to copy path", severity="error")
                else:
                    self.notify("No path available for this job", severity="warning")
            else:
                self.notify("Cannot find task ID for this job", severity="warning")

    # Status sort order (for sorting by status)
    STATUS_ORDER = {
        "running": 0,
        "scheduled": 1,
        "done": 2,
        "error": 3,
        "waiting": 4,
        "unscheduled": 5,
        "transient": 6,
        "phantom": 7,
    }

    # Failure reason sort order (within error status)
    # More actionable failures first
    FAILURE_ORDER = {
        "TIMEOUT": 0,  # Might just need retry
        "MEMORY": 1,  # Might need resource adjustment
        "REJECTED_TIMELIMIT": 2,  # Time limit exceeded partition max
        "REJECTED_OTHER": 3,  # Other rejection reason
        "DEPENDENCY": 4,  # Need to fix upstream job first
        "FAILED": 5,  # Generic failure
    }

    def _get_status_sort_key(self, job):
        """Get sort key for a job based on resolved status and failure reason.

        Returns tuple (status_order, failure_order) for proper sorting.
        """
        resolved_state, _ = StateProvider.get_resolved_state(
            job, self._current_experiment_obj
        )
        state_name = resolved_state.name if resolved_state else "unknown"
        status_order = self.STATUS_ORDER.get(state_name, 99)

        # For error jobs, also sort by failure reason
        if state_name == "error":
            failure_reason = resolved_state.failure_reason if resolved_state else None
            if failure_reason:
                failure_order = self.FAILURE_ORDER.get(failure_reason.name, 99)
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
        "co2": "CO2",
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
        table.add_column("CO2", key="co2", width=8)
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
            self.experiment_job_info = {}
            self.app.call_from_thread(self._on_data_loaded, [])
            return

        # Fetch data (this is the slow part for remote)
        tags_map = self.state_provider.get_tags_map(experiment_id, run_id)
        dependencies_map = self.state_provider.get_dependencies_map(
            experiment_id, run_id
        )
        experiment_job_info = self.state_provider.get_experiment_job_info(
            experiment_id, run_id
        )
        jobs = self.state_provider.get_jobs(experiment_id, run_id=run_id)

        # Get the experiment object for resolved state
        experiment_obj = self.state_provider.get_experiment(experiment_id)

        # Update on main thread
        self.app.call_from_thread(
            self._on_data_loaded,
            jobs,
            tags_map,
            dependencies_map,
            experiment_job_info,
            experiment_obj,
        )

    def _on_data_loaded(
        self,
        jobs: list,
        tags_map: dict = None,
        dependencies_map: dict = None,
        experiment_job_info: dict = None,
        experiment_obj=None,
    ) -> None:
        """Handle loaded data on main thread"""
        # Hide loading indicator
        self.query_one("#jobs-loading", Static).add_class("hidden")

        # Update maps
        if tags_map is not None:
            self.tags_map = tags_map
        if dependencies_map is not None:
            self.dependencies_map = dependencies_map
        if experiment_job_info is not None:
            self.experiment_job_info = experiment_job_info
        self._current_experiment_obj = experiment_obj

        # Refresh display with loaded jobs
        if self._tree_mode:
            # In tree mode, update the tree with filtered jobs
            filtered = jobs
            if self.filter_fn:
                filtered = [j for j in jobs if self.filter_fn(j)]
            tree_view = self.query_one(JobsTreeView)
            tree_view.set_data(
                jobs=filtered,
                tags_map=self.tags_map,
                experiment_obj=self._current_experiment_obj,
                experiment_job_info=self.experiment_job_info,
                task_id_map=self.task_id_map,
                current_experiment=self.current_experiment,
            )
        else:
            self._refresh_jobs_with_data(jobs)

    def refresh_jobs(self) -> None:
        """Refresh the jobs list from state provider (always background)"""
        if not self.current_experiment:
            return

        self._load_experiment_data(self.current_experiment, self.current_run_id)

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
            # Use experiment_job_info timestamp for submittime
            def _get_submittime(j):
                info = self.experiment_job_info.get(j.identifier)
                if info and info.timestamp:
                    return datetime.fromtimestamp(info.timestamp)
                return datetime.max

            jobs.sort(
                key=_get_submittime,
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
            current_statuses = {}
            for job in jobs:
                rs, _ = StateProvider.get_resolved_state(
                    job, self._current_experiment_obj
                )
                current_statuses[job.identifier] = rs.name if rs else "unknown"
            if (
                hasattr(self, "_last_statuses")
                and self._last_statuses != current_statuses
            ):
                status_changed = True
            self._last_statuses = current_statuses

        needs_rebuild = self._needs_rebuild or jobs_changed or status_changed
        self._needs_rebuild = False

        # Build row data for all jobs and update task_id_map
        rows_data = {}
        for job in jobs:
            job_id = job.identifier
            task_id = job.task_id
            # Track task_id for each job_id
            self.task_id_map[job_id] = task_id or ""

            # Resolve display state from experiment and execution states
            resolved_state, conflict_state = StateProvider.get_resolved_state(
                job, self._current_experiment_obj
            )

            # Format status with icon (and progress % if running)
            if resolved_state and resolved_state == JobState.RUNNING:
                progress_list = job.progress or []
                if progress_list:
                    # We only report main progress here (level 0)
                    last_progress = progress_list[0]
                    progress_pct = last_progress.progress * 100
                    status_text = f"🏃 {progress_pct:.0f}%"
                else:
                    status_text = "🏃"
            else:
                status_text = resolved_state.icon if resolved_state else "❓"

            # Show both icons when states conflict (scheduler/exec)
            if conflict_state is not None:
                status_text = f"{conflict_state.icon}/{resolved_state.icon if resolved_state else '❓'}"

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
            job_info = self.experiment_job_info.get(job.identifier)
            if job_info and job_info.timestamp:
                submitted = datetime.fromtimestamp(job_info.timestamp).strftime(
                    "%Y-%m-%d %H:%M"
                )

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

            # Carbon metrics
            carbon_metrics = getattr(job, "carbon_metrics", None)
            if carbon_metrics is not None:
                co2_text = format_co2_kg(carbon_metrics.co2_kg)
            else:
                co2_text = "-"

            job_id_short = job_id[:7]
            rows_data[job_id] = (
                job_id_short,
                task_id,
                status_text,
                tags_text,
                submitted,
                duration,
                co2_text,
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
                    co2_text,
                ) = row_data
                table.update_cell(job_id, "job_id", job_id_short, update_width=True)
                table.update_cell(job_id, "task", task_id, update_width=True)
                table.update_cell(job_id, "status", status_text, update_width=True)
                table.update_cell(job_id, "tags", tags_text, update_width=True)
                table.update_cell(job_id, "submitted", submitted, update_width=True)
                table.update_cell(job_id, "duration", duration, update_width=True)
                table.update_cell(job_id, "co2", co2_text, update_width=True)

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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle job row highlight (cursor moved) - update status bar"""
        if event.row_key and self.current_experiment:
            job_id = str(event.row_key.value)
            task_id = self.task_id_map.get(job_id, "")
            if task_id:
                job = self.state_provider.get_job(task_id, job_id)
                if job:
                    resolved_state, conflict_state = StateProvider.get_resolved_state(
                        job, self._current_experiment_obj
                    )
                    status_name = resolved_state.name if resolved_state else "unknown"
                    icon = resolved_state.icon if resolved_state else "❓"
                    status_text = f"{icon} {status_name}"
                    failure_reason = (
                        resolved_state.failure_reason if resolved_state else None
                    )
                    if failure_reason:
                        status_text += f" ({failure_reason.name})"
                    if conflict_state is not None:
                        exec_name = resolved_state.name if resolved_state else "?"
                        status_text = (
                            f"{conflict_state.icon} {conflict_state.name}"
                            f" / {icon} {exec_name}"
                        )
                    self.post_message(JobHighlighted(job_id, status_text))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle job selection"""
        if event.row_key and self.current_experiment:
            job_id = str(event.row_key.value)
            task_id = self.task_id_map.get(job_id, "")
            self.post_message(JobSelected(job_id, task_id, self.current_experiment))
