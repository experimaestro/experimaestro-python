"""Experiments list widget for the TUI"""

from datetime import datetime
from typing import Optional
from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import DataTable, Input, Label, Static
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
from experimaestro.carbon.utils import format_co2_kg
from experimaestro.ui.experiments import (
    EXPERIMENT_GROUPINGS,
    ExperimentGrouping,
    experiment_matches,
    group_experiments,
)

#: Icons of the experiment statuses (see BaseExperiment.status_category)
STATUS_ICONS = {"failed": "❌", "running": "🏃", "done": "✓", "stopped": "⏹"}


class ExperimentsList(Widget):
    """Widget displaying list of experiments"""

    BINDINGS = [
        Binding("d", "show_runs", "Runs"),
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("ctrl+d", "delete_experiment", "Delete", show=False),
        Binding("ctrl+k", "kill_experiment", "Kill", show=False),
        Binding("S", "sort_by_status", "Sort ⚑", show=False),
        Binding("D", "sort_by_date", "Sort Date", show=False),
        Binding("O", "sort_alphabetical", "Sort Name", show=False),
        Binding("g", "cycle_grouping", "Group"),
        Binding("/", "toggle_search", "Search"),
        Binding("escape", "clear_search", show=False, priority=True),
    ]

    current_experiment: reactive[Optional[str]] = reactive(None)
    collapsed: reactive[bool] = reactive(False)

    # Track current sort state (newest first by default)
    _sort_column: Optional[str] = "started"
    _sort_reverse: bool = True

    # Status sort order (for sorting by status)
    STATUS_ORDER = {
        "running": 0,  # Running experiments first
        "failed": 1,  # Failed experiments (need attention)
        "stopped": 2,  # Stopped before all their jobs finished
        "done": 3,  # Completed experiments
        "empty": 4,  # Empty experiments last
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
        "co2": "CO2",
    }

    # Columns that support sorting (column key -> sort column name)
    SORTABLE_COLUMNS = {
        "id": "id",
        "status": "status",
        "started": "started",
    }

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.experiments = []
        #: Text searched in the experiment IDs and tags (case-insensitive)
        self.search = ""
        #: How the experiments are grouped
        self.grouping: ExperimentGrouping = "none"
        #: Names of the collapsed groups
        self.collapsed_groups: set[str] = set()
        #: Experiment of each row (None for a group header): rows are keyed by
        #: the experiment ID, or by group and experiment ID when grouped (an
        #: experiment can be in several groups); group headers by "\0" + name
        self.row_experiments: dict[str, str | None] = {}
        self._runs_counts: dict[str, str] = {}

    def _matches(self, experiment) -> bool:
        """Whether an experiment is shown with the current search"""
        return experiment_matches(experiment, self.search)

    def action_cycle_grouping(self) -> None:
        """Group the experiments by nothing, prefix, or tag (in turn)"""
        index = EXPERIMENT_GROUPINGS.index(self.grouping)
        self.grouping = EXPERIMENT_GROUPINGS[(index + 1) % len(EXPERIMENT_GROUPINGS)]
        self.collapsed_groups.clear()
        self._update_title()
        self._show_experiments()
        self.notify(
            "Not grouped" if self.grouping == "none" else f"Grouped by {self.grouping}",
            severity="information",
        )

    def _update_title(self) -> None:
        title = self.query_one("#experiments-title", Label)
        if self.grouping == "none":
            title.update("Experiments")
        else:
            title.update(f"Experiments (grouped by {self.grouping})")

    def action_toggle_search(self) -> None:
        """Show the search field and give it the focus"""
        search = self.query_one("#experiments-search", Input)
        search.display = True
        search.focus()

    def action_clear_search(self) -> None:
        """Handle escape: clear and hide the search field, or go back"""
        search = self.query_one("#experiments-search", Input)
        if search.display:
            search.value = ""
            search.display = False
            self.query_one("#experiments-table", DataTable).focus()
        else:
            self.app.action_go_back()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "experiments-search":
            self.search = event.value.strip()
            self.refresh_experiments()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Keep the search and go back to the table"""
        if event.input.id == "experiments-search":
            if not self.search:
                event.input.display = False
            self.query_one("#experiments-table", DataTable).focus()

    def _get_selected_experiment_id(self) -> Optional[str]:
        """Get the experiment ID from the currently selected row"""
        table = self.query_one("#experiments-table", DataTable)
        if table.cursor_row is None:
            return None
        row_key = list(table.rows.keys())[table.cursor_row]
        if row_key:
            return self.row_experiments.get(str(row_key.value))
        return None

    def action_copy_path(self) -> None:
        """Copy the experiment path to clipboard"""
        from experimaestro.tui.clipboard import copy

        exp_id = self._get_selected_experiment_id()
        if not exp_id:
            return

        exp_info = next(
            (exp for exp in self.experiments if exp.experiment_id == exp_id),
            None,
        )
        if exp_info and exp_info.workdir:
            path_str = self.state_provider.translate_path(exp_info.workdir)
            if copy(path_str):
                self.notify(f"Path copied: {path_str}", severity="information")
            else:
                self.notify("Failed to copy path", severity="error")
        else:
            self.notify("No path available for this experiment", severity="warning")

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

    def action_sort_alphabetical(self) -> None:
        """Sort experiments alphabetically by ID"""
        if self._sort_column == "id":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "id"
            self._sort_reverse = False
        self._update_column_headers()
        self.refresh_experiments()
        order = "desc" if self._sort_reverse else "asc"
        self.notify(f"Sorted alphabetically ({order})", severity="information")

    def _sort_experiments(self, experiments: list) -> list:
        """Return experiments sorted according to the selected column"""
        experiments_sorted = list(experiments)
        match self._sort_column:
            case "status":
                experiments_sorted.sort(
                    key=self._get_status_sort_key, reverse=self._sort_reverse
                )
            case "started":
                # Sort by started time (experiments without start time go to end)
                experiments_sorted.sort(
                    key=lambda e: e.started_at or datetime.min,
                    reverse=self._sort_reverse,
                )
            case "id":
                # Sort alphabetically by experiment ID
                experiments_sorted.sort(
                    key=lambda e: (e.experiment_id or "").lower(),
                    reverse=self._sort_reverse,
                )
            # Default: no sorting, use order from state provider
        return experiments_sorted

    @staticmethod
    def _format_experiment_status(exp) -> str:
        """Format the experiment status (see BaseExperiment.status_category)"""
        finished, total = exp.finished_jobs, exp.total_jobs
        match exp.status_category:
            case "failed":
                return f"❌ {exp.failed_jobs} failed"
            case "running":
                return f"🏃 {finished}/{total}"
            case "done":
                return "✓ Done"
            case "stopped":
                return f"⏹ {finished}/{total}"
            case _:
                return "Empty"

    def _get_status_sort_key(self, exp):
        """Get sort key for an experiment based on status

        Primary: status order (running, failed, done, empty)
        Secondary: started_at descending (newest first)
        """
        status_order = self.STATUS_ORDER.get(exp.status_category, 99)
        # Secondary sort by start time (newest first via negative timestamp)
        started = exp.started_at
        started_ts = started.timestamp() if started else 0
        return (status_order, -started_ts)

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
            yield Label("Experiments", classes="section-title", id="experiments-title")
            yield Static("Loading experiments...", id="experiments-loading")
            yield Input(
                placeholder="Search experiments (ID contains…)",
                id="experiments-search",
            )
            yield DataTable(id="experiments-table", cursor_type="row")

    def on_mount(self) -> None:
        """Initialize the experiments table"""
        # Start expanded
        self.add_class("expanded")
        self._initial_load = True

        self.query_one("#experiments-search", Input).display = False
        table = self.query_one("#experiments-table", DataTable)
        table.add_column("ID", key="id")
        table.add_column("Run", key="run")
        table.add_column("#", key="runs", width=3)
        table.add_column("Host", key="host")
        table.add_column("Status", key="status")
        table.add_column("Started", key="started")
        table.add_column("Duration", key="duration")
        table.add_column("CO2", key="co2", width=8)
        self._update_column_headers()
        self.refresh_experiments()

    def refresh_experiments(self, *, refresh: bool = False) -> None:
        """Refresh the experiments list from state provider (background)

        Args:
            refresh: Ask the state provider for a full re-read instead of
                letting it answer from its cache (explicit user refresh)
        """
        # Guard: ensure the table is mounted before querying
        try:
            self.query_one("#experiments-table", DataTable)
        except Exception:
            return
        # Show loading indicator
        try:
            self.query_one("#experiments-loading", Static).remove_class("hidden")
        except Exception:
            pass
        self._load_experiments(refresh)

    @work(thread=True, exclusive=True, group="experiments_load")
    def _load_experiments(self, refresh: bool = False) -> None:
        """Load experiments in background thread"""
        try:
            experiments = self.state_provider.get_experiments(refresh=refresh)
            self.log.debug(
                f"Refreshing experiments: found {len(experiments)} experiments"
            )
        except Exception as e:
            self.log.error(f"ERROR refreshing experiments: {e}")
            import traceback

            self.log.error(traceback.format_exc())
            experiments = []

        # Fetch runs counts in background too (only for offline monitoring)
        runs_counts: dict[str, str] = {}
        if not self.state_provider.is_live:
            for exp in experiments:
                if exp.runs_count is not None:
                    runs_counts[exp.experiment_id] = str(exp.runs_count)
                else:
                    try:
                        runs = self.state_provider.get_experiment_runs(
                            exp.experiment_id, refresh=refresh
                        )
                        runs_counts[exp.experiment_id] = str(len(runs))
                    except Exception as e:
                        self.log.error(
                            f"Error getting runs for {exp.experiment_id}: {e}"
                        )
                        runs_counts[exp.experiment_id] = "-"

        self.app.call_from_thread(self._on_experiments_loaded, experiments, runs_counts)

    def _on_experiments_loaded(
        self, experiments: list, runs_counts: dict[str, str]
    ) -> None:
        """Handle loaded experiments on main thread"""
        # Hide loading indicator
        try:
            self.query_one("#experiments-loading", Static).add_class("hidden")
        except Exception:
            pass

        self.experiments = experiments
        self._runs_counts = runs_counts
        self._show_experiments()

        # Update collapsed header if viewing an experiment
        if self.collapsed and self.current_experiment:
            self._update_collapsed_header(self.current_experiment)

        # Auto-select if there's only one experiment on initial load
        if getattr(self, "_initial_load", False) and len(self.experiments) == 1:
            self._initial_load = False
            exp = self.experiments[0]
            exp_id = exp.experiment_id
            run_id = getattr(exp, "current_run_id", None)
            self.current_experiment = exp_id
            self.collapse_to_experiment(exp_id)
            self.post_message(ExperimentSelected(exp_id, run_id))

    def _experiment_row(self, exp) -> tuple[str, ...]:
        """Cells of the row of an experiment"""
        if exp.started_at:
            started = exp.started_at.strftime("%Y-%m-%d %H:%M")
        else:
            started = "-"

        duration = "-"
        if exp.started_at:
            if exp.ended_at:
                elapsed = (exp.ended_at - exp.started_at).total_seconds()
            else:
                # Still running - show elapsed time
                elapsed = (datetime.now() - exp.started_at).total_seconds()
            duration = format_duration(elapsed)

        # CO2 impact (latest run of each job if known)
        carbon = exp.carbon_summary
        co2_text = format_co2_kg(carbon.co2_kg) if carbon and carbon.co2_kg > 0 else "-"

        tags = "".join(f" [dim]#{tag}[/dim]" for tag in exp.experiment_tags)
        return (
            f"{exp.experiment_id}{tags}",
            getattr(exp, "current_run_id", None) or "-",
            self._runs_counts.get(exp.experiment_id, "-"),
            getattr(exp, "hostname", None) or "-",
            self._format_experiment_status(exp),
            started,
            duration,
            co2_text,
        )

    def _group_row(self, name: str, experiments: list) -> tuple[str, ...]:
        """Cells of the header row of a group"""
        arrow = "▶" if name in self.collapsed_groups else "▼"
        counts: dict[str, int] = {}
        for exp in experiments:
            counts[exp.status_category] = counts.get(exp.status_category, 0) + 1
        summary = " ".join(
            f"{STATUS_ICONS.get(status, '∅')}{count}"
            for status, count in sorted(
                counts.items(), key=lambda item: self.STATUS_ORDER.get(item[0], 99)
            )
        )
        return (
            f"[b]{arrow} {name}[/b] ({len(experiments)})",
            "",
            "",
            "",
            summary,
            "",
            "",
            "",
        )

    def _show_experiments(self) -> None:
        """Fill the table with the experiments matching the search, sorted and
        grouped"""
        try:
            table = self.query_one("#experiments-table", DataTable)
        except Exception:
            return

        # Guard: ensure columns have been added (on_mount may not have run yet)
        if len(table.columns) == 0:
            return

        experiments_sorted = self._sort_experiments(
            [exp for exp in self.experiments if self._matches(exp)]
        )

        # Rows (key, experiment ID or None for a group, cells)
        rows: list[tuple[str, str | None, tuple[str, ...]]] = []
        if self.grouping == "none":
            for exp in experiments_sorted:
                exp_id = exp.experiment_id
                rows.append((exp_id, exp_id, self._experiment_row(exp)))
        else:
            for name, group in group_experiments(experiments_sorted, self.grouping):
                rows.append((f"\0{name}", None, self._group_row(name, group)))
                if name in self.collapsed_groups:
                    continue
                for exp in group:
                    exp_id = exp.experiment_id
                    rows.append(
                        (f"{name}\t{exp_id}", exp_id, self._experiment_row(exp))
                    )

        selected_key, selected_exp = None, None
        if table.cursor_row is not None and table.row_count > 0:
            try:
                row_keys = list(table.rows.keys())
                if table.cursor_row < len(row_keys):
                    selected_key = str(row_keys[table.cursor_row].value)
                    selected_exp = self.row_experiments.get(selected_key)
            except (IndexError, KeyError):
                pass

        self.row_experiments = {key: exp_id for key, exp_id, _ in rows}
        existing_order = [str(k.value) for k in table.rows.keys()]
        if existing_order == [key for key, _, _ in rows]:
            # Same rows: update the cells in place
            columns = list(table.columns)
            for key, _, cells in rows:
                for column, value in zip(columns, cells):
                    table.update_cell(key, column, value, update_width=True)
            return

        # Rebuild, keeping the selected row (or experiment)
        table.clear()
        new_cursor_row = None
        for idx, (key, exp_id, cells) in enumerate(rows):
            table.add_row(*cells, key=key)
            if key == selected_key or (
                new_cursor_row is None and exp_id and exp_id == selected_exp
            ):
                new_cursor_row = idx
        if new_cursor_row is not None and table.row_count > 0:
            table.move_cursor(row=new_cursor_row)

    def update_experiment_stats(self, experiment_id: str) -> None:
        """Update stats for a single experiment without full list refresh.

        Reloads just one experiment from the state provider and updates
        its rows in the DataTable in-place. Falls back to full refresh
        if the experiment is not found in the table.
        """
        try:
            self.query_one("#experiments-table", DataTable)
        except Exception:
            return

        # Load single experiment
        exp = self.state_provider.get_experiment(experiment_id)
        if exp is None:
            return

        if experiment_id not in self.row_experiments.values():
            # Experiment not in table yet — need full refresh (unless hidden by
            # the search or a collapsed group)
            if self._matches(exp):
                self.refresh_experiments()
            return

        # Update the cached experiments list in-place
        for i, cached_exp in enumerate(self.experiments):
            if cached_exp.experiment_id == experiment_id:
                self.experiments[i] = exp
                break

        # Updates its rows (and the group summaries)
        self._show_experiments()

        # Update collapsed header if viewing this experiment
        if self.collapsed and self.current_experiment == experiment_id:
            self._update_collapsed_header(experiment_id)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle experiment selection"""
        if event.row_key:
            key = str(event.row_key.value)
            exp_id = self.row_experiments.get(key)
            if exp_id is None:
                # Group header: collapse or expand the group
                name = key[1:]
                self.collapsed_groups.symmetric_difference_update({name})
                self._show_experiments()
                return
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

        run_id = getattr(exp_info, "current_run_id", None)
        status = self._format_experiment_status(exp_info)

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
