"""Main Textual TUI application for experiment monitoring"""

import logging
from pathlib import Path
from typing import Optional
from textual.app import App, ComposeResult
from textual import work
from textual.containers import Vertical
from textual.widgets import (
    Header,
    Footer,
    DataTable,
    TabbedContent,
    TabPane,
)
from textual.binding import Binding

from experimaestro.scheduler.state_provider import (
    StateProvider,
    StateEvent,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
    JobUpdatedEvent,
    JobExperimentUpdatedEvent,
    ServiceUpdatedEvent,
)
from experimaestro.tui.log_viewer import LogViewerScreen
from experimaestro.tui.utils import format_duration, get_status_icon  # noqa: F401
from experimaestro.tui.messages import (
    ExperimentSelected,
    ExperimentDeselected,
    JobSelected,
    JobDeselected,
    ViewJobLogs,
    ViewJobLogsRequest,
    LogsSyncComplete,
    LogsSyncFailed,
    DeleteJobRequest,
    DeleteExperimentRequest,
    KillJobRequest,
    KillExperimentRequest,
    FilterChanged,  # noqa: F401
    SearchApplied,  # noqa: F401
    SizeCalculated,  # noqa: F401
    ShowRunsRequest,
    RunSelected,
)
from experimaestro.tui.dialogs import (
    QuitConfirmScreen,
    DeleteConfirmScreen,
    KillConfirmScreen,
    HelpScreen,
)
from experimaestro.tui.widgets import (
    CaptureLog,
    ExperimentsList,
    ServicesList,
    JobsTable,
    JobDetailView,
    OrphanJobsScreen,
    RunsList,
)


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
        Binding("j", "focus_jobs", "Jobs", show=False),
        Binding("s", "focus_services", "Services", show=False),
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
        self._listener_registered = False

        # Initialize state provider before compose
        if state_provider:
            self.state_provider = state_provider
            self.owns_provider = False  # Don't close external provider
        else:
            from experimaestro.scheduler.workspace_state_provider import (
                WorkspaceStateProvider,
            )

            # Get singleton provider instance for this workspace
            self.state_provider = WorkspaceStateProvider.get_instance(
                self.workdir,
                standalone=True,  # Start event file watcher for monitoring
            )
            self.owns_provider = False  # Provider is singleton, don't close

        # Set subtitle to show scheduler status
        self._update_scheduler_status()

    def _update_scheduler_status(self) -> None:
        """Update the subtitle to reflect scheduler status"""
        if self.state_provider.is_live:
            self.sub_title = "● Scheduler Running"
        else:
            self.sub_title = "○ Offline"

    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()

        if self.show_logs:
            # Tabbed layout with logs
            with TabbedContent(id="main-tabs"):
                with TabPane("Monitor", id="monitor-tab"):
                    yield from self._compose_monitor_view()
                with TabPane("Logs", id="logs-tab"):
                    yield CaptureLog(id="logs", auto_scroll=True, wrap=True)
        else:
            # Simple layout without logs
            with Vertical(id="main-container"):
                yield from self._compose_monitor_view()

        yield Footer()

    def _compose_monitor_view(self):
        """Compose the monitor view with experiments, runs, jobs/services tabs, and job details"""
        yield ExperimentsList(self.state_provider)
        # Runs list (hidden initially, shown when 'd' pressed on experiment)
        yield RunsList(self.state_provider)
        # Tabbed view for jobs and services (hidden initially)
        with TabbedContent(id="experiment-tabs", classes="hidden"):
            with TabPane("Jobs", id="jobs-tab"):
                yield JobsTable(self.state_provider)
            with TabPane("Services", id="services-tab"):
                yield ServicesList(self.state_provider)
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

        # Register as listener for state change notifications
        # The state provider handles its own notification strategy internally
        if self.state_provider:
            self.state_provider.add_listener(self._on_state_event)
            self._listener_registered = True
            self.log("Registered state listener for notifications")

    def _on_state_event(self, event: StateEvent) -> None:
        """Handle state change events from the state provider

        This may be called from the state provider's thread or the main thread,
        so we check before using call_from_thread.
        """
        import threading

        if threading.current_thread() is threading.main_thread():
            # Already in main thread, call directly
            self._handle_state_event(event)
        else:
            # From background thread, use call_from_thread
            self.call_from_thread(self._handle_state_event, event)

    def _handle_state_event(self, event: StateEvent) -> None:
        """Process state event on the main thread using handler dispatch"""
        self.log.debug(f"State event {type(event).__name__}")

        # Dispatch to handler if one exists for this event type
        if handler := self.STATE_EVENT_HANDLERS.get(type(event)):
            handler(self, event)

    def _handle_experiment_updated(self, event: ExperimentUpdatedEvent) -> None:
        """Handle ExperimentUpdatedEvent - refresh experiments list and jobs"""
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

        # Also refresh jobs table if we're viewing the affected experiment
        # (this handles the case when experiment finishes and events are deleted)
        for jobs_table in self.query(JobsTable):
            if jobs_table.current_experiment == event.experiment_id:
                jobs_table.refresh_jobs()

    def _handle_job_updated(self, event: JobUpdatedEvent) -> None:
        """Handle JobUpdatedEvent - refresh job display"""
        event_exp_id = event.experiment_id

        # Refresh jobs table if we're viewing the affected experiment
        for jobs_table in self.query(JobsTable):
            if jobs_table.current_experiment == event_exp_id:
                jobs_table.refresh_jobs()

        # Also refresh job detail if we're viewing the affected job
        for job_detail_container in self.query("#job-detail-container"):
            if not job_detail_container.has_class("hidden"):
                for job_detail_view in self.query(JobDetailView):
                    if job_detail_view.current_job_id == event.job_id:
                        job_detail_view.refresh_job_detail()

        # Also update the experiment stats in the experiments list
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

    def _handle_run_updated(self, event: RunUpdatedEvent) -> None:
        """Handle RunUpdatedEvent - refresh experiments list"""
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

    def _handle_service_updated(self, event: ServiceUpdatedEvent) -> None:
        """Handle ServiceUpdatedEvent - refresh services list"""
        event_exp_id = event.experiment_id

        for services_list in self.query(ServicesList):
            if services_list.current_experiment == event_exp_id:
                services_list.refresh_services()

    def _handle_job_experiment_updated(self, event: JobExperimentUpdatedEvent) -> None:
        """Handle JobExperimentUpdatedEvent - update tags, dependencies, and refresh job list"""
        event_exp_id = event.experiment_id

        # Update tags_map, dependencies_map, and refresh jobs for the affected experiment
        for jobs_table in self.query(JobsTable):
            if jobs_table.current_experiment == event_exp_id:
                # Add the new job's tags to the cache
                if event.tags:
                    jobs_table.tags_map[event.job_id] = event.tags
                # Add the new job's dependencies to the cache
                if event.depends_on:
                    jobs_table.dependencies_map[event.job_id] = event.depends_on
                # Refresh to show the new job
                jobs_table.refresh_jobs()

        # Also update experiment stats
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

    STATE_EVENT_HANDLERS = {
        ExperimentUpdatedEvent: _handle_experiment_updated,
        JobUpdatedEvent: _handle_job_updated,
        RunUpdatedEvent: _handle_run_updated,
        ServiceUpdatedEvent: _handle_service_updated,
        JobExperimentUpdatedEvent: _handle_job_experiment_updated,
    }

    def on_experiment_selected(self, message: ExperimentSelected) -> None:
        """Handle experiment selection - show jobs/services tabs"""
        self.log(
            f"Experiment selected: {message.experiment_id} (run: {message.run_id})"
        )

        # Set up services list
        services_list = self.query_one(ServicesList)
        services_list.set_experiment(message.experiment_id)

        # Set up jobs table
        jobs_table_widget = self.query_one(JobsTable)
        jobs_table_widget.set_experiment(message.experiment_id, message.run_id)

        # Show the tabbed content
        tabs = self.query_one("#experiment-tabs", TabbedContent)
        tabs.remove_class("hidden")

        # Focus the jobs table
        jobs_table = self.query_one("#jobs-table", DataTable)
        jobs_table.focus()

    def on_experiment_deselected(self, message: ExperimentDeselected) -> None:
        """Handle experiment deselection - hide jobs/services tabs"""
        # Hide the tabbed content
        tabs = self.query_one("#experiment-tabs", TabbedContent)
        tabs.add_class("hidden")
        # Also hide job detail if visible
        job_detail_container = self.query_one("#job-detail-container")
        job_detail_container.add_class("hidden")

    def on_job_selected(self, message: JobSelected) -> None:
        """Handle job selection - show job detail view"""
        self.log(f"Job selected: {message.job_id} from {message.experiment_id}")

        # Hide tabs, show job detail
        tabs = self.query_one("#experiment-tabs", TabbedContent)
        tabs.add_class("hidden")

        job_detail_container = self.query_one("#job-detail-container")
        job_detail_container.remove_class("hidden")

        # Set the job to display
        job_detail_view = self.query_one(JobDetailView)
        job_detail_view.set_job(message.job_id, message.experiment_id)

    def on_job_deselected(self, message: JobDeselected) -> None:
        """Handle job deselection - go back to jobs view"""
        # Hide job detail, show tabs
        job_detail_container = self.query_one("#job-detail-container")
        job_detail_container.add_class("hidden")

        tabs = self.query_one("#experiment-tabs", TabbedContent)
        tabs.remove_class("hidden")

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
        # Check if job detail is visible -> go back to jobs/services tabs
        job_detail_container = self.query_one("#job-detail-container")
        if not job_detail_container.has_class("hidden"):
            self.post_message(JobDeselected())
            return

        # Check if experiment tabs visible -> go back to experiments list
        experiment_tabs = self.query_one("#experiment-tabs", TabbedContent)
        if not experiment_tabs.has_class("hidden"):
            experiments_list = self.query_one(ExperimentsList)
            if experiments_list.collapsed:
                experiments_list.expand_experiments()
                experiment_tabs.add_class("hidden")
                self.post_message(ExperimentDeselected())

    def action_view_logs(self) -> None:
        """View logs for the current job (if job detail is visible)"""
        job_detail_container = self.query_one("#job-detail-container")
        if not job_detail_container.has_class("hidden"):
            job_detail_view = self.query_one(JobDetailView)
            job_detail_view.action_view_logs()

    def action_show_orphans(self) -> None:
        """Show orphan jobs screen"""
        self.push_screen(OrphanJobsScreen(self.state_provider))

    @work(thread=True, exclusive=True)
    def _sync_and_view_logs(self, job_path: Path, task_id: str) -> None:
        """Sync logs from remote and then view them (runs in worker thread)"""
        try:
            # Sync the job directory
            local_path = self.state_provider.sync_path(str(job_path))
            if not local_path:
                self.post_message(LogsSyncFailed("Failed to sync logs from remote"))
                return

            job_path = local_path

            # Log files are named after the last part of the task ID
            task_name = task_id.split(".")[-1]
            stdout_path = job_path / f"{task_name}.out"
            stderr_path = job_path / f"{task_name}.err"

            # Collect existing log files
            log_files = []
            if stdout_path.exists():
                log_files.append(str(stdout_path))
            if stderr_path.exists():
                log_files.append(str(stderr_path))

            if not log_files:
                self.post_message(
                    LogsSyncFailed(f"No log files found: {task_name}.out/.err")
                )
                return

            # Signal completion via message
            job_id = job_path.name
            self.post_message(LogsSyncComplete(log_files, job_id))

        except Exception as e:
            self.post_message(LogsSyncFailed(str(e)))

    def on_logs_sync_complete(self, message: LogsSyncComplete) -> None:
        """Handle successful log sync - show log viewer"""
        self.push_screen(LogViewerScreen(message.log_files, message.job_id))

    def on_logs_sync_failed(self, message: LogsSyncFailed) -> None:
        """Handle failed log sync"""
        self.notify(message.error, severity="warning")

    def on_view_job_logs(self, message: ViewJobLogs) -> None:
        """Handle request to view job logs - push LogViewerScreen"""
        job_path = Path(message.job_path)

        # For remote monitoring, sync the job directory first (in worker thread)
        if self.state_provider.is_remote:
            self.notify("Syncing logs from remote...", timeout=5)
            self._sync_and_view_logs(job_path, message.task_id)
            return

        # Local monitoring - no sync needed
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

    def on_show_runs_request(self, message: ShowRunsRequest) -> None:
        """Handle request to show experiment runs"""
        runs_list = self.query_one(RunsList)
        runs_list.set_experiment(message.experiment_id, message.current_run_id)

    def on_run_selected(self, message: RunSelected) -> None:
        """Handle run selection - show jobs for the selected run"""
        self.log(
            f"Run selected: {message.run_id} (current={message.is_current}) "
            f"for {message.experiment_id}"
        )

        # Set up jobs table with the selected run
        jobs_table_widget = self.query_one(JobsTable)
        jobs_table_widget.set_experiment(
            message.experiment_id,
            message.run_id,
            is_past_run=not message.is_current,
        )

        # Set up services list
        services_list = self.query_one(ServicesList)
        services_list.set_experiment(message.experiment_id)

        # Show the tabbed content
        tabs = self.query_one("#experiment-tabs", TabbedContent)
        tabs.remove_class("hidden")

        # Collapse experiments list
        experiments_list = self.query_one(ExperimentsList)
        experiments_list.collapse_to_experiment(message.experiment_id)

        # Focus the jobs table
        jobs_table = self.query_one("#jobs-table", DataTable)
        jobs_table.focus()

    def action_focus_jobs(self) -> None:
        """Switch to the jobs tab"""
        tabs = self.query_one("#experiment-tabs", TabbedContent)
        if not tabs.has_class("hidden"):
            tabs.active = "jobs-tab"
            jobs_table = self.query_one("#jobs-table", DataTable)
            jobs_table.focus()
        else:
            self.notify("Select an experiment first", severity="warning")

    def action_focus_services(self) -> None:
        """Switch to the services tab"""
        tabs = self.query_one("#experiment-tabs", TabbedContent)
        if not tabs.has_class("hidden"):
            tabs.active = "services-tab"
            services_table = self.query_one("#services-table", DataTable)
            services_table.focus()
        else:
            self.notify("Select an experiment first", severity="warning")

    def action_switch_focus(self) -> None:
        """Switch focus between experiments table and current tab"""
        focused = self.focused
        if focused:
            experiments_table = self.query_one("#experiments-table", DataTable)
            tabs = self.query_one("#experiment-tabs", TabbedContent)

            if focused == experiments_table and not tabs.has_class("hidden"):
                # Focus the active tab's table
                if tabs.active == "services-tab":
                    self.query_one("#services-table", DataTable).focus()
                else:
                    self.query_one("#jobs-table", DataTable).focus()
            else:
                experiments_table.focus()

    def action_quit(self) -> None:
        """Show quit confirmation dialog"""

        def handle_quit_response(confirmed: bool) -> None:
            if confirmed:
                self.exit()

        self.push_screen(
            QuitConfirmScreen(has_active_experiment=self.state_provider.is_live),
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
