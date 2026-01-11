"""Main Textual TUI application for experiment monitoring"""

import logging
from pathlib import Path
from typing import Optional
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Header,
    Footer,
    DataTable,
    TabbedContent,
    TabPane,
)
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.state_status import (
    EventBase,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
    JobStateChangedEvent,
    JobProgressEvent,
    JobSubmittedEvent,
    ServiceAddedEvent,
    ServiceStateChangedEvent,
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
    RunsList,
    GlobalServiceSyncs,
)
from experimaestro.tui.widgets.stray_jobs import OrphanJobsTab


class ExperimaestroUI(App):
    """Textual TUI for monitoring experiments"""

    TITLE = "Experimaestro UI"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "show_help", "Help"),
        Binding("escape", "go_back", "Back", show=False),
        Binding("l", "view_logs", "Logs", show=False),
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
            workdir: Workspace directory (required if state_provider not provided
                and not using deferred mode)
            watch: Enable filesystem watching for workspace mode
            state_provider: Pre-initialized state provider (for active experiments).
                If None and workdir is provided, creates a WorkspaceStateProvider.
                If None and workdir is None, starts in deferred mode (logs only)
                and waits for set_state_provider() to be called.
            show_logs: Whether to show the logs tab (for active experiments)
        """
        super().__init__()
        self.workdir = workdir
        self.watch = watch
        self.show_logs = show_logs
        self._listener_registered = False
        self._monitor_mounted = False

        # Initialize state provider before compose
        if state_provider:
            self.state_provider = state_provider
        elif workdir:
            from experimaestro.scheduler.workspace_state_provider import (
                WorkspaceStateProvider,
            )

            # Get singleton provider instance for this workspace
            self.state_provider = WorkspaceStateProvider.get_instance(self.workdir)
        else:
            # Deferred mode: no provider yet, will be set later via set_state_provider()
            self.state_provider = None

        # Set subtitle to show scheduler status
        self._update_scheduler_status()

    def _update_scheduler_status(self) -> None:
        """Update the subtitle to reflect scheduler status"""
        if self.state_provider is None:
            self.sub_title = "○ Waiting for experiment..."
        elif self.state_provider.is_live:
            self.sub_title = "● Running experiment"
        else:
            self.sub_title = "○ Monitoring workspace"

    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()

        if self.state_provider is None:
            # Deferred mode: only show logs, monitor will be added later
            with TabbedContent(id="main-tabs"):
                with TabPane("Logs", id="logs-tab"):
                    yield CaptureLog(id="logs", auto_scroll=True, wrap=True)
        elif self.show_logs:
            # Tabbed layout with logs and services
            with TabbedContent(id="main-tabs"):
                with TabPane("Monitor", id="monitor-tab"):
                    yield from self._compose_monitor_view()
                with TabPane("Services (0)", id="services-sync-tab"):
                    yield GlobalServiceSyncs(self.state_provider)
                with TabPane("Orphans (0)", id="orphan-tab"):
                    yield OrphanJobsTab(self.state_provider)
                with TabPane("Logs", id="logs-tab"):
                    yield CaptureLog(id="logs", auto_scroll=True, wrap=True)
            self._monitor_mounted = True
        else:
            # Simple layout without logs but with services
            with TabbedContent(id="main-tabs"):
                with TabPane("Monitor", id="monitor-tab"):
                    yield from self._compose_monitor_view()
                with TabPane("Services (0)", id="services-sync-tab"):
                    yield GlobalServiceSyncs(self.state_provider)
                with TabPane("Orphans (0)", id="orphan-tab"):
                    yield OrphanJobsTab(self.state_provider)
            self._monitor_mounted = True

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

        # If monitor is mounted, refresh experiments
        if self._monitor_mounted:
            experiments_list = self.query_one(ExperimentsList)
            experiments_list.refresh_experiments()

        # Register as listener for state change notifications
        # The state provider handles its own notification strategy internally
        if self.state_provider:
            self.state_provider.add_listener(self._on_state_event)
            self._listener_registered = True
            self.log("Registered state listener for notifications")

    def set_state_provider(self, state_provider: StateProvider) -> None:
        """Set the state provider and mount monitor widgets (for deferred mode)

        Call this method from a background thread after starting the experiment.
        The TUI will add the Monitor, Services, and Orphans tabs.

        Args:
            state_provider: The state provider (typically the Scheduler)
        """
        self.state_provider = state_provider
        self._update_scheduler_status()

        # Mount monitor widgets if not already done
        if not self._monitor_mounted:
            self._mount_monitor_widgets()

        # Register listener
        if not self._listener_registered:
            self.state_provider.add_listener(self._on_state_event)
            self._listener_registered = True
            self.log("Registered state listener for notifications")

    def _mount_monitor_widgets(self) -> None:
        """Mount the monitor widgets dynamically (for deferred mode)"""
        tabs = self.query_one("#main-tabs", TabbedContent)

        # Create monitor pane with all its children composed
        monitor_pane = TabPane("Monitor", id="monitor-tab")
        tabs.add_pane(monitor_pane, before="logs-tab")

        # Create widgets
        experiments_list = ExperimentsList(self.state_provider)
        runs_list = RunsList(self.state_provider)
        jobs_table = JobsTable(self.state_provider)
        services_list = ServicesList(self.state_provider)
        job_detail_view = JobDetailView(self.state_provider)

        # Mount experiments and runs lists
        monitor_pane.mount(experiments_list)
        monitor_pane.mount(runs_list)

        # Create experiment tabs with children using compose_add_child
        experiment_tabs = TabbedContent(id="experiment-tabs", classes="hidden")
        jobs_pane = TabPane("Jobs", id="jobs-tab")
        services_pane = TabPane("Services", id="services-tab")
        jobs_pane.compose_add_child(jobs_table)
        services_pane.compose_add_child(services_list)
        experiment_tabs.compose_add_child(jobs_pane)
        experiment_tabs.compose_add_child(services_pane)
        monitor_pane.mount(experiment_tabs)

        # Create job detail container
        job_detail_container = Vertical(id="job-detail-container", classes="hidden")
        job_detail_container.compose_add_child(job_detail_view)
        monitor_pane.mount(job_detail_container)

        # Create and mount services sync tab
        services_sync_pane = TabPane("Services (0)", id="services-sync-tab")
        services_sync_pane.compose_add_child(GlobalServiceSyncs(self.state_provider))
        tabs.add_pane(services_sync_pane, before="logs-tab")

        # Create and mount orphans tab (only if not live)
        if not self.state_provider.is_live:
            orphan_pane = TabPane("Orphans (0)", id="orphan-tab")
            orphan_pane.compose_add_child(OrphanJobsTab(self.state_provider))
            tabs.add_pane(orphan_pane, before="logs-tab")

        self._monitor_mounted = True

        # Refresh experiments list
        experiments_list.refresh_experiments()

    def update_services_tab_title(self) -> None:
        """Update the Services tab title with running service count"""
        try:
            # Count running services from state provider
            from experimaestro.scheduler.services import ServiceState

            all_services = self.state_provider.get_services()
            running_count = sum(
                1
                for s in all_services
                if hasattr(s, "state") and s.state == ServiceState.RUNNING
            )

            # Find and update the tab pane title
            tabs = self.query_one("#main-tabs", TabbedContent)
            tab = tabs.get_tab("services-sync-tab")
            if tab:
                tab.label = f"Services ({running_count})"
        except Exception:
            pass

    def update_orphan_tab_title(self) -> None:
        """Update the Orphans tab title with orphan job count

        Format: Orphans (X/Y) where X=running (stray), Y=non-running (finished)
        """
        try:
            orphan_tab = self.query_one(OrphanJobsTab)
            running = orphan_tab.running_count
            finished = orphan_tab.finished_count
            # Find and update the tab pane title
            tabs = self.query_one("#main-tabs", TabbedContent)
            tab = tabs.get_tab("orphan-tab")
            if tab:
                tab.label = f"Orphans ({running}/{finished})"
        except Exception:
            pass

    def update_logs_tab_title(self) -> None:
        """Update the Logs tab title to show unread indicator (bold when unread)"""
        if not self.show_logs:
            return
        try:
            from rich.text import Text

            log_widget = self.query_one(CaptureLog)
            tabs = self.query_one("#main-tabs", TabbedContent)
            tab = tabs.get_tab("logs-tab")
            if tab:
                if log_widget.has_unread:
                    tab.label = Text("Logs *", style="bold")
                else:
                    tab.label = "Logs"
        except Exception:
            pass

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Handle tab switching"""
        # event.pane is the TabPane, event.tab is the Tab widget (header)
        if event.pane.id == "logs-tab" and self.show_logs:
            try:
                log_widget = self.query_one(CaptureLog)
                log_widget.mark_as_read()
            except Exception:
                pass
        elif event.pane.id == "services-sync-tab":
            # Refresh global services when switching to Services tab
            try:
                global_services = self.query_one(GlobalServiceSyncs)
                global_services.refresh_services()
            except Exception:
                pass

    def _on_state_event(self, event: EventBase) -> None:
        """Handle state change events from the state provider

        This may be called from the state provider's thread or the main thread,
        so we check before using call_from_thread.
        """
        import threading

        self.log.info(f"_on_state_event called with: {type(event).__name__}")

        if threading.current_thread() is threading.main_thread():
            # Already in main thread, call directly
            self._handle_state_event(event)
        else:
            # From background thread, use call_from_thread
            self.call_from_thread(self._handle_state_event, event)

    def _handle_state_event(self, event: EventBase) -> None:
        """Process state event on the main thread using handler dispatch"""
        self.log.info(f"State event: {event}")

        # Dispatch to handler if one exists for this event type
        handler = self.STATE_EVENT_HANDLERS.get(type(event))
        if handler:
            self.log.info(f"Dispatching to handler: {handler.__name__}")
            try:
                handler(self, event)
            except Exception as e:
                self.log.error(f"Error in handler: {e}")
        else:
            self.log.warning(f"No handler for event type: {type(event).__name__}")

    def _handle_experiment_updated(self, event: ExperimentUpdatedEvent) -> None:
        """Handle ExperimentUpdatedEvent - refresh experiments list and jobs"""
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

        # Also refresh jobs table if we're viewing the affected experiment
        # (this handles the case when experiment finishes and events are deleted)
        for jobs_table in self.query(JobsTable):
            if jobs_table.current_experiment == event.experiment_id:
                jobs_table.refresh_jobs()

    def _handle_run_updated(self, event: RunUpdatedEvent) -> None:
        """Handle RunUpdatedEvent - refresh experiments list"""
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

    def _handle_service_added(self, event: ServiceAddedEvent) -> None:
        """Handle ServiceAddedEvent - refresh services list and update tab title"""
        event_exp_id = event.experiment_id
        self.log.info(
            f"ServiceAddedEvent received: exp={event_exp_id}, service={event.service_id}"
        )

        # Refresh the global services widget
        try:
            global_services = self.query_one(GlobalServiceSyncs)
            self.log.info("Calling GlobalServiceSyncs.refresh_services()")
            global_services.refresh_services()
        except Exception as e:
            self.log.warning(f"Failed to refresh global services: {e}")

        # Refresh per-experiment services list
        for services_list in self.query(ServicesList):
            if services_list.current_experiment == event_exp_id:
                services_list.refresh_services()

    def _handle_service_state_changed(self, event: ServiceStateChangedEvent) -> None:
        """Handle ServiceStateChangedEvent - update tab title when service state changes"""
        # Update the Services tab title (running count may have changed)
        self.update_services_tab_title()

        # Also refresh global services widget if visible
        try:
            global_services = self.query_one(GlobalServiceSyncs)
            global_services.refresh_services()
        except Exception:
            pass

        # Refresh per-experiment services list
        for services_list in self.query(ServicesList):
            if services_list.current_experiment == event.experiment_id:
                services_list.refresh_services()

    def _handle_job_submitted(self, event: JobSubmittedEvent) -> None:
        """Handle JobSubmittedEvent - update tags, dependencies, and refresh job list"""
        event_exp_id = event.experiment_id

        # Update tags_map, dependencies_map, and refresh jobs for the affected experiment
        for jobs_table in self.query(JobsTable):
            if jobs_table.current_experiment == event_exp_id:
                # Add the new job's tags to the cache
                if event.tags:
                    jobs_table.tags_map[event.job_id] = {
                        tag.key: tag.value for tag in event.tags
                    }
                # Add the new job's dependencies to the cache
                if event.depends_on:
                    jobs_table.dependencies_map[event.job_id] = event.depends_on
                # Refresh to show the new job
                jobs_table.refresh_jobs()

        # Also update experiment stats
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

    def _handle_job_state_changed(self, event: JobStateChangedEvent) -> None:
        """Handle JobStateChangedEvent - refresh job display

        This event is dispatched once per job state change.
        Used for progress updates and state changes from job processes.
        """
        # Refresh all jobs tables that might contain this job
        for jobs_table in self.query(JobsTable):
            jobs_table.refresh_jobs()

        # Also refresh job detail if we're viewing this job
        for job_detail_container in self.query("#job-detail-container"):
            if not job_detail_container.has_class("hidden"):
                for job_detail_view in self.query(JobDetailView):
                    if job_detail_view.current_job_id == event.job_id:
                        job_detail_view.refresh_job_detail()

        # Also update the experiment stats in the experiments list
        for exp_list in self.query(ExperimentsList):
            exp_list.refresh_experiments()

    def _handle_job_progress(self, event: JobProgressEvent) -> None:
        """Handle JobProgressEvent - refresh job progress display

        This event is dispatched when a job reports progress updates.
        """
        # Refresh all jobs tables that might contain this job
        for jobs_table in self.query(JobsTable):
            jobs_table.refresh_jobs()

        # Also refresh job detail if we're viewing this job
        for job_detail_container in self.query("#job-detail-container"):
            if not job_detail_container.has_class("hidden"):
                for job_detail_view in self.query(JobDetailView):
                    if job_detail_view.current_job_id == event.job_id:
                        job_detail_view.refresh_job_detail()

    STATE_EVENT_HANDLERS = {
        ExperimentUpdatedEvent: _handle_experiment_updated,
        JobStateChangedEvent: _handle_job_state_changed,
        JobProgressEvent: _handle_job_progress,
        RunUpdatedEvent: _handle_run_updated,
        ServiceAddedEvent: _handle_service_added,
        ServiceStateChangedEvent: _handle_service_state_changed,
        JobSubmittedEvent: _handle_job_submitted,
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

    def on_view_job_logs(self, message: ViewJobLogs) -> None:
        """Handle request to view job logs - push LogViewerScreen

        For remote monitoring, switches to log viewer immediately with loading state,
        then starts adaptive sync in background.
        """
        job_path = Path(message.job_path)
        job_id = job_path.name

        # For remote monitoring, switch screen immediately with loading state
        if self.state_provider.is_remote:
            # Push screen immediately - it will handle sync and show loading state
            self.push_screen(
                LogViewerScreen(
                    log_files=[],  # Will be populated after sync
                    job_id=job_id,
                    sync_func=self.state_provider.sync_path,
                    remote_path=str(job_path),
                    task_id=message.task_id,
                    job_state=message.job_state,
                )
            )
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
        self.push_screen(LogViewerScreen(log_files, job_id))

    def on_view_job_logs_request(self, message: ViewJobLogsRequest) -> None:
        """Handle log viewing request from jobs table"""
        job = self.state_provider.get_job(message.job_id, message.experiment_id)
        if not job or not job.path or not job.task_id:
            self.notify("Cannot find job logs", severity="warning")
            return
        self.post_message(ViewJobLogs(str(job.path), job.task_id, job.state))

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
