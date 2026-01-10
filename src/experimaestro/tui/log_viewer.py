"""Log viewer screen for viewing job logs efficiently"""

from pathlib import Path
from typing import Callable, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane

from experimaestro.scheduler.interfaces import JobState


# Default chunk size for reading file (64KB)
CHUNK_SIZE = 64 * 1024
# How many bytes to read from end of file initially
INITIAL_TAIL_SIZE = 256 * 1024  # 256KB


class LogFile:
    """Efficient log file reader that tracks position and watches for changes"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.position = 0
        self.size = 0
        self._update_size()

    def _update_size(self) -> None:
        """Update the known file size"""
        try:
            self.size = self.path.stat().st_size
        except OSError:
            self.size = 0

    def read_tail(self, max_bytes: int = INITIAL_TAIL_SIZE) -> str:
        """Read the last N bytes of the file"""
        if not self.path.exists():
            return ""

        self._update_size()
        if self.size == 0:
            return ""

        try:
            with open(self.path, "r", errors="replace") as f:
                # Start from max_bytes before end, or beginning
                start_pos = max(0, self.size - max_bytes)
                f.seek(start_pos)

                # If we're not at the start, skip to the next newline
                if start_pos > 0:
                    f.readline()  # Skip partial line

                content = f.read()
                self.position = f.tell()
                return content
        except Exception:
            return ""

    def read_new_content(self) -> str:
        """Read any new content since last read"""
        if not self.path.exists():
            return ""

        self._update_size()

        # File was truncated or rotated
        if self.size < self.position:
            self.position = 0

        if self.position >= self.size:
            return ""

        try:
            with open(self.path, "r", errors="replace") as f:
                f.seek(self.position)
                content = f.read()
                self.position = f.tell()
                return content
        except Exception:
            return ""

    def has_new_content(self) -> bool:
        """Check if there's new content without reading it"""
        self._update_size()
        return self.size > self.position


class LogWidget(Vertical):
    """Widget for displaying a single log file with efficient loading"""

    def __init__(self, file_path: str, widget_id: str):
        super().__init__(id=widget_id)
        self.file_path = file_path
        self.log_file = LogFile(file_path)
        self.following = True

    def compose(self) -> ComposeResult:
        yield Static(f"📄 {self.file_path}", classes="log-file-path")
        yield RichLog(id=f"{self.id}-content", wrap=True, highlight=True, markup=False)

    def on_mount(self) -> None:
        """Load initial content from tail of file"""
        content = self.log_file.read_tail()
        if content:
            log_widget = self.query_one(f"#{self.id}-content", RichLog)
            for line in content.splitlines():
                log_widget.write(line)

    def refresh_content(self) -> None:
        """Check for and append new content"""
        if not self.following:
            return

        new_content = self.log_file.read_new_content()
        if new_content:
            log_widget = self.query_one(f"#{self.id}-content", RichLog)
            for line in new_content.splitlines():
                log_widget.write(line)

    def scroll_to_end(self) -> None:
        """Scroll to end of log"""
        log_widget = self.query_one(f"#{self.id}-content", RichLog)
        log_widget.scroll_end()

    def scroll_to_top(self) -> None:
        """Scroll to top of log"""
        log_widget = self.query_one(f"#{self.id}-content", RichLog)
        log_widget.scroll_home()


class LogViewerScreen(Screen, inherit_bindings=False):
    """Screen for viewing job logs efficiently

    Supports both local and remote log viewing:
    - Local: reads log files directly
    - Remote: uses adaptive sync to periodically rsync logs from remote
    """

    CSS = """
    LogViewerScreen {
        background: $surface;
    }

    .log-file-path {
        background: $boost;
        padding: 0 1;
        height: auto;
        text-style: bold;
        color: $text-muted;
    }

    .loading-message {
        content-align: center middle;
        height: 100%;
        text-style: italic;
        color: $text-muted;
    }

    .sync-status {
        background: $boost;
        padding: 0 1;
        height: auto;
        text-style: italic;
        color: $text-muted;
    }

    LogWidget {
        height: 1fr;
    }

    LogWidget RichLog {
        height: 1fr;
        border: solid $primary;
    }
    """

    BINDINGS = [
        Binding("f", "toggle_follow", "Follow"),
        Binding("g", "go_to_top", "Top"),
        Binding("G", "go_to_bottom", "Bottom"),
        Binding("r", "sync_now", "Sync", show=False),
        Binding("escape", "close_viewer", "Back", priority=True),
        Binding("q", "close_viewer", "Quit", priority=True),
    ]

    def __init__(
        self,
        log_files: list[str],
        job_id: str,
        sync_func: Optional[Callable[[str], Optional[Path]]] = None,
        remote_path: Optional[str] = None,
        task_id: Optional[str] = None,
        job_state: Optional[JobState] = None,
    ):
        """Initialize the log viewer

        Args:
            log_files: List of local log file paths
            job_id: Job identifier
            sync_func: Function to sync remote path (for remote monitoring)
            remote_path: Remote job directory path (for remote monitoring)
            task_id: Task ID for log file naming (for remote monitoring)
            job_state: Current job state (for adaptive sync decisions)
        """
        super().__init__()
        self.log_files = log_files
        self.job_id = job_id
        self.following = True
        self.log_widgets: list[LogWidget] = []

        # Remote sync support
        self.sync_func = sync_func
        self.remote_path = remote_path
        self.task_id = task_id
        self.job_state = job_state
        self._synchronizer = None
        self._loading = bool(sync_func and not log_files)

    def compose(self) -> ComposeResult:
        yield Header()

        if self._loading:
            # Show loading message while waiting for first sync
            yield Static(
                "Syncing logs from remote...", id="loading", classes="loading-message"
            )
        elif len(self.log_files) == 1:
            # Single file - simple view
            widget = LogWidget(self.log_files[0], "log-0")
            self.log_widgets.append(widget)
            yield widget
        else:
            # Multiple files - tabbed view
            with TabbedContent():
                for i, log_file in enumerate(self.log_files):
                    file_name = Path(log_file).name
                    with TabPane(file_name, id=f"tab-{i}"):
                        widget = LogWidget(log_file, f"log-{i}")
                        self.log_widgets.append(widget)
                        yield widget

        # Sync status for remote monitoring
        if self.sync_func:
            yield Static("", id="sync-status", classes="sync-status")

        yield Footer()

    def on_mount(self) -> None:
        """Start watching for changes"""
        self.set_interval(0.5, self._refresh_logs)

        # Start adaptive sync for remote monitoring
        if self.sync_func and self.remote_path:
            self._start_adaptive_sync()

    def _start_adaptive_sync(self) -> None:
        """Start adaptive sync for remote log files"""
        import logging

        from experimaestro.scheduler.remote.adaptive_sync import AdaptiveSynchronizer

        logger = logging.getLogger("xpm.tui.log_viewer")

        # Only use adaptive sync for running jobs
        is_running = self.job_state and self.job_state.running()
        logger.info(
            f"Starting sync: job_state={self.job_state}, "
            f"is_running={is_running}, remote_path={self.remote_path}"
        )

        # Update loading message to show we're syncing
        try:
            loading = self.query_one("#loading", Static)
            loading.update("Syncing logs from remote (this may take a moment)...")
        except Exception:
            pass

        if is_running:
            # Build a name for logging
            sync_name = f"job:{self.task_id}" if self.task_id else f"job:{self.job_id}"
            self._synchronizer = AdaptiveSynchronizer(
                sync_func=self.sync_func,
                remote_path=self.remote_path,
                name=sync_name,
                on_sync_start=lambda: self.app.call_from_thread(self._on_sync_start),
                on_sync_complete=lambda p: self.app.call_from_thread(
                    self._on_sync_complete, p
                ),
                on_sync_error=lambda e: self.app.call_from_thread(
                    self._on_sync_error, e
                ),
            )
            self._synchronizer.start()
        else:
            # For completed jobs, just sync once
            self._do_single_sync()

    def _do_single_sync(self) -> None:
        """Do a single sync for completed jobs"""
        import threading

        def sync_thread():
            try:
                local_path = self.sync_func(self.remote_path)
                if local_path:
                    self.app.call_from_thread(self._on_sync_complete, local_path)
                else:
                    self.app.call_from_thread(self._on_sync_error, "Sync failed")
            except Exception as e:
                self.app.call_from_thread(self._on_sync_error, str(e))

        thread = threading.Thread(target=sync_thread, daemon=True)
        thread.start()

    def _on_sync_start(self) -> None:
        """Handle sync start - update status"""
        try:
            status = self.query_one("#sync-status", Static)
            status.update("⟳ Syncing...")
        except Exception:
            pass

    def _on_sync_complete(self, local_path) -> None:
        """Handle sync completion - update log widgets"""
        import logging

        logger = logging.getLogger("xpm.tui.log_viewer")

        # Ensure local_path is a Path object
        if not isinstance(local_path, Path):
            local_path = Path(local_path)

        logger.info(f"Sync complete: {local_path}, loading={self._loading}")

        # Update sync status
        try:
            status = self.query_one("#sync-status", Static)
            if self._synchronizer:
                status.update(f"✓ Next sync: {self._synchronizer.interval:.0f}s")
            else:
                status.update("✓ Synced")
        except Exception as e:
            logger.warning(f"Failed to update sync status: {e}")

        # If loading, find log files and create widgets
        if self._loading:
            self._loading = False
            task_name = self.task_id.split(".")[-1] if self.task_id else "task"
            stdout_path = local_path / f"{task_name}.out"
            stderr_path = local_path / f"{task_name}.err"

            logger.info(f"Looking for log files: {stdout_path}, {stderr_path}")

            log_files = []
            if stdout_path.exists():
                log_files.append(str(stdout_path))
            if stderr_path.exists():
                log_files.append(str(stderr_path))

            if not log_files:
                try:
                    loading = self.query_one("#loading", Static)
                    loading.update(
                        f"No log files found: {task_name}.out/.err in {local_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update loading: {e}")
                return

            logger.info(f"Found log files: {log_files}")
            self.log_files = log_files
            self._rebuild_log_widgets()
        else:
            # Just refresh existing widgets
            for widget in self.log_widgets:
                widget.refresh_content()

    def _on_sync_error(self, error: str) -> None:
        """Handle sync error"""
        try:
            status = self.query_one("#sync-status", Static)
            status.update(f"Sync error: {error}")
        except Exception:
            pass

        if self._loading:
            try:
                loading = self.query_one("#loading", Static)
                loading.update(f"Error: {error}")
            except Exception:
                pass

    def _rebuild_log_widgets(self) -> None:
        """Rebuild log widgets after first sync"""
        # Remove loading message
        try:
            loading = self.query_one("#loading", Static)
            loading.remove()
        except Exception:
            pass

        # Add log widgets
        self.log_widgets = []
        if len(self.log_files) == 1:
            widget = LogWidget(self.log_files[0], "log-0")
            self.log_widgets.append(widget)
            self.mount(widget, before=self.query_one("#sync-status"))
        else:
            tabbed = TabbedContent()
            self.mount(tabbed, before=self.query_one("#sync-status"))
            for i, log_file in enumerate(self.log_files):
                file_name = Path(log_file).name
                pane = TabPane(file_name, id=f"tab-{i}")
                widget = LogWidget(log_file, f"log-{i}")
                self.log_widgets.append(widget)
                pane.compose_add_child(widget)
                tabbed.add_pane(pane)

    def _refresh_logs(self) -> None:
        """Refresh all log widgets"""
        if not self.following:
            return

        for widget in self.log_widgets:
            widget.refresh_content()

    def action_close_viewer(self) -> None:
        """Go back to the job detail view"""
        import logging

        logger = logging.getLogger("xpm.tui.log_viewer")

        # Warn if closing during first sync
        if self._loading:
            self.notify("Sync in progress, please wait...", severity="warning")
            logger.info("User tried to close during first sync, ignoring")
            return

        # Stop adaptive sync if running
        if self._synchronizer:
            logger.info("Closing log viewer, stopping sync")
            self._synchronizer.stop()

        self.dismiss()

    def action_toggle_follow(self) -> None:
        """Toggle following mode"""
        self.following = not self.following
        for widget in self.log_widgets:
            widget.following = self.following
        status = "ON" if self.following else "OFF"
        self.notify(f"Follow mode: {status}")

    def action_go_to_top(self) -> None:
        """Scroll to top"""
        self.following = False
        for widget in self.log_widgets:
            widget.following = False
            widget.scroll_to_top()

    def action_go_to_bottom(self) -> None:
        """Scroll to bottom and resume following"""
        self.following = True
        for widget in self.log_widgets:
            widget.following = True
            widget.scroll_to_end()

    def action_sync_now(self) -> None:
        """Trigger immediate sync (for remote monitoring)"""
        if self._synchronizer:
            self._synchronizer.sync_now()
            self.notify("Syncing...")
        elif self.sync_func and self.remote_path:
            self._do_single_sync()
            self.notify("Syncing...")
