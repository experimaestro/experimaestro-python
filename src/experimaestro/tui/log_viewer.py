"""Log viewer screen for viewing job logs efficiently

Logs are read from their :class:`~experimaestro.logs.source.LogSource`
(local file, or remote file read by blocks) with the reader shared with the
web UI (``logs/reader.py``): their end first, then their new content while
following, and their earlier content when scrolled to the top. Reading may
wait for the network: it runs in a worker thread, and "Loading…" is shown
when it takes time.
"""

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane

from experimaestro.logs.reader import LogFile
from experimaestro.logs.source import LocalLogSource, LogSource

#: Delay (seconds) before showing that a read is in progress
LOADING_DELAY = 0.3


class LogWidget(Vertical):
    """Widget for displaying a single log file"""

    def __init__(self, source: LogSource, file_path: str, widget_id: str):
        super().__init__(id=widget_id)
        self.file_path = file_path
        self.log_file = LogFile(source)
        self.following = True
        self._last_partial = ""
        #: The completed lines shown (to show earlier ones before them)
        self._lines: list[str] = []
        #: A read is running (one at a time)
        self._busy = False
        self._loaded = False

    def compose(self) -> ComposeResult:
        yield Static(f"📄 {self.file_path}", classes="log-file-path")
        yield Static(
            "Loading earlier lines…", id=f"{self.id}-earlier", classes="log-status"
        )
        yield RichLog(id=f"{self.id}-content", wrap=True, highlight=True, markup=False)
        yield Static("", id=f"{self.id}-partial", classes="log-partial", markup=False)
        yield Static("", id=f"{self.id}-status", classes="log-status", markup=False)

    def on_mount(self) -> None:
        self.query_one(f"#{self.id}-earlier").display = False
        self.query_one(f"#{self.id}-partial").display = False
        self._show_status("Loading…")
        self.watch(self.query_one(RichLog), "scroll_y", self._on_scroll)
        self._start(self._read_tail)

    # --- Reading (in a worker thread, one read at a time)

    def _start(self, read) -> bool:
        if self._busy:
            return False
        self._busy = True
        if self._loaded:
            self.set_timer(LOADING_DELAY, self._show_loading)
        else:
            # Shown at once when opening (nothing is shown yet)
            self._show_loading()
        read()
        return True

    def _show_loading(self) -> None:
        if self._busy:
            self._show_status("Loading…")

    def _show_status(self, text: str) -> None:
        status = self.query_one(f"#{self.id}-status", Static)
        status.update(text)
        status.display = bool(text)

    def _done(self, error: Exception | None = None) -> None:
        self._busy = False
        if error is not None:
            self._show_status(f"Cannot read the log: {error}")
        elif self._lines or self._last_partial:
            self._show_status("")
        elif self.log_file.exists:
            self._show_status("(no output)")
        else:
            self._show_status("(no log file yet)")
        self._loaded = True
        # Scrolled to the top while reading
        self.call_after_refresh(self._load_earlier)

    def _call(self, callback, *args) -> None:
        """Call from the worker thread (unless the widget is gone)"""
        if self.is_attached:
            self.app.call_from_thread(callback, *args)

    @work(thread=True, group="log-read")
    def _read_tail(self) -> None:
        try:
            lines, partial = self.log_file.read_tail()
        except Exception as e:
            self._call(self._done, e)
            return
        self._call(self._apply, True, lines, partial)

    @work(thread=True, group="log-read")
    def _read_new(self) -> None:
        generation = self.log_file.generation
        try:
            lines, partial = self.log_file.read_new_content()
        except Exception as e:
            self._call(self._done, e)
            return
        self._call(self._apply, generation != self.log_file.generation, lines, partial)

    @work(thread=True, group="log-read")
    def _read_earlier(self) -> None:
        try:
            lines = self.log_file.read_before()
        except Exception as e:
            self._call(self._prepend, [], e)
            return
        self._call(self._prepend, lines, None)

    # --- Display

    def _apply(self, reset: bool, complete_lines: list[str], partial: str) -> None:
        log_widget = self.query_one(f"#{self.id}-content", RichLog)
        if reset:
            log_widget.clear()
            self._lines = []
        if complete_lines:
            self._lines.extend(complete_lines)
            for line in complete_lines:
                log_widget.write(line)
        if partial != self._last_partial or reset:
            partial_widget = self.query_one(f"#{self.id}-partial", Static)
            partial_widget.update(partial)
            partial_widget.display = bool(partial)
            self._last_partial = partial
        self._done()

    def _on_scroll(self, scroll_y: float) -> None:
        self._load_earlier()

    def _load_earlier(self) -> None:
        """Load earlier lines when the top is shown (and not read yet)"""
        log_widget = self.query_one(f"#{self.id}-content", RichLog)
        if log_widget.scroll_y <= 0 and self._loaded and self.log_file.start > 0:
            if self._start(self._read_earlier):
                self.query_one(f"#{self.id}-earlier").display = True

    def _prepend(self, lines: list[str], error: Exception | None) -> None:
        """Show earlier lines before the ones shown, keeping what is shown in
        place"""
        self.query_one(f"#{self.id}-earlier").display = False
        if lines:
            log_widget = self.query_one(f"#{self.id}-content", RichLog)
            shown, self._lines = self._lines, lines + self._lines
            scroll_y = log_widget.scroll_y
            log_widget.clear()
            for line in lines:
                log_widget.write(line, scroll_end=False)
            added = len(log_widget.lines)
            for line in shown:
                log_widget.write(line, scroll_end=False)
            log_widget.scroll_to(y=scroll_y + added, animate=False)
        self._done(error)

    def refresh_content(self) -> None:
        """Check for and append new content"""
        if self.following and self._loaded:
            self._start(self._read_new)

    def check_now(self) -> None:
        """Check the log for changes without waiting"""
        self.log_file.source.touch()
        self.refresh_content()

    def scroll_to_end(self) -> None:
        """Scroll to end of log"""
        log_widget = self.query_one(f"#{self.id}-content", RichLog)
        log_widget.scroll_end()

    def scroll_to_top(self) -> None:
        """Scroll to top of log (earlier lines are then loaded)"""
        log_widget = self.query_one(f"#{self.id}-content", RichLog)
        log_widget.scroll_home()


class LogViewerScreen(Screen, inherit_bindings=False):
    """Screen for viewing logs, one tab per log file"""

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

    .log-status {
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

    .log-partial {
        background: $boost;
        padding: 0 1;
        height: auto;
        color: $text;
        text-style: italic;
    }
    """

    BINDINGS = [
        Binding("f", "toggle_follow", "Follow"),
        Binding("g", "go_to_top", "Top"),
        Binding("G", "go_to_bottom", "Bottom"),
        Binding("r", "check_now", "Refresh", show=False),
        Binding("escape", "close_viewer", "Back", priority=True),
        Binding("q", "close_viewer", "Quit", priority=True),
    ]

    def __init__(self, logs: list[tuple[str, LogSource]], job_id: str):
        """Initialize the log viewer

        Args:
            logs: The logs (displayed path, source), shown in this order
            job_id: Job identifier
        """
        super().__init__()
        self.logs = logs
        self.job_id = job_id
        self.following = True
        self.log_widgets: list[LogWidget] = []

    def compose(self) -> ComposeResult:
        yield Header()

        if len(self.logs) == 1:
            # Single file - simple view
            path, source = self.logs[0]
            widget = LogWidget(source, path, "log-0")
            self.log_widgets.append(widget)
            yield widget
        else:
            # Multiple files - tabbed view
            with TabbedContent():
                for i, (path, source) in enumerate(self.logs):
                    with TabPane(Path(path).name, id=f"tab-{i}"):
                        widget = LogWidget(source, path, f"log-{i}")
                        self.log_widgets.append(widget)
                        yield widget

        yield Footer()

    def on_mount(self) -> None:
        """Start watching for changes"""
        self.set_interval(0.5, self._refresh_logs)

    def _refresh_logs(self) -> None:
        """Refresh all log widgets"""
        if not self.following:
            return

        for widget in self.log_widgets:
            widget.refresh_content()

    def action_close_viewer(self) -> None:
        """Go back to the job detail view"""
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

    def action_check_now(self) -> None:
        """Check the logs for changes without waiting"""
        for widget in self.log_widgets:
            widget.check_now()


def create_service_log_viewer(service) -> LogViewerScreen:
    """Create a log viewer for a service

    The logs are read locally: services run on this host (also when
    monitoring a remote workspace).

    Args:
        service: Service object with stdout/stderr properties

    Returns:
        LogViewerScreen configured for service logs
    """
    # stderr first (shown on opening): errors and progress bars go there
    logs = [
        (str(path), LocalLogSource(path))
        for path in (service.stderr, service.stdout)
        if path and path.exists()
    ]
    return LogViewerScreen(logs, job_id=service.id)
