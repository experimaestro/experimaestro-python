"""Log viewer screen for viewing job logs efficiently"""

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane


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
    """Screen for viewing job logs efficiently"""

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
        Binding("escape", "close_viewer", "Back", priority=True),
        Binding("q", "close_viewer", "Quit", priority=True),
    ]

    def __init__(self, log_files: list[str], job_id: str):
        super().__init__()
        self.log_files = log_files
        self.job_id = job_id
        self.following = True
        self.log_widgets: list[LogWidget] = []

    def compose(self) -> ComposeResult:
        yield Header()

        if len(self.log_files) == 1:
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
