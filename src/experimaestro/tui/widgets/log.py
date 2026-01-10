"""Log capture widget for the TUI"""

from pathlib import Path
from textual import events
from textual.widgets import RichLog
from textual.binding import Binding
from rich.text import Text


class CaptureLog(RichLog):
    """Custom RichLog widget that captures print statements with log highlighting

    Features:
    - Captures print statements with log level highlighting
    - Toggle follow mode (auto-scroll) with Ctrl+F
    - Save log to file with Ctrl+S
    - Copy log to clipboard with Ctrl+Y
    - Tracks unread lines when tab is not visible
    """

    BINDINGS = [
        Binding("ctrl+f", "toggle_follow", "Follow"),
        Binding("ctrl+s", "save_log", "Save"),
        Binding("ctrl+y", "copy_log", "Copy"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._has_unread = False

    @property
    def has_unread(self) -> bool:
        """Whether there are unread log lines"""
        return self._has_unread

    def mark_as_read(self) -> None:
        """Mark all log lines as read"""
        self._has_unread = False
        self._update_tab_title()

    def _update_tab_title(self) -> None:
        """Update the Logs tab title to show unread indicator"""
        try:
            self.app.update_logs_tab_title()
        except Exception:
            pass

    def on_mount(self) -> None:
        """Enable print capturing when widget is mounted"""
        self.begin_capture_print()

    def on_unmount(self) -> None:
        """Stop print capturing when widget is unmounted"""
        self.end_capture_print()

    def _format_log_line(self, text: str) -> Text:
        """Format a log line with appropriate styling based on log level"""
        result = Text()

        # Check for common log level patterns
        if text.startswith("ERROR:") or ":ERROR:" in text:
            result.append(text, style="bold red")
        elif text.startswith("WARNING:") or ":WARNING:" in text:
            result.append(text, style="yellow")
        elif text.startswith("INFO:") or ":INFO:" in text:
            result.append(text, style="green")
        elif text.startswith("DEBUG:") or ":DEBUG:" in text:
            result.append(text, style="dim")
        elif text.startswith("CRITICAL:") or ":CRITICAL:" in text:
            result.append(text, style="bold white on red")
        else:
            result.append(text)

        return result

    def _is_logs_tab_active(self) -> bool:
        """Check if the Logs tab is currently active"""
        try:
            from textual.widgets import TabbedContent

            tabs = self.app.query_one("#main-tabs", TabbedContent)
            return tabs.active == "logs-tab"
        except Exception:
            return False

    def on_print(self, event: events.Print) -> None:
        """Handle print events from captured stdout/stderr"""
        if text := event.text.strip():
            self.write(self._format_log_line(text))
            # Mark as unread only if Logs tab is not active
            if not self._has_unread and not self._is_logs_tab_active():
                self._has_unread = True
                self._update_tab_title()

    def action_toggle_follow(self) -> None:
        """Toggle auto-scroll (follow) mode"""
        self.auto_scroll = not self.auto_scroll
        status = "enabled" if self.auto_scroll else "disabled"
        self.notify(f"Follow mode {status}")

    def action_save_log(self) -> None:
        """Save log content to a file"""
        from datetime import datetime

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"experiment_log_{timestamp}.txt"

        # Get log content from RichLog's lines (Strip objects have .text property)
        try:
            content = self._get_log_content()
        except Exception as e:
            self.notify(f"Failed to get log content: {e}", severity="error")
            return

        if not content:
            self.notify("No log content to save", severity="warning")
            return

        # Try to save to current directory
        try:
            filepath = Path(default_filename)
            filepath.write_text(content)
            path_str = str(filepath.absolute())
            # Copy path to clipboard
            try:
                import pyperclip

                pyperclip.copy(path_str)
                self.notify(f"Log saved and path copied: {path_str}")
            except Exception:
                # Clipboard may not be available (headless systems)
                self.notify(f"Log saved to {path_str}", timeout=60)
        except Exception as e:
            self.notify(f"Failed to save log: {e}", severity="error")

    def action_copy_log(self) -> None:
        """Copy log content to clipboard"""
        content = self._get_log_content()
        if not content:
            self.notify("No log content to copy", severity="warning")
            return

        try:
            import pyperclip

            pyperclip.copy(content)
            line_count = len(self.lines)
            self.notify(f"Copied {line_count} lines to clipboard")
        except Exception as e:
            self.notify(f"Failed to copy: {e}", severity="error")

    def _get_log_content(self) -> str:
        """Get the full log content as plain text from RichLog's lines"""
        # self.lines is a list of Strip objects, each has a .text property
        return "\n".join(line.text for line in self.lines)
