"""Modal dialog screens for the TUI"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static
from textual.screen import ModalScreen
from textual.binding import Binding


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
                    "⚠️  The experiment is still in progress.\nQuitting will prevent new jobs from being launched.",
                    id="quit-warning",
                )
            else:
                yield Static("Are you sure you want to quit?", id="quit-message")

            with Horizontal(id="quit-buttons"):
                yield Button("Quit", variant="error", id="quit-yes")
                yield Button("Cancel", variant="primary", id="quit-no")

    def on_mount(self) -> None:
        """Focus Cancel button by default"""
        self.query_one("#quit-no", Button).focus()

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


class HelpScreen(ModalScreen[None]):
    """Modal screen showing keyboard shortcuts"""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("?", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        help_text = """
[bold]Keyboard Shortcuts[/bold]

[bold cyan]Navigation[/bold cyan]
  q         Quit application
  Esc       Go back / Close dialog
  r         Refresh data
  ?         Show this help
  j         Switch to Jobs tab
  s         Switch to Services tab

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

[bold cyan]Services[/bold cyan]
  s         Start service
  x         Stop service
  u         Copy URL

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
