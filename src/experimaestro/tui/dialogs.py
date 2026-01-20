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
  d         Show experiment runs
  Ctrl+d    Delete experiment
  Ctrl+k    Kill all running jobs
  S         Sort by status
  D         Sort by date

[bold cyan]Jobs[/bold cyan]
  l         View job logs
  Ctrl+d    Delete job
  Ctrl+k    Kill running job
  /         Open search filter
  c         Clear search filter
  S         Sort by status
  T         Sort by task
  D         Sort by date
  f         Copy folder path

[bold cyan]Log Viewer[/bold cyan]
  f         Toggle follow mode
  g         Go to top
  G         Go to bottom
  r         Sync now (remote)
  Esc/q     Close viewer

[bold cyan]Services[/bold cyan]
  s         Start service
  x         Stop service
  u         Copy URL

[bold cyan]Orphan Jobs[/bold cyan]
  r         Refresh
  T         Sort by task
  Z         Sort by size
  Ctrl+d    Delete selected
  Ctrl+D    Delete all
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


class WarningDialog(ModalScreen[str | None]):
    """Generic modal dialog for displaying warnings with action buttons

    Returns:
        - action_key: User selected an action
        - None: User dismissed the dialog
    """

    def __init__(
        self,
        warning_key: str,
        title: str,
        description: str,
        actions: dict[str, str],
        severity: str = "warning",
    ) -> None:
        """Initialize the warning dialog.

        Args:
            warning_key: Unique identifier for this warning
            title: Dialog title
            description: Message to display
            actions: Dict mapping action_key to button label
            severity: "info", "warning", or "error"
        """
        super().__init__()
        self.warning_key = warning_key
        self.title_text = title
        self.description = description
        self.actions = actions
        self.severity = severity

    def compose(self) -> ComposeResult:
        # Choose icon based on severity
        icon = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
        }.get(self.severity, "⚠️")

        with Vertical(id="warning-dialog"):
            yield Static(f"{icon}  {self.title_text}", id="warning-title")
            yield Static(self.description, id="warning-message")

            # Create action buttons
            with Horizontal(id="warning-buttons"):
                for action_key, button_label in self.actions.items():
                    # Determine button variant based on action key
                    if action_key in ("clean", "remove", "delete"):
                        variant = "warning"
                    elif action_key in ("dismiss", "cancel", "close"):
                        variant = "default"
                    else:
                        variant = "primary"

                    yield Button(
                        button_label, variant=variant, id=f"warning-action-{action_key}"
                    )

                # Always add dismiss button if not already present
                if "dismiss" not in self.actions:
                    yield Button(
                        "Dismiss", variant="default", id="warning-action-dismiss"
                    )

    def on_mount(self) -> None:
        """Focus dismiss button by default"""
        # Try to focus dismiss button, or first button if no dismiss
        try:
            self.query_one("#warning-action-dismiss", Button).focus()
        except Exception:
            buttons = self.query(Button)
            if buttons:
                buttons[0].focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Extract action key from button ID
        if event.button.id and event.button.id.startswith("warning-action-"):
            action_key = event.button.id[len("warning-action-") :]
            if action_key == "dismiss":
                self.dismiss(None)
            else:
                self.dismiss(action_key)
        else:
            self.dismiss(None)


# Legacy alias for backwards compatibility
StaleTokenAlertScreen = WarningDialog
