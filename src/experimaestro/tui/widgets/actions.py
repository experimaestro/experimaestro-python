"""Actions list widget for the TUI (alpha feature)"""

import logging
import threading

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Static

from experimaestro.actions import Interaction
from experimaestro.scheduler import action_operations
from experimaestro.scheduler.action_operations import PAST_RUN_WARNING
from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.tui.actions")


# =============================================================================
# Modal screens for TUI interaction
# =============================================================================


class ChoiceScreen(ModalScreen[str]):
    """Modal screen for selecting from a list of choices."""

    def __init__(self, key: str, label: str, choices: list[str]):
        super().__init__()
        self._key = key
        self._label = label
        self._choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-dialog"):
            yield Static(
                f"{self._label} [dim](key: {self._key})[/dim]",
                id="choice-title",
            )
            for i, choice in enumerate(self._choices):
                yield Button(choice, id=f"choice-{i}", variant="primary")
            yield Button("Cancel", id="choice-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "choice-cancel":
            self.dismiss("")
        elif event.button.id and event.button.id.startswith("choice-"):
            idx = int(event.button.id.split("-")[1])
            self.dismiss(self._choices[idx])


class CheckboxScreen(ModalScreen[bool]):
    """Modal screen for a yes/no question."""

    def __init__(self, key: str, label: str, default: bool):
        super().__init__()
        self._key = key
        self._label = label
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical(id="checkbox-dialog"):
            yield Static(
                f"{self._label} [dim](key: {self._key})[/dim]",
                id="checkbox-title",
            )
            with Horizontal(id="checkbox-buttons"):
                yield Button(
                    "Yes",
                    id="checkbox-yes",
                    variant="success" if self._default else "primary",
                )
                yield Button(
                    "No",
                    id="checkbox-no",
                    variant="primary" if self._default else "error",
                )

    def on_mount(self) -> None:
        focus_id = "checkbox-yes" if self._default else "checkbox-no"
        self.query_one(f"#{focus_id}", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "checkbox-yes")


class TextInputScreen(ModalScreen[str]):
    """Modal screen for text input."""

    def __init__(self, key: str, label: str, default: str):
        super().__init__()
        self._key = key
        self._label = label
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical(id="text-dialog"):
            yield Static(
                f"{self._label} [dim](key: {self._key})[/dim]",
                id="text-title",
            )
            yield Input(
                value=self._default,
                placeholder="Enter value...",
                id="text-input",
            )
            with Horizontal(id="text-buttons"):
                yield Button("OK", id="text-ok", variant="primary")
                yield Button("Cancel", id="text-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#text-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "text-ok":
            value = self.query_one("#text-input", Input).value
            self.dismiss(value)
        else:
            self.dismiss("")


class PastRunActionConfirmScreen(ModalScreen[bool]):
    """Confirms the execution of the action of a former run"""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, action_id: str, description: str, run_id: str):
        super().__init__()
        self._action_id = action_id
        self._description = description
        self._run_id = run_id

    def compose(self) -> ComposeResult:
        with Vertical(id="action-confirm-dialog"):
            yield Static("Run an action of a former run?", id="action-confirm-title")
            yield Static(
                f"Action {self._action_id[:12]} ({self._description}) "
                f"of run {self._run_id}",
                id="action-confirm-message",
            )
            yield Static(
                PAST_RUN_WARNING.format(run_id=self._run_id),
                id="action-confirm-warning",
            )
            with Horizontal(id="action-confirm-buttons"):
                yield Button("Run", variant="warning", id="action-confirm-yes")
                yield Button("Cancel", variant="primary", id="action-confirm-no")

    def on_mount(self) -> None:
        """Focus cancel button by default"""
        self.query_one("#action-confirm-no", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "action-confirm-yes")


# =============================================================================
# TUI Interaction implementation
# =============================================================================


class TUIInteraction(Interaction):
    """Interaction implementation using Textual modal screens.

    Must be used from a worker thread — uses ``app.call_from_thread``
    to push modal screens and waits for results via threading events.
    """

    def __init__(self, app):
        self._app = app

    def _push_screen_and_wait(self, screen: ModalScreen) -> object:
        """Push a modal screen from a worker thread and block until dismissed."""
        result_event = threading.Event()
        result_holder = [None]

        def on_dismiss(value):
            result_holder[0] = value
            result_event.set()

        self._app.call_from_thread(self._app.push_screen, screen, on_dismiss)
        result_event.wait()
        return result_holder[0]

    def choice(self, key: str, label: str, choices: list[str]) -> str:
        result = self._push_screen_and_wait(ChoiceScreen(key, label, choices))
        if not result:
            raise InterruptedError("Action cancelled by user")
        return result

    def checkbox(self, key: str, label: str, *, default: bool = False) -> bool:
        return self._push_screen_and_wait(CheckboxScreen(key, label, default))

    def text(self, key: str, label: str, *, default: str = "") -> str:
        result = self._push_screen_and_wait(TextInputScreen(key, label, default))
        if result is None:
            raise InterruptedError("Action cancelled by user")
        return result


# =============================================================================
# Actions list widget
# =============================================================================


class ActionsList(Vertical):
    """Widget displaying actions for selected experiment.

    Actions are those of the run shown (the current run by default). Press
    Enter to execute the selected action interactively (after a confirmation
    for the action of a former run).
    """

    BINDINGS = [
        Binding("enter", "execute_action", "Execute", priority=True),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.current_experiment: str | None = None
        self.current_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Actions (alpha) - Enter to execute",
            id="actions-header",
            classes="section-title",
        )
        yield DataTable(id="actions-table", cursor_type="row")

    def on_mount(self) -> None:
        """Set up the actions table"""
        table = self.query_one("#actions-table", DataTable)
        table.add_columns("ID", "Description", "Class")
        table.cursor_type = "row"

    def set_experiment(
        self, experiment_id: str | None, run_id: str | None = None
    ) -> None:
        """Set the current experiment and refresh actions"""
        self.current_experiment = experiment_id
        self.current_run_id = run_id
        self._refresh_display()

    def refresh_actions(self) -> None:
        """Refresh the actions display"""
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the table from the actions of the run shown"""
        table = self.query_one("#actions-table", DataTable)
        table.clear()

        if not self.current_experiment:
            return

        actions = action_operations.run_actions(
            self.state_provider, self.current_experiment, self.current_run_id
        )
        for action_id, action in actions.items():
            desc = action.description() if hasattr(action, "description") else ""
            action_class = getattr(action, "action_class", "")
            table.add_row(action_id, desc, action_class, key=action_id)

    def _get_selected_action_id(self) -> str | None:
        """Get the action_id of the currently selected row."""
        table = self.query_one("#actions-table", DataTable)
        if table.cursor_row is not None and table.row_count > 0:
            row_key = list(table.rows.keys())[table.cursor_row]
            if row_key:
                return str(row_key.value)
        return None

    def action_execute_action(self) -> None:
        """Execute the selected action (confirmed first for a former run)"""
        action_id = self._get_selected_action_id()
        if not action_id:
            self.notify("No action selected", severity="warning")
            return

        if not self.current_experiment:
            self.notify("No experiment selected", severity="warning")
            return

        experiment_id, run_id = self.current_experiment, self.current_run_id
        if not action_operations.is_past_run(
            self.state_provider, experiment_id, run_id
        ):
            self._execute(experiment_id, None, action_id)
            return

        action = action_operations.run_actions(
            self.state_provider, experiment_id, run_id
        ).get(action_id)
        description = action.description() if action is not None else ""

        def confirmed(ok: bool | None) -> None:
            if ok:
                self._execute(experiment_id, run_id, action_id)

        self.app.push_screen(
            PastRunActionConfirmScreen(action_id, description, run_id), confirmed
        )

    def _execute(self, experiment_id: str, run_id: str | None, action_id: str):
        self.notify(f"Loading action {action_id}...", severity="information")
        self._run_action_worker(experiment_id, run_id, action_id)

    @work(thread=True, exclusive=True, group="action_execute")
    def _run_action_worker(
        self, experiment_id: str, run_id: str | None, action_id: str
    ) -> None:
        """Load and execute action in a background thread."""
        try:
            action = action_operations.load_action(
                self.state_provider, experiment_id, action_id, run_id
            )
            interaction = TUIInteraction(self.app)
            action.execute(interaction)
            self.app.call_from_thread(
                self.notify, "Action completed successfully", severity="information"
            )
        except InterruptedError:
            self.app.call_from_thread(
                self.notify, "Action cancelled", severity="warning"
            )
        except Exception as e:
            logger.exception("Action execution failed")
            self.app.call_from_thread(
                self.notify, f"Action failed: {e}", severity="error"
            )
