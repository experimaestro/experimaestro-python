"""Experiment actions (alpha feature).

Actions are user-defined operations that can be registered during experiment
submission and executed after completion. They support interactive user input
via the :class:`Interaction` abstraction.

.. warning::
    This feature is in **alpha**. The API may change in future releases.
"""

from abc import ABC, abstractmethod

from experimaestro.core.objects import Config


class Interaction(ABC):
    """Abstract interface for user interaction during action execution.

    Each frontend (CLI, TUI, Web) provides its own implementation.
    """

    @abstractmethod
    def choice(self, key: str, label: str, choices: list[str]) -> str:
        """Ask the user to select from a list of choices.

        :param key: Unique identifier for this question (used for pre-fill)
        :param label: The question to display
        :param choices: List of valid choices
        :return: The selected choice
        """
        ...

    @abstractmethod
    def checkbox(self, key: str, label: str, *, default: bool = False) -> bool:
        """Ask the user a yes/no question.

        :param key: Unique identifier for this question (used for pre-fill)
        :param label: The question to display
        :param default: Default value if not provided
        :return: True or False
        """
        ...

    @abstractmethod
    def text(self, key: str, label: str, *, default: str = "") -> str:
        """Ask the user for text input.

        :param key: Unique identifier for this question (used for pre-fill)
        :param label: The prompt to display
        :param default: Default value if not provided
        :return: The text entered
        """
        ...


class CLIInteraction(Interaction):
    """CLI-based interaction using rich prompts.

    Uses rich for styled terminal output with numbered choice selection.
    Supports pre-filling answers via a dictionary, skipping interactive
    prompts for keys that are already provided.

    :param prefill: Dictionary mapping question keys to pre-filled answers
    """

    def __init__(self, prefill: dict[str, str] | None = None):
        self._prefill = prefill or {}

    def choice(self, key: str, label: str, choices: list[str]) -> str:
        if key in self._prefill:
            value = self._prefill[key]
            if value in choices:
                return value
            raise ValueError(
                f"Pre-filled value '{value}' for '{key}' not in choices: {choices}"
            )

        from rich.console import Console
        from rich.prompt import IntPrompt

        console = Console()
        console.print(f"\n[bold]{label}[/bold] [dim](--set {key}=...)[/dim]")
        for i, c in enumerate(choices, 1):
            console.print(f"  [cyan]{i}[/cyan]) {c}")

        while True:
            idx = IntPrompt.ask(
                "[dim]Select[/dim]",
                choices=[str(i) for i in range(1, len(choices) + 1)],
                show_choices=False,
            )
            return choices[idx - 1]

    def checkbox(self, key: str, label: str, *, default: bool = False) -> bool:
        if key in self._prefill:
            value = self._prefill[key].lower()
            if value in ("true", "yes", "1", "y"):
                return True
            if value in ("false", "no", "0", "n"):
                return False
            raise ValueError(
                f"Pre-filled value '{value}' for '{key}' is not a valid boolean"
            )

        from rich.prompt import Confirm

        return Confirm.ask(
            f"[bold]{label}[/bold] [dim](--set {key}=true|false)[/dim]",
            default=default,
        )

    def text(self, key: str, label: str, *, default: str = "") -> str:
        if key in self._prefill:
            return self._prefill[key]

        from rich.prompt import Prompt

        return Prompt.ask(
            f"[bold]{label}[/bold] [dim](--set {key}=...)[/dim]",
            default=default or None,
        )


class Action(Config):
    """Base class for experiment actions (alpha feature).

    Actions are Config subclasses that describe operations to perform on
    experiment results. They are registered during task submission and can
    be executed after experiment completion.

    Example::

        class ExportToHub(Action):
            model: Param[Model]

            def describe(self) -> str:
                return "Export model to HF Hub"

            def execute(self, interaction: Interaction):
                name = interaction.text("name", "Hub model name:", default="my-model")
                private = interaction.checkbox("private", "Private repo?")
                self.model.push_to_hub(name, private=private)
    """

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this action."""
        ...

    @abstractmethod
    def execute(self, interaction: Interaction) -> None:
        """Execute the action, interacting with the user as needed.

        :param interaction: Interface for asking questions to the user
        """
        ...
