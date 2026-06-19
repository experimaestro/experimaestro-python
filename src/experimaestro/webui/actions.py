"""Web-driven action interaction.

Actions (alpha) gather their inputs through an :class:`Interaction`. The TUI uses
modal dialogs; the CLI uses ``--set key=value`` prefills. For the web we reuse
the prefill idea: the frontend submits the answers it has, and whenever the
action asks for a field it doesn't yet have, we raise :class:`ActionInputRequired`
carrying the field spec so the frontend can prompt for it and re-submit.
"""

from typing import Any, Dict, List, Optional

from experimaestro.actions import Interaction


class ActionInputRequired(Exception):
    """Raised when an action needs an input the frontend hasn't provided yet."""

    def __init__(self, field: Dict[str, Any]):
        super().__init__(f"input required: {field.get('key')}")
        self.field = field


class WebInteraction(Interaction):
    """Interaction backed by a prefilled answer dict.

    Returns prefilled answers (validated/coerced like ``CLIInteraction``) and
    raises :class:`ActionInputRequired` for the first missing field.
    """

    def __init__(self, prefill: Optional[Dict[str, str]] = None):
        self._prefill = prefill or {}

    def choice(self, key: str, label: str, choices: List[str]) -> str:
        if key in self._prefill:
            value = self._prefill[key]
            if value in choices:
                return value
            raise ValueError(f"Value '{value}' for '{key}' not in choices: {choices}")
        raise ActionInputRequired(
            {"key": key, "label": label, "kind": "choice", "choices": choices}
        )

    def checkbox(self, key: str, label: str, *, default: bool = False) -> bool:
        if key in self._prefill:
            value = str(self._prefill[key]).lower()
            if value in ("true", "yes", "1", "y"):
                return True
            if value in ("false", "no", "0", "n"):
                return False
            raise ValueError(f"Value '{value}' for '{key}' is not a valid boolean")
        raise ActionInputRequired(
            {"key": key, "label": label, "kind": "checkbox", "default": default}
        )

    def text(self, key: str, label: str, *, default: str = "") -> str:
        if key in self._prefill:
            return self._prefill[key]
        raise ActionInputRequired(
            {"key": key, "label": label, "kind": "text", "default": default}
        )
