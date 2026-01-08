"""Transient task mode for experimaestro"""

from enum import IntEnum


class TransientMode(IntEnum):
    """Mode for transient tasks

    Transient tasks are intermediary tasks that may not need to persist
    after an experiment completes.
    """

    NONE = 0
    """Task is scheduled normally and kept after completion (default)"""

    TRANSIENT = 1
    """Task is only run if required by another non-transient task"""

    REMOVE = 2
    """Task is transient and its job folder is removed on experiment completion.
    Implies TRANSIENT behavior."""

    @property
    def is_transient(self) -> bool:
        """Returns True if this mode implies transient behavior"""
        return self != TransientMode.NONE

    @property
    def should_remove(self) -> bool:
        """Returns True if the job folder should be removed after completion"""
        return self == TransientMode.REMOVE
