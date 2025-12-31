"""Textual-based TUI for monitoring experiments"""

from .app import ExperimaestroUI

# Backward compatibility alias
ExperimentTUI = ExperimaestroUI

__all__ = ["ExperimaestroUI", "ExperimentTUI"]
