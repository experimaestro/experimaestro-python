"""Textual-based TUI for monitoring experiments"""

from pathlib import Path
from .app import ExperimaestroUI

# Backward compatibility alias
ExperimentTUI = ExperimaestroUI

APP_TCSS_PATH = Path(__file__).parent / "app.tcss"

__all__ = ["ExperimaestroUI", "ExperimentTUI", "APP_TCSS_PATH"]
