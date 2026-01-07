"""WebUI module for experimaestro

FastAPI-based web server with native WebSocket support for monitoring experiments.
Aligned with TUI architecture using StateProvider abstraction.
"""

from experimaestro.webui.server import WebUIServer

__all__ = ["WebUIServer"]
