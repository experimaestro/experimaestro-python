"""WebUI Server implementation

Aligned with TUI's ExperimaestroUI pattern, using StateProvider abstraction.
"""

import logging
import platform
import socket
import threading
import uuid
from typing import ClassVar, Optional

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.settings import ServerSettings

logger = logging.getLogger("xpm.webui")


class WebUIServer:
    """WebUI server for monitoring experiments

    Aligned with TUI's ExperimaestroUI pattern:
    - Uses StateProvider abstraction for data access
    - Supports both embedded mode (live scheduler) and offline mode (database)
    - Can wait for explicit quit from interface (like TUI)
    """

    _instance: ClassVar[Optional["WebUIServer"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    def instance(
        settings: ServerSettings = None,
        state_provider: StateProvider = None,
        wait_for_quit: bool = False,
    ) -> "WebUIServer":
        """Get or create the global server instance

        Args:
            settings: Server settings (optional, uses defaults if not provided)
            state_provider: StateProvider instance (required)
            wait_for_quit: If True, server waits for explicit quit from UI
        """
        if WebUIServer._instance is None:
            with WebUIServer._lock:
                if WebUIServer._instance is None:
                    if settings is None:
                        from experimaestro.settings import get_settings

                        settings = get_settings().server

                    if state_provider is None:
                        raise ValueError(
                            "state_provider parameter is required. "
                            "Pass the Scheduler instance or WorkspaceStateProvider."
                        )

                    WebUIServer._instance = WebUIServer(
                        settings, state_provider, wait_for_quit
                    )
        return WebUIServer._instance

    @staticmethod
    def clear_instance():
        """Clear the singleton instance (for testing)"""
        with WebUIServer._lock:
            WebUIServer._instance = None

    def __init__(
        self,
        settings: ServerSettings,
        state_provider: StateProvider,
        wait_for_quit: bool = False,
    ):
        """Initialize the WebUI server

        Args:
            settings: Server settings
            state_provider: StateProvider for accessing experiment/job data
            wait_for_quit: If True, wait for explicit quit from web interface
        """
        # Determine host binding
        if settings.autohost == "fqdn":
            settings.host = socket.getfqdn()
            logger.info("Auto host name (fqdn): %s", settings.host)
        elif settings.autohost == "name":
            settings.host = platform.node()
            logger.info("Auto host name (name): %s", settings.host)

        if settings.host is None or settings.host == "127.0.0.1":
            self.binding_host = "127.0.0.1"
        else:
            self.binding_host = "0.0.0.0"

        self.host = settings.host or "127.0.0.1"
        self.port = settings.port
        self.token = settings.token or uuid.uuid4().hex
        self.state_provider = state_provider
        self.wait_for_quit = wait_for_quit

        # Check if we have an active experiment (scheduler as state provider)
        from experimaestro.scheduler.base import Scheduler

        self._has_active_experiment = isinstance(state_provider, Scheduler)

        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cv_running = threading.Condition()
        self._quit_event = threading.Event()

        # Uvicorn server reference
        self._uvicorn_server = None

    def start(self):
        """Start the web server in a daemon thread"""
        logger.info("Starting the web server")

        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="webui-server",
        )
        self._thread.start()

        # Wait until server is ready
        with self._cv_running:
            self._cv_running.wait_for(lambda: self._running)

        logger.info(
            "Web server started on http://%s:%d/auth?xpm-token=%s",
            self.host,
            self.port,
            self.token,
        )

    def _run_server(self):
        """Run the uvicorn server (called in thread)"""
        import uvicorn

        from experimaestro.webui.app import create_app

        # Find available port if needed
        if self.port is None or self.port == 0:
            logger.info("Searching for an available port")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            self.port = sock.getsockname()[1]
            sock.close()

        # Create FastAPI app
        app = create_app(self)

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=self.binding_host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config)

        # Signal that we're running
        with self._cv_running:
            self._running = True
            self._cv_running.notify_all()

        # Run server (blocks until shutdown)
        self._uvicorn_server.run()
        logger.info("Web server stopped")

    def stop(self):
        """Stop the server gracefully"""
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True

    def request_quit(self):
        """Signal quit request from web interface"""
        logger.info("Quit requested from web interface")
        self._quit_event.set()
        self.stop()

    def wait(self):
        """Wait for explicit quit from web interface

        Call this after experiment completion when wait_for_quit=True.
        Blocks until user clicks Quit button in the web UI.
        """
        if not self.wait_for_quit:
            return

        logger.info("Waiting for quit from web interface...")
        self._quit_event.wait()
        logger.info("Quit signal received")

    @property
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running
