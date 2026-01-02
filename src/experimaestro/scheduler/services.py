import abc
from enum import Enum
import functools
import logging
import threading
from typing import Set

logger = logging.getLogger(__name__)


class ServiceListener:
    """A service listener"""

    def service_state_changed(service):
        pass


class ServiceState(Enum):
    """State of a service lifecycle.

    Services transition through these states:
    STOPPED -> STARTING -> RUNNING -> STOPPING -> STOPPED
    """

    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3


class Service:
    """An experiment service

    Services can be associated with an experiment. They send
    notifications to service listeners.

    To support restarting services from monitor mode, subclasses should
    override :meth:`state_dict` to return the data needed to recreate
    the service, and implement :meth:`from_state_dict` to recreate it.
    """

    id: str
    _state: ServiceState = ServiceState.STOPPED

    def __init__(self):
        self._listeners: Set[ServiceListener] = set()
        self._listeners_lock = threading.Lock()

    def state_dict(self) -> dict:
        """Return a dictionary representation for serialization.

        Subclasses should override this to include any parameters needed
        to recreate the service. The base implementation returns the
        class module and name.

        Returns:
            Dict with '__class__' key and any additional kwargs.
        """
        return {
            "__class__": f"{self.__class__.__module__}.{self.__class__.__name__}",
        }

    @staticmethod
    def from_state_dict(data: dict) -> "Service":
        """Recreate a service from a state dictionary.

        Args:
            data: Dictionary from :meth:`state_dict`

        Returns:
            A new Service instance, or raises if the class cannot be loaded.
        """
        import importlib

        class_path = data.get("__class__")
        if not class_path:
            raise ValueError("Missing '__class__' in service state_dict")

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # Remove __class__ and pass remaining as kwargs
        kwargs = {k: v for k, v in data.items() if k != "__class__"}
        return cls(**kwargs)

    def add_listener(self, listener: ServiceListener):
        """Adds a listener

        :param listener: The listener to add
        """
        with self._listeners_lock:
            self._listeners.add(listener)

    def remove_listener(self, listener: ServiceListener):
        """Removes a listener

        :param listener: The listener to remove
        """
        with self._listeners_lock:
            self._listeners.discard(listener)

    def description(self):
        return ""

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: ServiceState):
        # Set the state
        self._state = state

        # Notify listeners with thread-safe snapshot
        with self._listeners_lock:
            listeners_snapshot = list(self._listeners)

        for listener in listeners_snapshot:
            try:
                listener.service_state_changed(self)
            except Exception:
                logger.exception("Error notifying listener %s", listener)


class WebService(Service):
    """Base class for web-based experiment services.

    Web services provide HTTP endpoints that can be accessed through the
    experimaestro web interface. When an experiment is running with a port
    configured, web services are automatically proxied through the main
    experimaestro server.

    To implement a web service:

    1. Subclass ``WebService``
    2. Set a unique ``id`` class attribute
    3. Implement the :meth:`_serve` method to start your web server
    4. Set ``self.url`` and call ``running.set()`` when ready
    5. Optionally check ``self.should_stop()`` to handle graceful shutdown

    Example::

        class MyWebService(WebService):
            id = "myservice"

            def _serve(self, running: threading.Event):
                # Start your web server
                self.url = "http://localhost:8080"
                running.set()
                # Keep serving, checking for stop signal
                while not self.should_stop():
                    time.sleep(1)
    """

    def __init__(self):
        super().__init__()
        self.url = None
        self.thread = None
        self._stop_event = threading.Event()

    def should_stop(self) -> bool:
        """Check if the service should stop.

        Subclasses can call this in their _serve loop to check for
        graceful shutdown requests.

        :return: True if stop() has been called
        """
        return self._stop_event.is_set()

    def get_url(self):
        """Get the URL of this web service, starting it if needed.

        If the service is not running, this method will start it and
        block until the URL is available.

        :return: The URL where this service can be accessed
        """
        if self.state == ServiceState.STOPPED:
            self._stop_event.clear()
            self.state = ServiceState.STARTING
            self.running = threading.Event()
            self.serve()

        # Wait until the server is ready
        self.running.wait()
        self.state = ServiceState.RUNNING

        # Returns the URL
        return self.url

    def stop(self, timeout: float = 2.0):
        """Stop the web service.

        This method signals the service to stop and waits for the thread
        to terminate. If the thread doesn't stop gracefully within the
        timeout, it attempts to forcefully terminate it.

        :param timeout: Seconds to wait for graceful shutdown before forcing
        """
        if self.state == ServiceState.STOPPED:
            return

        self.state = ServiceState.STOPPING

        # Signal the service to stop
        self._stop_event.set()

        # Wait for the thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=timeout)

            # If thread is still alive, try to terminate it forcefully
            if self.thread.is_alive():
                self._force_stop_thread()

        self.url = None
        self.state = ServiceState.STOPPED

    def _force_stop_thread(self):
        """Attempt to forcefully stop the service thread.

        This uses ctypes to raise an exception in the thread. It's not
        guaranteed to work (e.g., if the thread is blocked in C code),
        but it's the best we can do in Python.
        """
        import ctypes

        if self.thread is None or not self.thread.is_alive():
            return

        thread_id = self.thread.ident
        if thread_id is None:
            return

        # Raise SystemExit in the target thread
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit)
        )

        if res == 0:
            # Thread ID was invalid
            pass
        elif res > 1:
            # Multiple threads affected - reset
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread_id), ctypes.c_long(0)
            )

    def serve(self):
        """Start the web service in a background thread.

        This method creates a daemon thread that calls :meth:`_serve`.
        """
        self.thread = threading.Thread(
            target=functools.partial(self._serve, self.running),
            name=f"service[{self.id}]",
        )
        self.thread.daemon = True
        self.thread.start()

    @abc.abstractmethod
    def _serve(self, running: threading.Event):
        """Start the web server (implement in subclasses).

        This method should:

        1. Start your web server
        2. Set ``self.url`` to the service URL
        3. Call ``running.set()`` to signal readiness
        4. Keep the server running (this runs in a background thread)
        5. Optionally check ``self.should_stop()`` for graceful shutdown

        :param running: Event to signal when ``self.url`` is set
        """
        ...
