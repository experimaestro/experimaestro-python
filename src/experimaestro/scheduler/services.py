import abc
from enum import Enum
import logging
import threading
from pathlib import Path
from typing import Callable, Optional, Set, TYPE_CHECKING

from experimaestro.scheduler.interfaces import BaseService

if TYPE_CHECKING:
    from experimaestro.scheduler.experiment import Experiment

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


class Service(BaseService):
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

    def set_experiment(self, xp: "Experiment") -> None:
        """Called when the service is added to an experiment.

        Override this method to access the experiment context (e.g., workdir).
        The default implementation does nothing.

        Args:
            xp: The experiment this service is being added to.
        """
        pass

    def state_dict(self) -> dict:
        """Return parameters needed to recreate this service.

        Subclasses should override this to return constructor arguments.
        Path values are automatically serialized and restored (with
        translation for remote monitoring).

        Example::

            def state_dict(self):
                return {
                    "log_dir": self.log_dir,  # Path is auto-handled
                    "name": self.name,
                }

        Returns:
            Dict with constructor kwargs (no need to include __class__).
        """
        return {}

    def _full_state_dict(self) -> dict:
        """Get complete state_dict including __class__ for serialization."""
        d = self.state_dict()
        d["__class__"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return d

    @staticmethod
    def serialize_state_dict(data: dict) -> dict:
        """Serialize a state_dict, converting Path objects to serializable format.

        This is called automatically when storing services. Path values are
        converted to {"__path__": "/path/string"} format.

        Args:
            data: Raw state_dict from service (should include __class__)

        Returns:
            Serializable dictionary with paths converted
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, Path):
                result[k] = {"__path__": str(v)}
            else:
                result[k] = v
        return result

    @staticmethod
    def from_state_dict(
        data: dict, path_translator: Optional[Callable[[str], Path]] = None
    ) -> "Service":
        """Recreate a service from a state dictionary.

        Args:
            data: Dictionary from :meth:`state_dict` (may be serialized)
            path_translator: Optional function to translate remote paths to local.
                Used by remote clients to map paths to local cache.

        Returns:
            A new Service instance, or raises if the class cannot be loaded.

        Raises:
            ValueError: If __unserializable__ is True or __class__ is missing
        """
        import importlib

        # Check if service is marked as unserializable
        if data.get("__unserializable__"):
            raise ValueError(
                f"Service cannot be recreated: {data.get('__reason__', 'unknown reason')}"
            )

        class_path = data.get("__class__")
        if not class_path:
            raise ValueError("Missing '__class__' in service state_dict")

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # Build kwargs, detecting and translating paths automatically
        kwargs = {}
        for k, v in data.items():
            if k.startswith("__"):
                continue  # Skip special keys
            if isinstance(v, dict) and "__path__" in v:
                # Serialized path - deserialize with optional translation
                path_str = v["__path__"]
                if path_translator:
                    kwargs[k] = path_translator(path_str)
                else:
                    kwargs[k] = Path(path_str)
            else:
                kwargs[k] = v

        logger.debug("Creating %s with kwargs: %s", cls.__name__, kwargs)
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
        self._start_lock = threading.Lock()
        self._running_event: Optional[threading.Event] = None

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
        block until the URL is available. If the service is already
        starting or running, returns the existing URL.

        :return: The URL where this service can be accessed
        :raises RuntimeError: If called while service is stopping
        """
        with self._start_lock:
            if self.state == ServiceState.STOPPING:
                raise RuntimeError("Cannot start service while it is stopping")

            if self.state == ServiceState.RUNNING:
                logger.debug("Service already running, returning existing URL")
                return self.url

            if self.state == ServiceState.STOPPED:
                logger.info(
                    "Starting service %s (id=%s)", self.__class__.__name__, id(self)
                )
                self._stop_event.clear()
                self.state = ServiceState.STARTING
                self._running_event = threading.Event()
                self.serve()
            else:
                logger.info(
                    "Service %s (id=%s) already starting, waiting for it",
                    self.__class__.__name__,
                    id(self),
                )

            # State is STARTING - wait for it to be ready
            running_event = self._running_event

        # Wait outside the lock to avoid blocking other callers
        if running_event:
            running_event.wait()
            # Set state to RUNNING (this will notify listeners)
            with self._start_lock:
                if self.state == ServiceState.STARTING:
                    self.state = ServiceState.RUNNING

        return self.url

    def stop(self, timeout: float = 2.0):
        """Stop the web service.

        This method signals the service to stop and waits for the thread
        to terminate. If the thread doesn't stop gracefully within the
        timeout, it attempts to forcefully terminate it.

        :param timeout: Seconds to wait for graceful shutdown before forcing
        """
        with self._start_lock:
            if self.state == ServiceState.STOPPED:
                return

            if self.state == ServiceState.STARTING:
                # Wait for service to finish starting before stopping
                running_event = self._running_event
            else:
                running_event = None

            self.state = ServiceState.STOPPING

        # Wait for starting to complete if needed (outside lock to avoid deadlock)
        if running_event is not None:
            running_event.wait()

        # Signal the service to stop
        self._stop_event.set()

        # Wait for the thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=timeout)

            # If thread is still alive, try to terminate it forcefully
            if self.thread.is_alive():
                self._force_stop_thread()

        with self._start_lock:
            self.url = None
            self._running_event = None
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
            target=self._serve_wrapper,
            name=f"service[{self.id}]",
        )
        self.thread.daemon = True
        self.thread.start()

    def _serve_wrapper(self):
        """Wrapper for _serve that handles state transitions."""
        running_event = self._running_event
        try:
            self._serve(running_event)
        finally:
            # Ensure the event is set even if _serve fails
            if running_event and not running_event.is_set():
                running_event.set()

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
