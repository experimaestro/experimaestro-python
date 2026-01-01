import abc
from enum import Enum
import functools
import threading
from typing import Set


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
    """

    id: str
    _state: ServiceState = ServiceState.STOPPED

    def __init__(self):
        self.listeners: Set[ServiceListener] = set()

    def add_listener(self, listener: ServiceListener):
        """Adds a listener

        :param listener: The listener to add
        """
        self.listeners.add(listener)

    def remove_listener(self, listener: ServiceListener):
        """Removes a listener

        :param listener: The listener to remove
        """
        self.listeners.remove(listener)

    def description(self):
        return ""

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: ServiceState):
        # Set the state
        self._state = state

        for listener in self.listeners:
            listener.service_state_changed(self)


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

    Example::

        class MyWebService(WebService):
            id = "myservice"

            def _serve(self, running: threading.Event):
                # Start your web server
                self.url = "http://localhost:8080"
                running.set()
                # Keep serving...
    """

    def __init__(self):
        super().__init__()
        self.url = None

    def get_url(self):
        """Get the URL of this web service, starting it if needed.

        If the service is not running, this method will start it and
        block until the URL is available.

        :return: The URL where this service can be accessed
        """
        if self.state == ServiceState.STOPPED:
            self.state = ServiceState.STARTING
            self.running = threading.Event()
            self.serve()

        # Wait until the server is ready
        self.running.wait()

        # Returns the URL
        return self.url

    def stop(self):
        """Stop the web service."""
        ...

    def serve(self):
        """Start the web service in a background thread.

        This method creates a daemon thread that calls :meth:`_serve`.
        """
        import threading

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

        :param running: Event to signal when ``self.url`` is set
        """
        ...
