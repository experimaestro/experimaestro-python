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
    """Web service"""

    def __init__(self):
        super().__init__()
        self.url = None

    def get_url(self):
        if self.state == ServiceState.STOPPED:
            self.state = ServiceState.STARTING
            self.running = threading.Event()
            self.serve()

        # Wait until the server is ready
        self.running.wait()

        # Returns the URL
        return self.url

    def stop(self):
        ...

    def serve(self):
        import threading

        self.thread = threading.Thread(
            target=functools.partial(self._serve, self.running),
            name=f"service[{self.id}]",
        )
        self.thread.daemon = True
        self.thread.start()

    @abc.abstractmethod
    def _server(self, running: threading.Event):
        """Starts the web service

        :param running: signals that `self.url` is set
        """
        ...
