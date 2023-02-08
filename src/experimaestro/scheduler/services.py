import abc
import functools
import threading


class Service:
    id: str

    def description(self):
        return ""


class WebService:
    def __init__(self):
        self.running = None
        self.url = None

    def get_url(self):
        if self.running is None:
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
