from typing import Callable, Dict, Optional
import ipywidgets as widgets
from experimaestro import experiment
from IPython.display import display
from .jobs import jobmonitor  # noqa: F401


class serverwidget:
    def __init__(
        self,
        name,
        *,
        port=12345,
        hook: Callable[["serverwidget"], None] = None,
        environment: Optional[Dict[str, str]] = None,
    ):
        self.name = name

        self.environment = environment or {}
        self.hook = hook
        self.port = port
        self.button = widgets.Button(description="Start the experimaestro server")
        self.output = widgets.Output()
        display(self.button, self.output)
        self.button.on_click(self.on_button_clicked)

        self.on_button_clicked(True)
        self.refresh()

    def refresh(self):
        self.output.clear_output()
        with self.output:
            if experiment.CURRENT:
                self.button.description = "Stop experimaestro server"
                print(f"Server started : http://localhost:{self.port}")
            else:
                self.button.description = "Start experimaestro server"
                print("Server stopped")

    def on_button_clicked(self, b):
        with self.output:
            if experiment.CURRENT:
                try:
                    experiment.CURRENT.__exit__(None, None, None)
                except Exception:
                    print("Error while stopping experimaestro")
                self.current = experiment.CURRENT
            else:
                self.current = experiment(
                    self.name, self.name, host="localhost", port=self.port
                ).__enter__()
                for key, value in self.environment.items():
                    self.current.setenv(key, value)
                if self.hook:
                    self.hook(self)

        self.refresh()
