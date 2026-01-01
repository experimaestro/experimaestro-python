# Services

Services can be used to add some useful functionalities when running experiments. For instance,
below is an example of tensorboard service

```python
import logging
import threading
from pathlib import Path

from experimaestro import Config, tagspath
from experimaestro.scheduler.services import WebService, ServiceState


def cleanupdir(path: Path):
    """Remove directory contents"""
    if path.exists():
        for child in path.iterdir():
            child.unlink()


class TensorboardService(WebService):
    id = "tensorboard"

    def __init__(self, path: Path):
        super().__init__()

        self.path = path
        cleanupdir(self.path)
        self.path.mkdir(exist_ok=True, parents=True)
        logging.info("You can monitor learning with:")
        logging.info("tensorboard --logdir=%s", self.path)
        self.url = None

    def add(self, config: Config, path: Path):
        (self.path / tagspath(config)).symlink_to(path)

    def description(self):
        return "Tensorboard service"

    def close(self):
        if self.server:
            self.state = ServiceState.STOPPING
            self.server.shutdown()
            self.state = ServiceState.STOPPED

    def _serve(self, running: threading.Event):
        import tensorboard as tb

        try:
            self.state = ServiceState.STARTING
            self.program = tb.program.TensorBoard()
            self.program.configure(
                host="localhost",
                logdir=str(self.path.absolute()),
                path_prefix=f"/services/{self.id}",
                port=0,
            )
            self.server = self.program._make_server()

            self.url = self.server.get_url()
            running.set()
            self.state = ServiceState.RUNNING
            self.server.serve_forever()
        except Exception:
            logging.exception("Error while starting tensorboard")
            running.set()
```

## Adding a service to an experiment

Services are added to an experiment using the
{meth}`~experimaestro.experiment.add_service` method. The method returns
the same service instance, allowing you to use it immediately.

```python
from experimaestro import experiment

with experiment("/path/to/workdir", "my_experiment", port=12345) as xp:
    # Add the tensorboard service to the experiment
    tb = xp.add_service(TensorboardService(xp.workdir / "runs"))

    # Submit a task
    learner = Learner.C(...).submit()

    # Register the task's log directory with tensorboard
    # This creates a symlink so tensorboard can monitor the run
    tb.add(learner, learner.logpath)

    # Wait for completion
    xp.wait()
```

When using the web interface (enabled by the `port` parameter), services
are accessible through the services menu. For `WebService` subclasses like
`TensorboardService`, the service URL is automatically proxied through the
experimaestro web interface.
