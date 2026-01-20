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
        self.server = None
        self.program = None
        logging.info("You can monitor learning with:")
        logging.info("tensorboard --logdir=%s", self.path)
        self.url = None

    def set_experiment(self, xp):
        """Called when added to an experiment - access xp.workdir etc."""
        self.xp = xp
        if xp.run_mode == RunMode.NORMAL_RUN:
            self.path.mkdir(exist_ok=True, parents=True)
            cleanupdir(self.path)

    def state_dict(self) -> dict:
        """Return constructor arguments for service recreation."""
        return {"path": self.path}  # Path is automatically handled

    def add(self, config: Config, path: Path):
        if self.xp.run_mode == RunMode.NORMAL_RUN:
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

## Service State Serialization

For services to be visible in monitoring, they must implement `state_dict()` to
return constructor arguments needed to recreate the service. The base
`Service.state_dict()` returns an empty dict, so subclasses with constructor
parameters must override it.

### Path Handling

`Path` values in `state_dict()` are automatically handled:

1. When storing: Paths are serialized to a special format
2. When recreating locally: Paths are restored as `Path` objects
3. When recreating remotely (SSH): Paths are translated to local cache and synced via rsync

Just return constructor arguments - paths are handled automatically:

```python
def state_dict(self):
    return {"log_dir": self.log_dir}  # Path is auto-handled
```

## Adding a service to an experiment

Services are added to an experiment using the
{py:meth}`~experimaestro.experiment.add_service` method. The method returns
the same service instance, allowing you to use it immediately.

```python
from experimaestro import experiment

with experiment("/path/to/workdir", "my_experiment", port=12345) as xp:
    # Add the tensorboard service to the experiment
    # set_experiment(xp) is called automatically by add_service
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

## TUI Service Management

When using the TUI monitor (`experimaestro experiments monitor`), services can
be managed using the following keyboard shortcuts:

| Shortcut | Action | Description |
|----------|--------|-------------|
| `ctrl+s` | Start | Start the selected service |
| `ctrl+k` | Stop | Stop the selected service |

These shortcuts work both in:

- **Experiment Services tab**: Shows services for the selected experiment
- **Main Services tab**: Shows all running services across all experiments
