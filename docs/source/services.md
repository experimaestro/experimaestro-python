# Services

Services can be used to add some useful functionalities when running experiments. For instance,
below is an example of tensorboard service

```python
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

Within an experiment, this can be used as follows:

```python
# Creates the tensorboard service
tb = xp.add_service(TensorboardService(xp.workdir / "runs"))

learner = Learner()
learner.submit()

# This will allow to monitor the run through tensorboard
# (in the web interface)
tb.add(learner, learner.logpath)
```
