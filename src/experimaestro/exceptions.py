class HandledException(Exception):
    pass


class GracefulTimeout(Exception):
    """Exception raised to signal a graceful timeout in resumable tasks.

    Raise this exception when a task needs to checkpoint and exit before
    a time limit (e.g., SLURM walltime). The task will be marked for retry
    rather than as failed.

    Example::

        ```python
            class LongTraining(ResumableTask):
                def execute(self):
                    for epoch in range(self.epochs):
                        remaining = self.remaining_time()
                        if remaining is not None and remaining < 300:
                            save_checkpoint(self.checkpoint, epoch)
                            raise GracefulTimeout("Not enough time for another epoch")
                        train_one_epoch()
        ```
    """

    def __init__(self, message: str = "Task stopped gracefully before timeout"):
        self.message = message
        super().__init__(message)
