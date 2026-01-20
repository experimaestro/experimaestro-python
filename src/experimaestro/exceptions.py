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


class TaskCancelled(Exception):
    """Exception raised when a task is cancelled by SIGTERM (e.g., from scancel).

    This exception is raised automatically by the task runner when SIGTERM is
    received. Tasks can catch this exception to perform cleanup before the job
    is killed (typically ~30-60 seconds for SLURM's KillWait).

    Attributes:
        message: Description of the cancellation
        remaining_time: Estimated time remaining before SIGKILL (if known), in seconds
    """

    def __init__(
        self,
        message: str = "Task cancelled by signal",
        remaining_time: float | None = None,
    ):
        self.message = message
        self.remaining_time = remaining_time
        super().__init__(message)
