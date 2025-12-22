class HandledException(Exception):
    pass


class GracefulTimeout(Exception):
    """Exception raised by ResumableTask to stop gracefully before a hard timeout.

    ResumableTasks can catch signals or monitor time remaining and raise this
    exception to stop cleanly before the scheduler (e.g., SLURM) terminates
    the process. The task will be retried if retry_count < max_retries.

    Example:
        class LongTraining(ResumableTask):
            def execute(self):
                for epoch in range(1000):
                    if time_remaining() < MIN_TIME_FOR_EPOCH:
                        raise GracefulTimeout("Not enough time for another epoch")
                    train_one_epoch()
                    save_checkpoint()
    """

    def __init__(self, message: str = "Task stopped gracefully before timeout"):
        self.message = message
        super().__init__(message)
