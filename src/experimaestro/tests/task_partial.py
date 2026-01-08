"""Task definitions for partial locking tests"""

from pathlib import Path
import time
from experimaestro import Task, Param, Meta, field, PathGenerator, partial, param_group
import logging

logging.basicConfig(level=logging.INFO)

# Define parameter groups
iter_group = param_group("iter")


class PartialTask(Task):
    """Task that uses partial and waits for a file before completing"""

    # Define a partial set
    checkpoints = partial(exclude_groups=[iter_group])

    # Parameter in iter_group - excluded from partial identifier
    x: Param[int] = field(groups=[iter_group])

    # The path to watch before completing
    path: Param[Path]

    # Path generated using the partial identifier
    checkpoint_path: Meta[Path] = field(
        default_factory=PathGenerator("checkpoint", partial=checkpoints)
    )

    def execute(self):
        print(time.time())  # noqa: T201
        # Create checkpoint directory
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Wait for signal file
        while not self.path.is_file():
            time.sleep(0.1)
        print(time.time())  # noqa: T201
