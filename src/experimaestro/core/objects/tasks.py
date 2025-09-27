import abc
from pathlib import Path
from typing import Any, Optional
from experimaestro import Task, PathGenerator, field, Meta


class CorruptedCheckpointError(Exception):
    """Raised when a checkpoint exists but is corrupted."""


class CheckpointingTask(Task, abc.ABC):
    checkpoint_path: Meta[Path] = field(default_factory=PathGenerator("checkpoints"))
    checkpointing_iterations: int  # save frequency
    keep_last_n: Optional[
        int
    ]  # keep this many most recent checkpoints (None = unlimited)
    keep_every_n: int  # always keep every Nth checkpoint

    def execute(self) -> None:
        """
        Main entry point for the task. Tries to resume from the most recent
        checkpoint if available, otherwise starts from scratch.
        """
        state: Optional[Any] = None
        try:
            state = self.load_checkpoint()
        except CorruptedCheckpointError:
            # Authorized: no state recovered, continue execution from scratch
            state = None

        # Run the main logic
        self._execute(state)

    def save(self, iteration: int, **kwargs: Any) -> None:
        """
        Save a checkpoint if iteration matches the checkpointing frequency.
        Calls `_save` on the appropriate checkpoint directory.
        Then optionally cleans up older checkpoints.
        """
        if iteration % self.checkpointing_iterations != 0:
            return

        # Directory for this checkpoint
        ckpt_dir = self.checkpoint_path / f"checkpoint-{iteration}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Perform the save
        self._save(ckpt_dir, iteration=iteration, **kwargs)

        # Cleanup according to policy
        self._cleanup_checkpoints()

    def load_checkpoint(self) -> Optional[Any]:
        """
        Load the most recent valid checkpoint.
        Returns the state, or None if no checkpoint found.
        Raises CorruptedCheckpointError if latest checkpoint is corrupted,
        but allows trying older checkpoints.
        """
        checkpoints = sorted(
            self.checkpoint_path.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
            reverse=True,
        )

        for ckpt_dir in checkpoints:
            try:
                return self._load(ckpt_dir)
            except CorruptedCheckpointError:
                # Try older checkpoints
                continue
        return None

    def _cleanup_checkpoints(self) -> None:
        """
        Remove older checkpoints according to `keep_last_n` and `keep_every_n`.
        """
        checkpoints = sorted(
            self.checkpoint_path.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )

        # Keep all if no cleanup policy is set
        if self.keep_last_n is None and self.keep_every_n <= 1:
            return

        to_keep = set()

        # Always keep last N
        if self.keep_last_n is not None:
            to_keep.update(checkpoints[-self.keep_last_n :])

        # Always keep every M-th
        if self.keep_every_n > 1:
            for ckpt in checkpoints:
                iteration = int(ckpt.name.split("-")[-1])
                if iteration % self.keep_every_n == 0:
                    to_keep.add(ckpt)

        # Remove the rest
        for ckpt in checkpoints:
            if ckpt not in to_keep:
                # best effort cleanup
                try:
                    if ckpt.is_dir():
                        for child in ckpt.iterdir():
                            if child.is_file():
                                child.unlink()
                            else:
                                # recursively remove directories
                                import shutil

                                shutil.rmtree(child)
                        ckpt.rmdir()
                except Exception:
                    pass

    @abc.abstractmethod
    def _execute(self, state: Optional[Any]) -> None:
        """
        Subclasses implement the actual task logic.
        Receives a recovered state or None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _save(self, ckpt_dir: Path, **kwargs: Any) -> None:
        """
        Subclasses implement how to persist state to `ckpt_dir`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _load(self, ckpt_dir: Path) -> Any:
        """
        Subclasses implement how to load state from `ckpt_dir`.
        Should raise `CorruptedCheckpointError` if checkpoint is invalid.
        """
        raise NotImplementedError
