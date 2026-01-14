# Test for future task outputs handling
# https://github.com/experimaestro/experimaestro-python/issues/90

from functools import partial
import json
import logging
from pathlib import Path
import sys
import time
from experimaestro import (
    Config,
    Param,
    Task,
    ResumableTask,
    DependentMarker,
    LightweightTask,
    field,
    PathGenerator,
)
from experimaestro.core.arguments import Meta
from experimaestro.tests.utils import TemporaryDirectory, TemporaryExperiment


class Model(Config):
    pass


class Checkpoint(Config):
    step: Param[int]
    model: Param[Model]


class CheckpointLoader(LightweightTask):
    checkpoint: Param[Checkpoint]

    def execute(self):
        pass


class Evaluate(Task):
    model: Param[Model]

    def execute(self):
        pass


class Validation(Config):
    model: Param[Model]

    def checkpoint(self, dep: DependentMarker, *, step: int) -> Checkpoint:
        return dep(Checkpoint.C(model=self.model, step=step))

    def compute(self, step: int):
        self.register_task_output(self.checkpoint, step=step)


class Learn(ResumableTask):
    model: Param[Model]
    validation: Param[Validation]

    # Control files for synchronization with tests
    max_step_file: Meta[Path] = field(default_factory=PathGenerator("max_step"))
    state_file: Meta[Path] = field(default_factory=PathGenerator("state.json"))

    def execute(self):
        start_step = 0

        if self.state_file.exists():
            with self.state_file.open("r") as f:
                state = json.load(f)
                start_step = state.get("last_step", 0)
                logging.info("Resuming from step %d", start_step)

        # Wait for max_step_file to know how far to go
        while not self.max_step_file.is_file():
            time.sleep(0.1)

        with self.max_step_file.open("r") as f:
            max_step = int(f.read().strip())
        self.max_step_file.unlink()

        # Use absolute value for step comparison
        # Negative max_step means: produce up to |max_step| then crash (simulate interruption)
        # Positive max_step means: produce up to max_step then complete normally
        abs_max = abs(max_step)

        for step in [15, 30, 45]:
            if step <= start_step:
                logging.info("Skipping already processed step %d", step)
                continue

            if step > abs_max:
                # We're past the limit, stop here
                break

            self.validation.compute(step)

            # Save state after each checkpoint
            with self.state_file.open("w") as f:
                json.dump({"last_step": step}, f)

            # If max_step is negative (e.g. -15), simulate exit after producing |max_step|
            if max_step < 0 and step >= abs_max:
                logging.warning("Simulating interruption after step %d", step)
                sys.exit(1)


def evaluate(evaluations, checkpoint: Checkpoint):
    logging.warning("Evaluating checkpoint %s", checkpoint)
    task = Evaluate.C(model=checkpoint.model)
    checkpoint_loader = CheckpointLoader.C(checkpoint=checkpoint)
    evaluations.append(task.submit(init_tasks=[checkpoint_loader]))


def test_task_dynamic_simple():
    """Test that dynamic task outputs trigger callbacks

    This test verifies that callbacks are guaranteed to complete before
    the experiment context exits. The callback waits for jobs to complete
    before submitting evaluations, which validates that the synchronization
    logic correctly waits for all callbacks to finish.
    """
    import asyncio

    evaluations = []
    xp_ref = [None]  # To access xp from callback

    def collect_checkpoint(checkpoint: Checkpoint):
        """Callback that waits for jobs to complete before evaluating

        This simulates a real-world scenario where the callback needs to wait
        for the triggering task to complete before it can proceed (e.g., to
        read outputs from the task's directory).
        """
        logging.info("Received checkpoint %s, waiting for jobs to complete", checkpoint)
        xp = xp_ref[0]

        # Wait for unfinished jobs to become 0 (all tasks completed)
        async def wait_for_jobs_done():
            async with xp.scheduler.exitCondition:
                while xp.unfinishedJobs > 0:
                    await xp.scheduler.exitCondition.wait()

        asyncio.run_coroutine_threadsafe(
            wait_for_jobs_done(), xp.scheduler.loop
        ).result()

        # Now submit evaluation
        logging.info("Jobs done, submitting evaluation for checkpoint %s", checkpoint)
        evaluate(evaluations, checkpoint)

    with TemporaryDirectory() as workdir:
        with TemporaryExperiment(
            "dynamic", timeout_multiplier=6, workdir=workdir
        ) as xp:
            xp_ref[0] = xp
            model = Model.C()
            validation = Validation.C(model=model)
            learn = Learn.C(model=model, validation=validation)
            learn.watch_output(validation.checkpoint, collect_checkpoint)

            learn.submit()

            # Allow the task to run up to step 30
            learn.max_step_file.parent.mkdir(parents=True, exist_ok=True)
            with learn.max_step_file.open("w") as f:
                f.write("30")

            logging.info("Experiment will wait for completion...")

        assert len(evaluations) == 2, f"Expected 2 evaluations, got {len(evaluations)}"


def test_task_dynamic_replay():
    """Test that dynamic outputs are replayed when a task is restarted

    Scenario:
    1. First run: task produces checkpoint for step 15, then exits (simulated timeout)
    2. Second run: task should replay the step 15 checkpoint and produce new ones
    """
    with TemporaryDirectory() as workdir:
        # First run: produce one checkpoint then exit
        evaluations_run1 = []
        try:
            with TemporaryExperiment(
                "dynamic_replay", timeout_multiplier=3, workdir=workdir
            ):
                model = Model.C()
                validation = Validation.C(model=model)
                learn = Learn.C(model=model, validation=validation)
                learn.watch_output(
                    validation.checkpoint, partial(evaluate, evaluations_run1)
                )

                learn.submit()

                # Allow task to produce step 15 checkpoint, then simulate crash
                # Negative value means: produce up to |value| then exit with error
                learn.max_step_file.parent.mkdir(parents=True, exist_ok=True)
                with learn.max_step_file.open("w") as f:
                    f.write("-15")

        except Exception as e:
            # Expected: the task will fail when trying to go past max_step
            logging.info("First run ended (expected): %s", e)

        # First run should have produced at least one evaluation (for step 15)
        assert len(evaluations_run1) == 1, (
            f"Run 1: Expected 1 evaluation, got {len(evaluations_run1)}"
        )

        # Second run: restart and continue
        evaluations_run2 = []
        with TemporaryExperiment(
            "dynamic_replay", timeout_multiplier=12, workdir=workdir
        ):
            model = Model.C()
            validation = Validation.C(model=model)
            learn = Learn.C(model=model, validation=validation)
            learn.watch_output(
                validation.checkpoint, partial(evaluate, evaluations_run2)
            )

            learn.submit()

            # Allow task to run to completion (step 45)
            learn.max_step_file.parent.mkdir(parents=True, exist_ok=True)
            with learn.max_step_file.open("w") as f:
                f.write("45")

        # Second run should have:
        # - Replayed the step 15 checkpoint (from first run)
        # - Produced step 30 and 45 checkpoints
        # Total: 3 evaluations (but step 15 was replayed, not re-produced)
        assert len(evaluations_run2) == 3, (
            f"Run 2: Expected 3 evaluations, got {len(evaluations_run2)}"
        )
