import tempfile
import shutil
import os
import time
from pathlib import Path
import logging
import signal

from experimaestro import experiment, Task
from experimaestro.scheduler.workspace import RunMode


# --- Centralized test timeout configuration ---
# Default base timeout in seconds
DEFAULT_TEST_TIMEOUT = 3

# Environment variable to override base timeout (in seconds)
ENV_TEST_TIMEOUT = "XPM_TEST_TIMEOUT"

# Environment variable for global multiplier (e.g., 5.0 for CI systems)
ENV_TEST_TIMEOUT_MULTIPLIER = "XPM_TEST_TIMEOUT_MULTIPLIER"


def get_test_timeout(multiplier: float = 1.0) -> int:
    """Get the test timeout in seconds.

    The final timeout is: base_timeout * global_multiplier * local_multiplier

    Args:
        multiplier: Local multiplier for tests that need more time (e.g., 2.0)

    Returns:
        Timeout in seconds

    Environment variables:
        XPM_TEST_TIMEOUT: Base timeout in seconds (default: 20)
        XPM_TEST_TIMEOUT_MULTIPLIER: Global multiplier, useful for CI (default: 1.0)
    """
    # Get base timeout
    env_timeout = os.environ.get(ENV_TEST_TIMEOUT)
    if env_timeout is not None:
        try:
            base = int(env_timeout)
        except ValueError:
            logging.warning(
                "Invalid %s value '%s', using default %d",
                ENV_TEST_TIMEOUT,
                env_timeout,
                DEFAULT_TEST_TIMEOUT,
            )
            base = DEFAULT_TEST_TIMEOUT
    else:
        base = DEFAULT_TEST_TIMEOUT

    # Get global multiplier (e.g., for CI environments)
    env_multiplier = os.environ.get(ENV_TEST_TIMEOUT_MULTIPLIER)
    if env_multiplier is not None:
        try:
            global_multiplier = float(env_multiplier)
        except ValueError:
            logging.warning(
                "Invalid %s value '%s', using 1.0",
                ENV_TEST_TIMEOUT_MULTIPLIER,
                env_multiplier,
            )
            global_multiplier = 1.0
    else:
        global_multiplier = 1.0

    return int(base * global_multiplier * multiplier)


class TimeInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, value):
        return self.start > value.end

    def __str__(self):
        return "%.4f - %.4f" % (self.start, self.end)

    def __repr__(self):
        return str(self)


def get_times(task: Task) -> TimeInterval:
    logging.info("Reading times from %s", task.stdout())
    return TimeInterval(
        *(float(t) for t in task.stdout().read_text().strip().split("\n"))
    )


def get_times_frompath(path) -> TimeInterval:
    s = path.read_text().strip().split("\n")
    logging.info("Read times: %s", s)
    return TimeInterval(*(float(t) for t in s))


class TemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self):
        self.path = Path(
            tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        )
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if os.environ.get("XPM_KEEPWORKDIR", False) == "1":
            logging.warning("NOT Removing %s" % self.path)
        else:
            logging.warning(
                "Cleaning up working directory %s (use XPM_KEEPWORKDIR=1 to keep it)",
                self.path,
            )
            shutil.rmtree(self.path, ignore_errors=True)


class timeout:
    """Context manager for test timeouts.

    Args:
        seconds: Timeout in seconds. If None, uses get_test_timeout().
        multiplier: Multiplier for the base timeout (only used if seconds is None).
        error_message: Custom error message.
    """

    def __init__(
        self,
        seconds: int | None = None,
        multiplier: float = 1.0,
        error_message: str | None = None,
    ):
        if seconds is None:
            seconds = get_test_timeout(multiplier)
        if error_message is None:
            error_message = f"test timed out after {seconds}s."
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        logging.error("Timeout - sending signal")
        import faulthandler

        faulthandler.dump_traceback()
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


class TemporaryExperiment:
    """Context manager for temporary experiments with automatic timeout.

    Args:
        name: Experiment name.
        workdir: Working directory. If None, creates a temporary directory.
        timeout_multiplier: Multiplier for the centralized timeout.
        run_mode: Run mode for the experiment.
    """

    def __init__(
        self,
        name: str,
        workdir: Path | None = None,
        timeout_multiplier: float = 1.0,
        run_mode: RunMode = RunMode.NORMAL,
    ):
        self.name = name
        self.workdir = workdir
        self.clean_workdir = workdir is None
        self.run_mode = run_mode
        self.timeout = timeout(multiplier=timeout_multiplier)

    def __enter__(self) -> experiment:
        self._start_time = time.time()

        if self.clean_workdir:
            self.workdir = TemporaryDirectory(prefix="xpm", suffix=self.name)
            workdir = self.workdir.__enter__()
        else:
            workdir = self.workdir

        self.experiment = experiment(workdir, self.name, run_mode=self.run_mode)
        self.experiment.__enter__()

        # Set some useful environment variables
        self.experiment.workspace.launcher.setenv(
            "PYTHONPATH", str(Path(__file__).parents[2])
        )
        self.timeout.__enter__()

        logging.info("Created new temporary experiment (%s)", workdir)
        return self.experiment

    def __exit__(self, *args):
        elapsed = time.time() - self._start_time
        logging.info(
            "Experiment '%s' completed in %.2fs (timeout: %ds)",
            self.name,
            elapsed,
            self.timeout.seconds,
        )

        self.experiment.__exit__(*args)
        self.timeout.__exit__(*args)
        if self.clean_workdir:
            self.workdir.__exit__(*args)


def is_posix():
    try:
        import posix  # noqa: F401

        return True
    except ImportError:
        return False
