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
        # Disable logging error reporting for cleanup (streams may close)
        logging.raiseExceptions = False

        # Use stderr directly to avoid logging errors on closed streams
        import sys

        try:
            print("Timeout - sending signal", file=sys.stderr)  # noqa: T201
        except Exception:
            pass  # Ignore errors if stderr is closed

        import faulthandler

        try:
            faulthandler.dump_traceback()
        except Exception:
            pass  # Ignore errors if output streams are closed

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
        no_environmental_impact: Disable carbon tracking (True by default for tests).
    """

    def __init__(
        self,
        name: str,
        workdir: Path | None = None,
        timeout_multiplier: float = 1.0,
        run_mode: RunMode = RunMode.NORMAL,
        no_environmental_impact: bool = True,
    ):
        self.name = name
        self.workdir = workdir
        self.clean_workdir = workdir is None
        self.run_mode = run_mode
        self.no_environmental_impact = no_environmental_impact
        self.timeout = timeout(multiplier=timeout_multiplier)

    def __enter__(self) -> experiment:
        self._start_time = time.time()

        if self.clean_workdir:
            self.workdir = TemporaryDirectory(prefix="xpm", suffix=self.name)
            workdir = self.workdir.__enter__()
        else:
            workdir = self.workdir

        self.experiment = experiment(
            workdir,
            self.name,
            run_mode=self.run_mode,
            no_environmental_impact=self.no_environmental_impact,
        )
        self.experiment.__enter__()

        # Set some useful environment variables
        self.experiment.workspace.launcher.setenv(
            "PYTHONPATH", str(Path(__file__).parents[2])
        )
        self.timeout.__enter__()

        logging.info("Created new temporary experiment (%s)", workdir)
        return self.experiment

    def _log_debug_info_on_timeout(self):
        """Log debugging information when a timeout occurs."""
        # Re-enable logging exceptions (disabled by signal handler)
        logging.raiseExceptions = True

        logger = logging.getLogger("xpm.test.timeout")

        logger.error("=" * 60)
        logger.error("TIMEOUT DEBUG INFO for experiment '%s'", self.name)
        logger.error("=" * 60)

        # Log job states
        resources_seen = {}
        try:
            from experimaestro.dynamic import DynamicDependency

            jobs = list(self.experiment.jobs.values())
            if not jobs:
                logger.error("Jobs: none")
            else:
                from collections import Counter

                state_counts = Counter(job.state.name for job in jobs)
                summary = ", ".join(
                    f"{name}: {count}" for name, count in sorted(state_counts.items())
                )
                logger.error("Jobs (%d total): %s", len(jobs), summary)

                for job in jobs:
                    # Collect dynamic resources from all jobs
                    for dep in job.dependencies:
                        if isinstance(dep, DynamicDependency):
                            resources_seen[id(dep.origin)] = dep.origin

                    if job.state.name != "done":
                        dyn_deps = [
                            repr(dep)
                            for dep in job.dependencies
                            if isinstance(dep, DynamicDependency)
                        ]
                        dep_str = f" {dyn_deps}" if dyn_deps else ""
                        logger.error(
                            "  [%s] %s (%s)%s",
                            job.state.name,
                            job.name,
                            job.identifier[:12],
                            dep_str,
                        )
        except Exception as e:
            logger.error("Error getting jobs: %s", e)

        # Log dynamic resource states
        try:
            if not resources_seen:
                logger.error("Dynamic resources: none found")
            else:
                for resource in resources_seen.values():
                    logger.error("  %r", resource)
        except Exception as e:
            logger.error("Error getting resources: %s", e)

        logger.error("=" * 60)

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start_time

        # Check if this is a timeout - print debug info before cleanup
        if exc_type is TimeoutError:
            logging.error(
                "Experiment '%s' TIMED OUT after %.2fs (timeout: %ds)",
                self.name,
                elapsed,
                self.timeout.seconds,
            )
            self._log_debug_info_on_timeout()
        else:
            logging.info(
                "Experiment '%s' completed in %.2fs (timeout: %ds)",
                self.name,
                elapsed,
                self.timeout.seconds,
            )

        self.experiment.__exit__(exc_type, exc_val, exc_tb)
        self.timeout.__exit__(exc_type, exc_val, exc_tb)
        if self.clean_workdir:
            self.workdir.__exit__(exc_type, exc_val, exc_tb)


def is_posix():
    try:
        import posix  # noqa: F401

        return True
    except ImportError:
        return False
