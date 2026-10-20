import sys

# Set test flag immediately to prevent cli/__init__.py from calling basicConfig
sys._called_from_test = True

import faulthandler  # noqa: E402
import logging  # noqa: E402
import pytest  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402

# Add NullHandler to root logger to prevent lastResort handler from causing
# double logging when using pytest -s with --log-cli-level
logging.getLogger().addHandler(logging.NullHandler())

from experimaestro.connectors import Process  # noqa: E402

# Set shorter poll interval for tests (before any imports that read it)
os.environ.setdefault("XPM_POLL_INTERVAL_MAX", "5.0")
os.environ.setdefault("XPM_PROCESS_POLL_INTERVAL", "0.01")


class MockProcess(Process):
    """Mock process for testing that immediately returns success.

    Used to create valid lock files in tests without actual job processes.
    """

    def __init__(self):
        pass

    @classmethod
    def fromspec(cls, connector, definition):
        return cls()

    def tospec(self):
        return {"type": "mock"}

    def wait(self) -> int:
        return 0

    async def aio_wait(self) -> int:
        return 0

    def kill(self):
        pass


# Register MockProcess handler at module load time
# Call handler() first to trigger real initialization from entry points
Process.handler("mock")  # This initializes HANDLERS if needed
Process.HANDLERS["mock"] = MockProcess


# --- Hang watchdog --------------------------------------------------------
#
# Opt-in (XPM_TEST_THREAD_DUMP_AFTER=<seconds>): arms
# faulthandler.dump_traceback_later so that every thread's stack is dumped
# after that delay per test. Off by default: faulthandler reads the frames of
# the other threads while they run, which can crash (segfault) or hang the
# process — it did so in CI on slow tests. pytest-timeout (300s) already dumps
# every thread's stack, safely, when a test hangs. ``repeat`` is off so
# legitimately-slow tests don't spam the log.

_THREAD_DUMP_AFTER = float(os.environ.get("XPM_TEST_THREAD_DUMP_AFTER", "0"))


@pytest.fixture(autouse=True)
def _hang_watchdog():
    """Dump every thread's stack if the test hangs (when enabled)"""
    if _THREAD_DUMP_AFTER <= 0:
        yield
        return
    # faulthandler.dump_traceback_later replaces any previous timer, so
    # arming it per-test is safe. Writes to the real stderr: sys.stderr is
    # pytest's capture, lost if the process dies.
    faulthandler.dump_traceback_later(
        _THREAD_DUMP_AFTER, repeat=False, file=sys.__stderr__
    )
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _close_watchers():
    """Stop the file watchers a test leaves behind

    The scheduler keeps one job event reader per workspace for its lifetime
    and tests often leave state providers open; as each test uses its own
    workspace, their watcher threads and polled directories would otherwise
    accumulate over the session and slow the following tests down.
    """
    yield
    from experimaestro.filewatcher import FileWatcherService
    from experimaestro.scheduler.base import Scheduler

    # The scheduler's job event readers (one per workspace, restarted when an
    # experiment uses the workspace again)
    if Scheduler.has_instance():
        Scheduler.instance()._stop_job_event_reader()
    service = FileWatcherService.instance()
    with service._watches_lock:
        watches = list(service._directory_watches)
    for watch in watches:
        watch.close()


@pytest.fixture(scope="session")
def xpmdirectory(tmp_path_factory):
    """Sets a temporary main directory"""
    workdir = tmp_path_factory.mktemp("workdir")
    logging.info("Main XPM directory is %s", workdir)
    os.environ["XPM_WORKDIR"] = str(workdir)

    yield workdir

    if os.environ.get("XPM_KEEPWORKDIR", False) == "1":
        logging.warning("NOT Removing %s" % workdir)
    else:
        shutil.rmtree(workdir)


@pytest.fixture(scope="function", autouse=True)
def reset_scheduler():
    """Reset scheduler state between tests to avoid state leakage with singleton pattern"""
    from experimaestro.scheduler.base import Scheduler
    from experimaestro.webui import WebUIServer
    from experimaestro.tokens import CounterToken
    from experimaestro.core.partial_lock import PartialJobResource
    from experimaestro.dynamic import ResourcePoller

    # Clear token and resource caches
    CounterToken.TOKENS.clear()
    PartialJobResource.RESOURCES.clear()

    # Reset ResourcePoller singleton
    ResourcePoller.reset()

    # Get the singleton instance if it exists
    if Scheduler._instance is not None:
        scheduler = Scheduler._instance
        # Clear job registrations but keep scheduler running
        logging.debug(
            f"FIXTURE: Clearing scheduler before test - jobs count: {len(scheduler.jobs)}"
        )
        # Clear experiment references from all jobs
        for job in scheduler.jobs.values():
            job.experiments.clear()
        scheduler.jobs.clear()
        scheduler.waitingjobs.clear()
        scheduler.experiments.clear()
        # Clear state provider experiment providers to avoid stale references
        if (
            hasattr(scheduler, "state_provider")
            and scheduler.state_provider is not None
        ):
            # Close all experiment providers
            for provider in scheduler.state_provider.experiment_providers.values():
                provider.close()
            scheduler.state_provider.experiment_providers.clear()
            logging.debug("FIXTURE: Cleared state provider experiment providers")

        # Also clear listeners to prevent stale listeners
        scheduler.clear_listeners()

        # Re-add state_provider as listener if it exists
        if (
            hasattr(scheduler, "state_provider")
            and scheduler.state_provider is not None
        ):
            scheduler.addlistener(scheduler.state_provider)

    # Reset server instance too
    if WebUIServer._instance is not None:
        logging.debug("FIXTURE: Clearing server instance")
        WebUIServer.clear_instance()

    yield

    # Cleanup after test - clear again
    if Scheduler._instance is not None:
        scheduler = Scheduler._instance
        logging.debug(
            f"FIXTURE: Clearing scheduler after test - jobs count: {len(scheduler.jobs)}"
        )
        # Clear experiment references from all jobs
        for job in scheduler.jobs.values():
            job.experiments.clear()
        scheduler.jobs.clear()
        scheduler.waitingjobs.clear()
        scheduler.experiments.clear()
        # Clear state provider experiment providers
        if (
            hasattr(scheduler, "state_provider")
            and scheduler.state_provider is not None
        ):
            for provider in scheduler.state_provider.experiment_providers.values():
                provider.close()
            scheduler.state_provider.experiment_providers.clear()
        scheduler.clear_listeners()
        # Re-add state_provider as listener if it exists
        if (
            hasattr(scheduler, "state_provider")
            and scheduler.state_provider is not None
        ):
            scheduler.addlistener(scheduler.state_provider)

    # Reset server after test
    if WebUIServer._instance is not None:
        WebUIServer.clear_instance()

    # Clear token and resource caches after test
    CounterToken.TOKENS.clear()
    PartialJobResource.RESOURCES.clear()

    # Reset ResourcePoller singleton after test
    ResourcePoller.reset()


def pytest_configure(config):
    # Note: sys._called_from_test is set at module level (top of file)
    # to ensure it's set before any imports that check for it

    # Enable testing mode with polling for reliable file watching in tests
    from experimaestro.filewatcher import FileWatcherService

    FileWatcherService.configure(testing_mode=True, polling_interval=0.01)


def pytest_unconfigure(config):
    import sys  # This was missing from the manual

    del sys._called_from_test
