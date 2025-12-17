import logging
import pytest
import os
import shutil


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
    from experimaestro.server import Server

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
        scheduler.listeners.clear()

        # Re-add state_provider as listener if it exists
        if (
            hasattr(scheduler, "state_provider")
            and scheduler.state_provider is not None
        ):
            scheduler.addlistener(scheduler.state_provider)

    # Reset server instance too
    if Server._instance is not None:
        logging.debug("FIXTURE: Clearing server instance")
        Server._instance = None

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
        scheduler.listeners.clear()
        # Re-add state_provider as listener if it exists
        if (
            hasattr(scheduler, "state_provider")
            and scheduler.state_provider is not None
        ):
            scheduler.addlistener(scheduler.state_provider)

    # Reset server after test
    if Server._instance is not None:
        Server._instance = None


# Sets a flag
def pytest_configure(config):
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    import sys  # This was missing from the manual

    del sys._called_from_test
