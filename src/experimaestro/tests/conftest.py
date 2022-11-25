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


# Sets a flag
def pytest_configure(config):
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    import sys  # This was missing from the manual

    del sys._called_from_test
