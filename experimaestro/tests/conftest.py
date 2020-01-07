import logging
import pytest
import os


@pytest.fixture(scope="session")
def xpmdirectory(tmp_path_factory):
    """Sets a temporary main directory"""
    fn = tmp_path_factory.mktemp("workdir")
    logging.info("Main XPM directory is %s", fn)
    os.environ["XPM_WORKDIR"] = str(fn)
    return fn
