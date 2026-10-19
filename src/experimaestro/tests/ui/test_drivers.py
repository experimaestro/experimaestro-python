"""Every driver implements the whole UI driver interface"""

import inspect

import pytest

from .drivers import UIDriver
from .tui_driver import TuiDriver
from .web_driver import WebDriver

OPERATIONS = sorted(
    name
    for name, member in inspect.getmembers(UIDriver)
    if inspect.iscoroutinefunction(member)
)


def _drivers():
    yield TuiDriver
    yield WebDriver
    try:
        from .browser_driver import BrowserDriver
    except ImportError:  # playwright is in the optional "browser" group
        return
    yield BrowserDriver


@pytest.mark.parametrize("driver", list(_drivers()), ids=lambda d: d.name)
def test_driver_implements_interface(driver):
    missing = [name for name in OPERATIONS if not hasattr(driver, name)]
    assert not missing, f"{driver.__name__} does not implement {missing}"
    for name in OPERATIONS:
        assert inspect.iscoroutinefunction(getattr(driver, name)), name
