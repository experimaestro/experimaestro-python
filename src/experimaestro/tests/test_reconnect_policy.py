"""Reconnection policy of the remote (SSH) state provider"""

from pydantic import TypeAdapter

from experimaestro.scheduler.remote.reconnect import ReconnectPolicy
from experimaestro.settings import ReconnectSettings, Settings


def test_backoff():
    """The delay doubles with each failed attempt, up to the maximum"""
    policy = ReconnectPolicy(
        ReconnectSettings(max_retries=10, window=100, delay=1, max_delay=5)
    )
    delays = []
    for now in range(5):
        delays.append(policy.next_delay(now))
        policy.attempted(now, success=False)
    assert delays == [1, 2, 4, 5, 5]

    # A success resets the delay
    policy.attempted(5, success=True)
    assert policy.next_delay(5) == 1


def test_max_retries_within_window():
    """At most max_retries attempts within the time window, whether they
    succeeded or not (a connection dropping right after reconnecting)"""
    policy = ReconnectPolicy(ReconnectSettings(max_retries=2, window=10, delay=1))
    policy.attempted(0, success=True)
    policy.attempted(1, success=True)
    assert policy.attempts(5) == 2
    assert policy.next_delay(5) is None

    # The first attempt leaves the window
    assert policy.attempts(10) == 1
    assert policy.next_delay(10) == 1


def test_disabled():
    policy = ReconnectPolicy(ReconnectSettings(max_retries=0))
    assert policy.next_delay(0) is None


def test_settings():
    """Global settings, overridden in the SSH settings of a workspace"""
    settings = TypeAdapter(Settings).validate_python(
        {
            "reconnect": {"max_retries": 3, "window": 60},
            "workspaces": [
                {
                    "id": "remote",
                    "path": "/remote",
                    "ssh": {"host": "cluster", "reconnect": {"max_retries": 7}},
                }
            ],
        }
    )
    assert settings.reconnect == ReconnectSettings(max_retries=3, window=60)
    assert settings.workspaces[0].ssh.reconnect.max_retries == 7
    assert ReconnectSettings() == ReconnectSettings(
        max_retries=5, window=300, delay=1, max_delay=30
    )
