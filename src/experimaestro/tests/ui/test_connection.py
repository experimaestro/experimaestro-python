"""The connection state of the state provider (a remote one reconnects when
its connection drops) is shown on every page"""

import asyncio

import pytest

from experimaestro.scheduler.state_status import ConnectionState

from .conftest import DRIVERS

pytestmark = pytest.mark.anyio


async def shown(ui, expected: str | None, timeout: float = 5.0) -> str | None:
    """The connection state shown, once it is the expected one (events are
    delivered asynchronously)"""
    deadline = asyncio.get_running_loop().time() + timeout
    while (state := await ui.connection_state()) != expected:
        if asyncio.get_running_loop().time() > deadline:
            break
        await asyncio.sleep(0.05)
    return state


async def change(provider, state: ConnectionState, **details) -> None:
    # Notified from another thread, as the remote provider does
    await asyncio.to_thread(provider.set_connection_state, state, **details)


async def test_connection_state(ui, provider):
    assert await ui.connection_state() is None

    await change(provider, ConnectionState.RECONNECTING, attempt=1, max_retries=5)
    assert await shown(ui, "reconnecting") == "reconnecting"

    await change(provider, ConnectionState.CONNECTED, attempt=1, max_retries=5)
    assert await shown(ui, None) is None

    await change(
        provider, ConnectionState.DISCONNECTED, attempt=5, max_retries=5, error="gone"
    )
    assert await shown(ui, "disconnected") == "disconnected"


async def test_reconnected_shows_the_state_again(ui, provider):
    """Once reconnected, the interfaces read the state again (events were
    missed while disconnected)"""
    await ui.open_experiment("baseline")
    before = await ui.visible_jobs()
    await change(provider, ConnectionState.RECONNECTING, attempt=1, max_retries=5)
    assert await shown(ui, "reconnecting") == "reconnecting"

    reads = []
    fetch_jobs = provider._fetch_jobs

    def record(*args, **kwargs):
        reads.append(args[0])
        return fetch_jobs(*args, **kwargs)

    provider._fetch_jobs = record
    # As the remote provider does once reconnected
    provider._query_cache.drop_if(lambda key, entry: key[0] != "services")
    await change(provider, ConnectionState.CONNECTED, attempt=1, max_retries=5)
    assert await shown(ui, None) is None

    deadline = asyncio.get_running_loop().time() + 5
    while not reads and asyncio.get_running_loop().time() < deadline:
        await ui.connection_state()
        await asyncio.sleep(0.05)
    assert "baseline" in reads
    assert await ui.visible_jobs() == before


@pytest.mark.parametrize("driver", sorted(DRIVERS))
async def test_connection_state_on_open(provider, driver):
    """An interface opened while reconnecting shows it"""
    provider._connection_state = ConnectionState.RECONNECTING
    async with DRIVERS[driver](provider) as ui:
        assert await ui.connection_state() == "reconnecting"
