"""Tests for the TUI services sub-service expansion."""

from experimaestro.tui.widgets.services import expand_services


class _Live:
    def __init__(self, id, subs=None):
        self.id = id
        self._subs = subs

    def subservices(self):
        return self._subs if self._subs is not None else [self]


def test_expand_keeps_parent_and_adds_subservices():
    wandb = _Live("wandb")
    tb = _Live("tensorboard")
    tb._subs = [tb, wandb]
    other = _Live("other")

    result = expand_services([tb, other])

    assert result["tensorboard"] is tb
    assert result["tensorboard/wandb"] is wandb
    assert result["other"] is other
    assert set(result) == {"tensorboard", "tensorboard/wandb", "other"}


def test_expand_robust_to_missing_subservice_api():
    class _NoSub:
        id = "x"  # has no subservices() -> treated as a single service

    s = _NoSub()
    assert expand_services([s]) == {"x": s}


def test_ssh_local_service_multiple_get_url_does_not_duplicate_synchronizers():
    """Verify that multiple or concurrent get_url() calls do not duplicate syncs or synchronizers"""
    import threading
    import time
    from pathlib import Path
    from unittest.mock import MagicMock
    from experimaestro.scheduler.services import ServiceState
    from experimaestro.scheduler.remote.client import SSHLocalService

    mock_inner = MagicMock()
    mock_inner.id = "tensorboard"
    mock_inner.state = ServiceState.STOPPED
    mock_inner.url = None
    mock_inner.sync_include_patterns = ["*events.out.tfevents*"]

    def fake_inner_get_url():
        time.sleep(0.05)
        mock_inner.state = ServiceState.RUNNING
        mock_inner.url = "http://localhost:6006"
        return mock_inner.url

    mock_inner.get_url.side_effect = fake_inner_get_url

    mock_state_provider = MagicMock()
    mock_state_provider.sync_path.return_value = Path("/tmp/local")

    ssh_service = SSHLocalService(
        inner_service=mock_inner,
        state_provider=mock_state_provider,
        remote_paths=["/remote/runs"],
    )

    # Launch two concurrent get_url() calls
    results = []

    def caller():
        url = ssh_service.get_url()
        results.append(url)

    t1 = threading.Thread(target=caller)
    t2 = threading.Thread(target=caller)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both returned the correct url
    assert results == ["http://localhost:6006", "http://localhost:6006"]
    # Only 1 set of synchronizers created!
    assert len(ssh_service._synchronizers) == 1
    # sync_path called only once for initial sync
    assert mock_state_provider.sync_path.call_count == 1

    # Calling get_url() a third time when running does not sync again
    url3 = ssh_service.get_url()
    assert url3 == "http://localhost:6006"
    assert len(ssh_service._synchronizers) == 1
    assert mock_state_provider.sync_path.call_count == 1

    ssh_service.stop()


def test_ssh_local_service_starting_state():
    """SSHLocalService reports STARTING while the inner service starts

    The files are synchronized in the background: the service does not wait
    for the sync, and is RUNNING as soon as the inner service is.
    """
    import threading
    from pathlib import Path
    from unittest.mock import MagicMock
    from experimaestro.scheduler.services import ServiceState
    from experimaestro.scheduler.remote.client import SSHLocalService

    mock_inner = MagicMock()
    mock_inner.id = "tensorboard"
    mock_inner.state = ServiceState.STOPPED
    mock_inner.url = None
    mock_inner.sync_include_patterns = None

    sync_started = threading.Event()
    finish_sync = threading.Event()

    def slow_sync(remote_path, include=None):
        sync_started.set()
        finish_sync.wait(timeout=5.0)
        return Path("/tmp/local")

    mock_state_provider = MagicMock()
    mock_state_provider.sync_path.side_effect = slow_sync

    inner_starting = threading.Event()
    finish_inner = threading.Event()

    def fake_inner_get_url():
        inner_starting.set()
        assert finish_inner.wait(timeout=5.0)
        mock_inner.state = ServiceState.RUNNING
        mock_inner.url = "http://localhost:6006"
        return mock_inner.url

    mock_inner.get_url.side_effect = fake_inner_get_url

    ssh_service = SSHLocalService(
        inner_service=mock_inner,
        state_provider=mock_state_provider,
        remote_paths=["/remote/runs"],
    )

    t = threading.Thread(target=ssh_service.get_url)
    t.start()
    try:
        # While the inner service is starting, the state must be STARTING
        assert inner_starting.wait(timeout=5.0)
        assert ssh_service.state == ServiceState.STARTING

        finish_inner.set()
        t.join(timeout=5.0)
        assert not t.is_alive()

        # The service is running even though the first sync is not over
        assert sync_started.wait(timeout=5.0)
        assert ssh_service.state == ServiceState.RUNNING
        assert ssh_service.sync_status == "⟳ Syncing"
    finally:
        finish_inner.set()
        finish_sync.set()
        t.join(timeout=5.0)
        ssh_service.stop()


def test_services_list_double_start_guard():
    """Verify that ServicesList ignores second start_service call while service is starting"""
    from unittest.mock import MagicMock
    from experimaestro.tui.widgets.services import ServicesList
    from experimaestro.scheduler.services import ServiceState

    state_provider = MagicMock()
    widget = ServicesList(state_provider=state_provider)

    mock_service = MagicMock()
    mock_service.id = "tensorboard"
    mock_service.state = ServiceState.STOPPED

    widget._get_selected_service = MagicMock(return_value=mock_service)
    widget.notify = MagicMock()
    widget._refresh_all_services = MagicMock()
    widget._start_service_worker = MagicMock()

    # First call: should initiate start
    widget.action_start_service()
    assert widget._start_service_worker.call_count == 1
    assert "tensorboard" in widget._starting_services

    # Second call while in _starting_services: should NOT launch another worker
    widget.action_start_service()
    assert widget._start_service_worker.call_count == 1
    widget.notify.assert_called_with(
        "Service 'tensorboard' is already starting", severity="warning"
    )

    # If state is RUNNING: should NOT launch another worker
    mock_service.state = ServiceState.RUNNING
    widget._starting_services.clear()
    widget.action_start_service()
    assert widget._start_service_worker.call_count == 1
    widget.notify.assert_called_with(
        "Service 'tensorboard' is already running", severity="warning"
    )
