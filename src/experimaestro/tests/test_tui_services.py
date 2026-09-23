"""Tests for the TUI services sub-service expansion."""

from experimaestro.tui.widgets.services import expand_services


class _Live:
    def __init__(self, id, subs=None):
        self.id = id
        self._subs = subs

    def subservices(self):
        return self._subs if self._subs is not None else [self]


class _Mock:
    """Stand-in for a MockService: exposes to_service()."""

    def __init__(self, id, live):
        self.id = id
        self._live = live

    def to_service(self):
        return self._live


def test_expand_keeps_parent_and_adds_subservices():
    wandb = _Live("wandb")
    tb_live = _Live("tensorboard")
    tb_live._subs = [tb_live, wandb]
    tb_mock = _Mock("tensorboard", tb_live)

    other = _Mock("other", _Live("other"))

    result = expand_services([tb_mock, other])

    # parent kept as the original (mock) object; extra sub added as the live sub
    assert result["tensorboard"] is tb_mock
    assert result["tensorboard/wandb"] is wandb
    assert result["other"] is other
    assert set(result) == {"tensorboard", "tensorboard/wandb", "other"}


def test_expand_robust_to_missing_subservice_api():
    class _NoSub:
        id = "x"

        def to_service(self):
            return self  # has no subservices() -> treated as a single service

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
    """Verify that SSHLocalService.state reports STARTING during get_url() execution"""
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
    mock_inner.sync_include_patterns = None

    sync_started = threading.Event()
    finish_sync = threading.Event()

    def slow_sync(remote_path, include=None):
        sync_started.set()
        finish_sync.wait(timeout=2.0)
        return Path("/tmp/local")

    mock_state_provider = MagicMock()
    mock_state_provider.sync_path.side_effect = slow_sync

    def fake_inner_get_url():
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

    sync_started.wait(timeout=2.0)
    # While sync is executing, ssh_service.state must be STARTING
    assert ssh_service.state == ServiceState.STARTING

    finish_sync.set()
    t.join(timeout=2.0)

    # After completion, state should be RUNNING
    assert ssh_service.state == ServiceState.RUNNING
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
    widget.notify.assert_called_with("Service 'tensorboard' is already starting...", severity="warning")

    # If state is RUNNING: should NOT launch another worker
    mock_service.state = ServiceState.RUNNING
    widget._starting_services.clear()
    widget.action_start_service()
    assert widget._start_service_worker.call_count == 1
    widget.notify.assert_called_with("Service 'tensorboard' is already running", severity="information")
