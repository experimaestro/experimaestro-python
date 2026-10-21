"""Tests for the Service sub-services abstraction."""

import pytest

from experimaestro.scheduler.services import Service, resolve_service
from experimaestro.scheduler.state_provider import MockService, MockSubService


class _Svc(Service):
    """Minimal concrete service for testing."""

    def __init__(self, id: str):
        super().__init__()
        self.id = id


class _Composite(Service):
    """A service exposing several startable sub-services."""

    def __init__(self):
        super().__init__()
        self.id = "monitoring"
        self._subs = [_Svc("tensorboard"), _Svc("wandb")]

    def subservices(self):
        return self._subs


def test_default_service_is_its_own_subservice():
    s = _Svc("a")
    assert s.subservices() == [s]
    assert s.get_subservice("a") is s
    with pytest.raises(KeyError):
        s.get_subservice("missing")


def test_full_state_dict_simple_service():
    # An empty list (not a missing key, as with older data): the service is
    # its own single sub-service
    d = _Svc("a").full_state_dict()
    assert d["subservices"] == []
    assert d["service_id"] == "a"


def test_composite_lists_and_serializes_subservices():
    c = _Composite()
    assert [s.id for s in c.subservices()] == ["tensorboard", "wandb"]

    d = c.full_state_dict()
    assert [s["service_id"] for s in d["subservices"]] == ["tensorboard", "wandb"]
    # Each descriptor carries the fields the UIs/CLI render
    for sub in d["subservices"]:
        assert set(sub) == {"service_id", "description", "state", "url"}
        assert sub["state"] == "STOPPED"


def test_get_subservice():
    c = _Composite()
    assert c.get_subservice("wandb") is c._subs[1]
    with pytest.raises(KeyError):
        c.get_subservice("nope")


def test_resolve_service_composite_and_plain():
    c = _Composite()
    services = {"monitoring": c, "tb": _Svc("tb")}

    assert resolve_service(services, "monitoring/wandb") is c._subs[1]
    assert resolve_service(services, "tb") is services["tb"]

    with pytest.raises(KeyError):
        resolve_service(services, "monitoring/nope")
    with pytest.raises(KeyError):
        resolve_service(services, "absent")


def _mock(service_class: str, subservices) -> MockService:
    return MockService.from_full_state_dict(
        {
            "service_id": "monitoring",
            "class": service_class,
            "state_dict": {},
            "subservices": subservices,
        }
    )


def test_mock_lists_stored_subservices_without_recreating():
    """Listing does not import the service module (e.g. tensorboard)"""
    stored = _Composite().full_state_dict()["subservices"]
    mock = _mock("experimaestro_absent_module.Service", stored)

    subs = mock.subservices()
    assert [s.id for s in subs] == ["tensorboard", "wandb"]
    assert all(isinstance(s, MockSubService) for s in subs)
    assert mock._live_service is None and mock.error is None

    # A plain service is its own sub-service
    plain = _mock("experimaestro_absent_module.Service", [])
    assert plain.subservices() == [plain]


def test_mock_subservice_resolves_live_subservice():
    stored = _Composite().full_state_dict()["subservices"]
    mock = _mock(f"{__name__}._Composite", stored)
    wandb = mock.subservices()[1]
    assert wandb.state.name == "STOPPED"

    live = wandb.to_service()
    assert live is mock._live_service.get_subservice("wandb")
    assert wandb.state is live.state


def test_mock_without_stored_subservices_recreates_service():
    """Data written before sub-services were stored"""
    mock = _mock(f"{__name__}._Composite", None)
    assert [s.id for s in mock.subservices()] == ["tensorboard", "wandb"]
    assert isinstance(mock._live_service, _Composite)


def test_mock_full_state_dict_keeps_subservices():
    stored = _Composite().full_state_dict()["subservices"]
    mock = _mock("experimaestro_absent_module.Service", stored)
    assert (
        MockService.from_full_state_dict(mock.full_state_dict()).subservices()[1].id
        == "wandb"
    )


def test_process_service_failing_to_start(tmp_path):
    """A service whose process cannot be started is in error, not starting"""
    from experimaestro.scheduler.services import ProcessWebService, ServiceState

    class _Process(ProcessWebService):
        id = "proc"

        def _build_command(self):
            return ["true"]

        def _wait_for_ready(self):
            return "http://localhost:1"

    service = _Process(log_directory=tmp_path)
    # The log file cannot be opened
    service.stdout.mkdir()
    with pytest.raises(IsADirectoryError):
        service.get_url()
    assert service.state == ServiceState.ERROR

    # It can be started again once fixed
    service.stdout.rmdir()
    assert service.get_url() == "http://localhost:1"
    assert service.state == ServiceState.RUNNING
    service.stop()
