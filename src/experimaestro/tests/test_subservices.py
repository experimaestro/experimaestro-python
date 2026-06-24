"""Tests for the Service sub-services abstraction."""

import pytest

from experimaestro.scheduler.services import Service, resolve_service


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


def test_full_state_dict_omits_subservices_for_simple_service():
    # Backward compatible: a plain service does not grow a subservices key.
    d = _Svc("a").full_state_dict()
    assert "subservices" not in d
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
