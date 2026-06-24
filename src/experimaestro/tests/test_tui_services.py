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
