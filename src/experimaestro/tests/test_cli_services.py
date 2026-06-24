"""Tests for the `experiments services` CLI commands."""

from click.testing import CliRunner

from experimaestro.cli import _iter_service_rows, cli
from experimaestro.scheduler.services import Service


class _Svc(Service):
    def __init__(self, id, description="", url=None, subs=None):
        super().__init__()
        self.id = id
        self._desc = description
        self.url = url
        self._subs = subs

    def description(self):
        return self._desc

    def subservices(self):
        return self._subs if self._subs is not None else [self]


def test_iter_service_rows_self_inclusive_subservice():
    # A service that is its own sub-service plus a wandb sub-service: the bare id
    # is the service itself; the extra sub gets a "<id>/<sub>" address.
    tb = _Svc("tensorboard", "TensorBoard", url="http://tb")
    tb._subs = [tb, _Svc("wandb", "W&B sync")]

    rows = list(_iter_service_rows({"tensorboard": tb}))

    assert rows == [
        ("tensorboard", "TensorBoard", "STOPPED", "http://tb"),
        ("tensorboard/wandb", "W&B sync", "STOPPED", None),
    ]


def test_iter_service_rows_pure_composite():
    # A composite that is not itself a sub-service: only "<id>/<sub>" rows.
    sub_a = _Svc("tensorboard", "TB view", url="http://tb")
    sub_b = _Svc("wandb", "W&B sync")
    mon = _Svc("monitoring", "Monitoring", subs=[sub_a, sub_b])
    plain = _Svc("tb", "TensorBoard", url="http://x")

    rows = list(_iter_service_rows({"tb": plain, "monitoring": mon}))

    assert rows == [
        ("monitoring/tensorboard", "TB view", "STOPPED", "http://tb"),
        ("monitoring/wandb", "W&B sync", "STOPPED", None),
        ("tb", "TensorBoard", "STOPPED", "http://x"),
    ]


def test_services_subcommands_registered():
    result = CliRunner().invoke(cli, ["experiments", "services", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "start" in result.output
