"""Regression test for ``experiment(python_path=...)``.

Task workers reconstruct the task module by file path and do not inherit the
launching interpreter's ``sys.path``. A task that imports a module *by name* at
run time therefore fails unless that module's root is made available to the
worker. ``experiment(python_path=[...])`` is the programmatic way to do so.
"""

import importlib
import sys

import pytest

from experimaestro import Param, Task, experiment
from experimaestro.scheduler import JobState

_PROBE_PKG = "xpm_python_path_probe"


class DynamicImportProbe(Task):
    """Imports ``module`` by name inside the worker (needs worker sys.path)."""

    module: Param[str]

    def execute(self):
        from pathlib import Path

        mod = importlib.import_module(self.module)
        Path("probe.txt").write_text(mod.VALUE)


def _write_probe_package(root):
    pkg = root / _PROBE_PKG
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 'imported-via-python-path'\n")
    return root


def test_python_path_makes_module_importable_in_worker(tmp_path):
    pkgroot = _write_probe_package(tmp_path / "pkgs")
    # The probe package must not be importable in the parent either.
    assert _PROBE_PKG not in sys.modules

    with experiment(
        tmp_path / "xp",
        "python_path_test",
        no_environmental_impact=True,
        python_path=[pkgroot],
    ):
        job = DynamicImportProbe.C(module=_PROBE_PKG).submit()
        assert job.__xpm__.job.wait() == JobState.DONE


@pytest.mark.xfail(
    reason="without python_path the worker cannot import the module", strict=True
)
def test_without_python_path_the_worker_cannot_import(tmp_path, monkeypatch):
    # Make sure the worker cannot find the package through the environment
    monkeypatch.delenv("PYTHONPATH", raising=False)
    _write_probe_package(tmp_path / "pkgs")
    sys.path_importer_cache.clear()

    with experiment(
        tmp_path / "xp", "python_path_control", no_environmental_impact=True
    ):
        job = DynamicImportProbe.C(module=_PROBE_PKG).submit()
        assert job.__xpm__.job.wait() == JobState.DONE
