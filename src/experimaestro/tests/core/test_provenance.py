"""Tests for per-parameter provenance (where/how a value was set)."""

import pytest

from experimaestro import Config, Param, field
from experimaestro.core.objects.config import SetInfo

pytestmark = pytest.mark.config


class A(Config):
    value: Param[int]
    opt: Param[int] = field(default=7, ignore_default=True)


def test_provenance_explicit_via_constructor():
    a = A.C(value=1)
    info = a.__xpm__.provenance_of("value")
    assert isinstance(info, SetInfo)
    assert info.kind == "explicit"
    assert "test_provenance.py" in info.location


def test_provenance_default():
    a = A.C(value=1)
    info = a.__xpm__.provenance_of("opt")
    assert info is not None and info.kind == "default"


def test_provenance_explicit_via_setattr():
    a = A.C(value=1)
    a.value = 2
    info = a.__xpm__.provenance_of("value")
    assert info.kind == "explicit"
    assert "test_provenance.py" in info.location


def test_provenance_records_call_site_line():
    a = A.C(value=1)
    info = a.__xpm__.provenance_of("value")
    # location is "<abspath>:<line>" of the .C(...) call above
    path, _, line = info.location.rpartition(":")
    assert path.endswith("test_provenance.py")
    assert int(line) > 0


def test_provenance_not_in_identifier():
    """Provenance must not affect the identifier."""
    a1 = A.C(value=1)
    a2 = A.C(value=1)
    assert a1.__xpm__.identifier.all == a2.__xpm__.identifier.all


def test_validation_error_includes_param_name_and_location():
    """A validation failure names the parameter and where it was set."""
    with pytest.raises(ValueError) as excinfo:
        A.C(value="not-an-int")
    message = str(excinfo.value)
    assert "parameter 'value'" in message
    assert "set at" in message
    assert "test_provenance.py" in message
