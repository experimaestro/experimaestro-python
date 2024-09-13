import sys
import pytest
from experimaestro.connectors.ssh import SshPath

# --- Test SSH path and SSH path manipulation


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="requires python3.10 or higher")
def test_absolute():
    path = SshPath("ssh://host//a/path")
    assert path.host == "host"
    assert path.is_absolute()


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="requires python3.10 or higher")
def test_relative():
    path = SshPath("ssh://host")
    assert path.host == "host"
    assert path.hostpath == ""
    assert path._parts == []
    assert not path.is_absolute()


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="requires python3.10 or higher")
def test_relative_withpath():
    path = SshPath("ssh://host/relative/path")
    assert path.host == "host"
    assert path.hostpath == "relative/path"
    assert not path.is_absolute()


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="requires python3.10 or higher")
def test_relative_absolute():
    path = SshPath("ssh://host") / "/absolute/path"
    assert path.host == "host"
    assert path.hostpath == "/absolute/path"
    assert path.is_absolute()


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="requires python3.10 or higher")
def test_relative_compose():
    path = SshPath("ssh://host/abc") / "relative/path"
    assert path.host == "host"
    assert path.hostpath == "abc/relative/path"
    assert not path.is_absolute()
