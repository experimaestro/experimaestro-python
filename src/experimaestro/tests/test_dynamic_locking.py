"""Tests for the dynamic locking infrastructure.

Tests for DynamicLockFile, JobDependencyLock, and related classes.
"""

import json
import pytest
from pathlib import Path

from experimaestro.locking import (
    DynamicLockFile,
    JobDependencyLock,
    LockError,
)
from experimaestro.tokens import TokenLockFile
from experimaestro.core.partial_lock import PartialLockFile


# --- Test implementations ---


class MockResource:
    """Mock resource for testing DynamicLockFile."""

    def __init__(self):
        self.accounted_count = 0

    def _account_lock_file(self, lf):
        self.accounted_count += 1

    def _unaccount_lock_file(self, lf):
        self.accounted_count -= 1


class MockLockFile(DynamicLockFile):
    """Mock implementation of DynamicLockFile for testing."""

    value: int

    def from_information(self, info) -> None:
        if info is None:
            self.value = 0
        elif isinstance(info, dict):
            self.value = info.get("value", 0)
        else:
            raise ValueError(f"Invalid information format: {info}")

    def to_information(self) -> dict:
        return {"value": self.value}


class MockJobLock(JobDependencyLock):
    """Mock implementation of JobDependencyLock for testing."""

    def __init__(self, lock_file_path: Path = None):
        self.lock_file_path = lock_file_path
        self.acquired = False
        self.released = False

    def acquire(self) -> None:
        self.verify_lock_file()
        self.acquired = True

    def release(self) -> None:
        self.released = True
        super().release()


# --- DynamicLockFile tests ---


class TestDynamicLockFile:
    """Tests for DynamicLockFile."""

    def test_create_writes_json_file(self, tmp_path: Path):
        """create() should write a JSON file with correct structure."""
        lock_path = tmp_path / "test.lock"
        job_uri = "/path/to/job"
        resource = MockResource()

        lock_file = MockLockFile.create(
            lock_path, resource, job_uri, information={"value": 42}
        )

        assert lock_path.is_file()
        content = json.loads(lock_path.read_text())
        assert content["job_uri"] == job_uri
        assert content["information"] == {"value": 42}
        assert lock_file.job_uri == job_uri
        assert lock_file.value == 42

    def test_create_with_none_information(self, tmp_path: Path):
        """create() with None information should use defaults."""
        lock_path = tmp_path / "test.lock"
        job_uri = "/path/to/job"
        resource = MockResource()

        lock_file = MockLockFile.create(lock_path, resource, job_uri, information=None)

        assert lock_file.value == 0
        content = json.loads(lock_path.read_text())
        assert content["information"] == {"value": 0}

    def test_load_reads_json_file(self, tmp_path: Path):
        """Loading should read JSON file correctly."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text(
            json.dumps({"job_uri": "/some/job", "information": {"value": 123}})
        )
        resource = MockResource()

        lock_file = MockLockFile(lock_path, resource)

        assert lock_file.job_uri == "/some/job"
        assert lock_file.value == 123

    def test_load_missing_file(self, tmp_path: Path):
        """Loading missing file should set defaults."""
        lock_path = tmp_path / "nonexistent.lock"
        resource = MockResource()

        lock_file = MockLockFile(lock_path, resource)

        assert lock_file.job_uri is None
        # Note: value is not set when file doesn't exist

    def test_load_invalid_information_raises(self, tmp_path: Path):
        """Loading with invalid information format should raise."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "job_uri": "/some/job",
                    "information": "invalid",  # Should be dict
                }
            )
        )
        resource = MockResource()

        with pytest.raises(ValueError, match="Invalid information format"):
            MockLockFile(lock_path, resource)

    def test_delete_removes_file(self, tmp_path: Path):
        """delete() should remove the lock file."""
        lock_path = tmp_path / "test.lock"
        resource = MockResource()
        lock_file = MockLockFile.create(
            lock_path, resource, "/job", information={"value": 1}
        )

        assert lock_path.is_file()
        lock_file.delete()
        assert not lock_path.is_file()

    def test_delete_missing_file_noop(self, tmp_path: Path):
        """delete() on missing file should not raise."""
        lock_path = tmp_path / "nonexistent.lock"
        resource = MockResource()
        lock_file = MockLockFile(lock_path, resource)

        # Should not raise
        lock_file.delete()


# --- JobDependencyLock tests ---


class TestJobDependencyLock:
    """Tests for JobDependencyLock."""

    def test_verify_lock_file_passes_when_exists(self, tmp_path: Path):
        """verify_lock_file() should pass when file exists."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("{}")

        lock = MockJobLock(lock_file_path=lock_path)
        lock.verify_lock_file()  # Should not raise

    def test_verify_lock_file_raises_when_missing(self, tmp_path: Path):
        """verify_lock_file() should raise LockError when file is missing."""
        lock_path = tmp_path / "nonexistent.lock"

        lock = MockJobLock(lock_file_path=lock_path)

        with pytest.raises(LockError, match="Lock file missing"):
            lock.verify_lock_file()

    def test_verify_lock_file_noop_when_path_none(self):
        """verify_lock_file() should be no-op when lock_file_path is None."""
        lock = MockJobLock(lock_file_path=None)
        lock.verify_lock_file()  # Should not raise

    def test_release_deletes_lock_file(self, tmp_path: Path):
        """release() should delete the lock file."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("{}")

        lock = MockJobLock(lock_file_path=lock_path)
        lock.release()

        assert not lock_path.is_file()

    def test_release_noop_when_path_none(self):
        """release() should not raise when lock_file_path is None."""
        lock = MockJobLock(lock_file_path=None)
        lock.release()  # Should not raise

    def test_context_manager_acquire_release(self, tmp_path: Path):
        """Context manager should acquire on enter and release on exit."""
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("{}")

        lock = MockJobLock(lock_file_path=lock_path)

        with lock:
            assert lock.acquired
            assert lock_path.is_file()

        assert lock.released
        assert not lock_path.is_file()

    def test_acquire_fails_if_lock_file_missing(self, tmp_path: Path):
        """acquire() should fail if lock file verification fails."""
        lock_path = tmp_path / "nonexistent.lock"
        lock = MockJobLock(lock_file_path=lock_path)

        with pytest.raises(LockError):
            lock.acquire()


# --- TokenLockFile tests ---


class TestTokenLockFile:
    """Tests for TokenLockFile."""

    def test_create_with_count(self, tmp_path: Path):
        """create() should store count in information."""
        lock_path = tmp_path / "test.token"
        job_uri = "/path/to/job"
        resource = MockResource()

        lock_file = TokenLockFile.create(
            lock_path, resource, job_uri, information={"count": 5}
        )

        assert lock_file.count == 5
        content = json.loads(lock_path.read_text())
        assert content["information"]["count"] == 5

    def test_load_json_format(self, tmp_path: Path):
        """Loading should read JSON format correctly."""
        lock_path = tmp_path / "test.token"
        lock_path.write_text(
            json.dumps({"job_uri": "/some/job", "information": {"count": 10}})
        )
        resource = MockResource()

        lock_file = TokenLockFile(lock_path, resource)

        assert lock_file.job_uri == "/some/job"
        assert lock_file.count == 10

    def test_to_information(self, tmp_path: Path):
        """to_information() should return count dict."""
        lock_path = tmp_path / "test.token"
        resource = MockResource()
        lock_file = TokenLockFile.create(
            lock_path, resource, "/job", information={"count": 3}
        )

        assert lock_file.to_information() == {"count": 3}

    def test_from_information_none(self, tmp_path: Path):
        """from_information(None) should set count to 0."""
        lock_path = tmp_path / "test.token"
        resource = MockResource()
        lock_file = TokenLockFile.create(lock_path, resource, "/job", information=None)

        assert lock_file.count == 0


# --- PartialLockFile tests ---


class TestPartialLockFile:
    """Tests for PartialLockFile."""

    def test_create_with_partial_name(self, tmp_path: Path):
        """create() should store partial_name in information."""
        lock_path = tmp_path / "holder.json"
        job_uri = "/path/to/job"
        resource = MockResource()

        lock_file = PartialLockFile.create(
            lock_path, resource, job_uri, information={"partial_name": "checkpoints"}
        )

        assert lock_file.partial_name == "checkpoints"
        content = json.loads(lock_path.read_text())
        assert content["information"]["partial_name"] == "checkpoints"

    def test_load_json_format(self, tmp_path: Path):
        """Loading should read JSON format correctly."""
        lock_path = tmp_path / "holder.json"
        lock_path.write_text(
            json.dumps(
                {"job_uri": "/some/job", "information": {"partial_name": "outputs"}}
            )
        )
        resource = MockResource()

        lock_file = PartialLockFile(lock_path, resource)

        assert lock_file.job_uri == "/some/job"
        assert lock_file.partial_name == "outputs"

    def test_to_information(self, tmp_path: Path):
        """to_information() should return partial_name dict."""
        lock_path = tmp_path / "holder.json"
        resource = MockResource()
        lock_file = PartialLockFile.create(
            lock_path, resource, "/job", information={"partial_name": "data"}
        )

        assert lock_file.to_information() == {"partial_name": "data"}

    def test_from_information_none(self, tmp_path: Path):
        """from_information(None) should set partial_name to empty string."""
        lock_path = tmp_path / "holder.json"
        resource = MockResource()
        lock_file = PartialLockFile.create(
            lock_path, resource, "/job", information=None
        )

        assert lock_file.partial_name == ""

    def test_from_information_invalid_raises(self, tmp_path: Path):
        """from_information with invalid format should raise."""
        lock_path = tmp_path / "holder.json"
        lock_path.write_text(json.dumps({"job_uri": "/job", "information": "invalid"}))
        resource = MockResource()

        with pytest.raises(ValueError, match="Invalid information format"):
            PartialLockFile(lock_path, resource)
