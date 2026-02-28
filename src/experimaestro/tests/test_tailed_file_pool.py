"""Tests for TailedFilePool

Tests the bounded file descriptor pool for efficient file tailing.
"""

import threading
from pathlib import Path

import pytest

from experimaestro.filewatcher import TailedFilePool


@pytest.fixture
def pool():
    p = TailedFilePool(max_open=3)
    yield p
    p.close_all()


@pytest.fixture
def small_pool():
    """Pool with max_open=2 for testing eviction"""
    p = TailedFilePool(max_open=2)
    yield p
    p.close_all()


class TestBasicReading:
    def test_read_new_lines_basic(self, tmp_path, pool):
        """Write lines, read them back"""
        f = tmp_path / "test.jsonl"
        f.write_text("line1\nline2\nline3\n")

        lines = pool.read_new_lines(f)
        assert lines == ["line1", "line2", "line3"]

    def test_read_new_lines_incremental(self, tmp_path, pool):
        """Two writes, two reads, each gets only new content"""
        f = tmp_path / "test.jsonl"
        f.write_text("line1\nline2\n")

        lines1 = pool.read_new_lines(f)
        assert lines1 == ["line1", "line2"]

        with f.open("a") as fh:
            fh.write("line3\nline4\n")

        lines2 = pool.read_new_lines(f)
        assert lines2 == ["line3", "line4"]

    def test_incomplete_line_buffered(self, tmp_path, pool):
        """Line without \\n is not returned, position stays before it"""
        f = tmp_path / "test.jsonl"
        f.write_text("complete\nincomplete")

        lines = pool.read_new_lines(f)
        assert lines == ["complete"]

        # Reading again still doesn't return the incomplete line
        lines = pool.read_new_lines(f)
        assert lines == []

    def test_incomplete_line_completed(self, tmp_path, pool):
        """After \\n is appended, line is returned"""
        f = tmp_path / "test.jsonl"
        f.write_text("complete\nincomplete")

        lines = pool.read_new_lines(f)
        assert lines == ["complete"]

        # Append newline to complete the line
        with f.open("a") as fh:
            fh.write(" now complete\n")

        lines = pool.read_new_lines(f)
        assert lines == ["incomplete now complete"]

    def test_empty_file(self, tmp_path, pool):
        """Empty file returns no lines"""
        f = tmp_path / "test.jsonl"
        f.write_text("")

        lines = pool.read_new_lines(f)
        assert lines == []

    def test_empty_lines_preserved(self, tmp_path, pool):
        """Empty lines (just \\n) are returned as empty strings"""
        f = tmp_path / "test.jsonl"
        f.write_text("line1\n\nline3\n")

        lines = pool.read_new_lines(f)
        assert lines == ["line1", "", "line3"]


class TestFDLimit:
    def test_fd_limit_enforced(self, tmp_path, small_pool):
        """Open max_open+1 files, verify only max_open FDs are kept open"""
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.jsonl"
            f.write_text(f"line-{i}\n")
            files.append(f)

        for f in files:
            small_pool.read_new_lines(f)

        assert small_pool.open_count <= 2

    def test_lru_eviction(self, tmp_path, small_pool):
        """Least-recently-read file is evicted first"""
        f0 = tmp_path / "file0.jsonl"
        f1 = tmp_path / "file1.jsonl"
        f2 = tmp_path / "file2.jsonl"
        for f in [f0, f1, f2]:
            f.write_text("line\n")

        # Read f0, f1 (both in pool)
        small_pool.read_new_lines(f0)
        small_pool.read_new_lines(f1)
        assert small_pool.open_count == 2

        # Read f2 - should evict f0 (least recently read)
        small_pool.read_new_lines(f2)
        assert small_pool.open_count == 2

        # f0 should be evicted, f1 and f2 should be open
        with small_pool._lock:
            assert f0 not in small_pool._open_files
            assert f1 in small_pool._open_files
            assert f2 in small_pool._open_files

    def test_evicted_file_still_readable(self, tmp_path, small_pool):
        """Evicted file can still be read (open/seek/read/close)"""
        f0 = tmp_path / "file0.jsonl"
        f1 = tmp_path / "file1.jsonl"
        f2 = tmp_path / "file2.jsonl"
        for f in [f0, f1, f2]:
            f.write_text("initial\n")

        small_pool.read_new_lines(f0)
        small_pool.read_new_lines(f1)
        small_pool.read_new_lines(f2)  # Evicts f0

        # Write more to f0 and read it (one-shot, evicts f1)
        with f0.open("a") as fh:
            fh.write("after-eviction\n")

        lines = small_pool.read_new_lines(f0)
        assert lines == ["after-eviction"]

    def test_active_file_re_enters_pool(self, tmp_path, small_pool):
        """Reading an evicted file with room re-opens its FD"""
        f0 = tmp_path / "file0.jsonl"
        f1 = tmp_path / "file1.jsonl"
        f2 = tmp_path / "file2.jsonl"
        for f in [f0, f1, f2]:
            f.write_text("line\n")

        small_pool.read_new_lines(f0)
        small_pool.read_new_lines(f1)
        small_pool.read_new_lines(f2)  # Evicts f0

        with small_pool._lock:
            assert f0 not in small_pool._open_files

        # Read f0 again - should evict LRU (f1) and re-enter pool
        with f0.open("a") as fh:
            fh.write("new\n")
        small_pool.read_new_lines(f0)

        with small_pool._lock:
            assert f0 in small_pool._open_files


class TestEdgeCases:
    def test_file_truncation_resets_position(self, tmp_path, pool):
        """File shrinks -> position resets to 0"""
        f = tmp_path / "test.jsonl"
        f.write_text("long content here\n")

        pool.read_new_lines(f)
        assert pool.get_position(f) > 0

        # Truncate and write shorter content
        f.write_text("short\n")

        lines = pool.read_new_lines(f)
        assert lines == ["short"]

    def test_file_deletion_handled(self, tmp_path, pool):
        """Deleted file returns empty, is removed from pool"""
        f = tmp_path / "test.jsonl"
        f.write_text("line\n")

        pool.read_new_lines(f)
        f.unlink()

        lines = pool.read_new_lines(f)
        assert lines == []
        assert pool.open_count == 0

    def test_nonexistent_file(self, tmp_path, pool):
        """Reading a file that never existed returns empty"""
        f = tmp_path / "nonexistent.jsonl"
        lines = pool.read_new_lines(f)
        assert lines == []

    def test_remove_closes_fd(self, tmp_path, pool):
        """Explicit remove closes FD"""
        f = tmp_path / "test.jsonl"
        f.write_text("line\n")

        pool.read_new_lines(f)
        assert pool.open_count == 1

        pool.remove(f)
        assert pool.open_count == 0
        assert pool.get_position(f) == 0

    def test_close_all(self, tmp_path, pool):
        """close_all closes everything"""
        for i in range(3):
            f = tmp_path / f"file{i}.jsonl"
            f.write_text(f"line-{i}\n")
            pool.read_new_lines(f)

        assert pool.open_count == 3

        pool.close_all()
        assert pool.open_count == 0

    def test_set_position(self, tmp_path, pool):
        """set_position changes read offset"""
        f = tmp_path / "test.jsonl"
        f.write_text("line1\nline2\nline3\n")

        # Skip first line by setting position
        pool.set_position(f, len("line1\n"))

        lines = pool.read_new_lines(f)
        assert lines == ["line2", "line3"]

    def test_get_position_default(self, tmp_path, pool):
        """get_position returns 0 for unknown files"""
        f = tmp_path / "unknown.jsonl"
        assert pool.get_position(f) == 0


class TestConcurrency:
    def test_concurrent_access(self, tmp_path):
        """Multiple threads reading different files"""
        pool = TailedFilePool(max_open=4)
        errors: list[str] = []
        num_threads = 4
        lines_per_file = 50

        # Create files
        files = []
        for i in range(num_threads):
            f = tmp_path / f"file{i}.jsonl"
            content = "".join(f"thread{i}-line{j}\n" for j in range(lines_per_file))
            f.write_text(content)
            files.append(f)

        def reader(file_path: Path, thread_id: int):
            try:
                lines = pool.read_new_lines(file_path)
                if len(lines) != lines_per_file:
                    errors.append(
                        f"Thread {thread_id}: expected {lines_per_file} lines, "
                        f"got {len(lines)}"
                    )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [
            threading.Thread(target=reader, args=(files[i], i))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent access errors: {errors}"
        pool.close_all()
