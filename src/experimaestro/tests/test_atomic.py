"""Tests for atomic file writes (experimaestro.utils.atomic)"""

import errno
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from experimaestro.utils import atomic
from experimaestro.utils.atomic import (
    atomic_create,
    atomic_open,
    atomic_write,
    atomic_write_json,
    read_json,
)


def _temporary_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.name.endswith(".tmp")]


def test_atomic_write_replaces_content(tmp_path: Path):
    path = tmp_path / "a.json"
    atomic_write_json(path, {"x": 1})
    atomic_write_json(path, {"x": 2}, fsync=True)
    assert json.loads(path.read_text()) == {"x": 2}

    atomic_write(path, b"bytes")
    assert path.read_bytes() == b"bytes"
    assert _temporary_files(tmp_path) == []


def test_atomic_open_failure_keeps_previous_content(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic_open(path) as fp:
            fp.write("new")
            raise RuntimeError("failure")
    assert path.read_text() == "old"
    assert _temporary_files(tmp_path) == []


def test_atomic_write_keeps_permissions(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("old")
    path.chmod(0o640)
    atomic_write(path, "new")
    assert path.stat().st_mode & 0o777 == 0o640


def test_atomic_write_applies_umask(tmp_path: Path):
    path = tmp_path / "a.txt"
    old_umask = os.umask(0o022)
    try:
        atomic_write(path, "new")
    finally:
        os.umask(old_umask)
    assert path.stat().st_mode & 0o777 == 0o644


def test_atomic_create(tmp_path: Path):
    path = tmp_path / "a.txt"
    assert atomic_create(path, "first")
    assert not atomic_create(path, "second")
    assert path.read_text() == "first"
    assert _temporary_files(tmp_path) == []


def test_atomic_create_without_hardlinks(tmp_path: Path, monkeypatch):
    def no_link(src, dst):
        raise OSError(errno.EPERM, "no hard links")

    monkeypatch.setattr(atomic.os, "link", no_link)
    path = tmp_path / "a.txt"
    assert atomic_create(path, "first")
    assert not atomic_create(path, "second")
    assert path.read_text() == "first"
    assert _temporary_files(tmp_path) == []


def test_read_json(tmp_path: Path):
    path = tmp_path / "a.json"
    assert read_json(path, default=None) is None
    with pytest.raises(FileNotFoundError):
        read_json(path)

    path.write_text("")
    assert read_json(path, default={}, delay=0) == {}
    with pytest.raises(json.JSONDecodeError):
        read_json(path, delay=0)


def test_read_json_retries_on_estale(tmp_path: Path, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text('{"x": 1}')
    calls = []
    read_text = Path.read_text

    def stale_once(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            raise OSError(errno.ESTALE, "stale file handle")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", stale_once)
    assert read_json(path, delay=0) == {"x": 1}
    assert len(calls) == 2


def _write_many(path: str, value: int, count: int):
    for _ in range(count):
        atomic_write_json(path, {"value": value, "padding": "x" * 10_000})


def test_concurrent_writers_never_expose_partial_files(tmp_path: Path):
    """Readers only see complete files while several processes write"""
    path = tmp_path / "shared.json"
    atomic_write_json(path, {"value": -1})

    ctx = multiprocessing.get_context("spawn")
    writers = [
        ctx.Process(target=_write_many, args=(str(path), i, 200)) for i in range(3)
    ]
    for writer in writers:
        writer.start()
    while any(writer.is_alive() for writer in writers):
        # Raises if the file is empty or partially written
        json.loads(path.read_text())
    for writer in writers:
        writer.join()
        assert writer.exitcode == 0

    assert _temporary_files(tmp_path) == []
