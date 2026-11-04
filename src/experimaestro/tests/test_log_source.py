"""Log sources: remote logs read by blocks, files replaced while read"""

import base64
import json
import os
import socket

import psutil
import pytest

from experimaestro.logs.reader import LogFile
from experimaestro.logs.source import (
    REMOTE_LINE_SEARCH,
    LocalLogSource,
    ProbeDelay,
    RemoteLogSource,
    ServerLogSource,
)
from experimaestro.scheduler.remote.cache_dir import (
    MARKER,
    PREFIX,
    remove_stale_cache_dirs,
)
from experimaestro.scheduler.remote.server import SSHStateProviderServer


class Remote:
    """Reads the files of a workspace as the remote server does, counting the
    requests and the bytes sent"""

    def __init__(self, workspace):
        self.server = SSHStateProviderServer(workspace)
        self.requests: list[tuple[int, int]] = []
        self.sent = 0

    def read_file(self, path, offset, length, fingerprint_length, unless_size):
        answer = self.server._handle_read_file(
            {
                "path": path,
                "offset": offset,
                "length": length,
                "fingerprint_length": fingerprint_length,
                "unless_size": unless_size,
            }
        )
        if answer.get("data"):
            # Fetched blocks (not the probes of an unchanged file)
            self.requests.append((offset, length))
        self.sent += len(answer.get("data", ""))
        return answer


@pytest.fixture
def remote(tmp_path):
    (tmp_path / "remote" / "job").mkdir(parents=True)
    return Remote(tmp_path / "remote")


@pytest.fixture(autouse=True)
def probe_always(monkeypatch):
    """Probe at each read (no waiting between probes)"""
    monkeypatch.setattr(ProbeDelay, "due", lambda self: True)


def source(remote, tmp_path) -> RemoteLogSource:
    return RemoteLogSource(remote, "job/task.err", tmp_path / "cache" / "task.err")


def write(remote, text: str, mode="w"):
    with open(remote.server.workspace_path / "job" / "task.err", mode) as f:
        f.write(text)


LINES = [f"line {i:05d} {'x' * 50}" for i in range(10_000)]


def test_remote_log_read_by_blocks(remote, tmp_path):
    """Only the blocks read are fetched, in a sparse local file; the whole log
    can then be read back"""
    write(remote, "".join(f"{line}\n" for line in LINES))
    log = source(remote, tmp_path)
    reader = LogFile(log)
    lines, _ = reader.read_tail()
    assert lines == LINES[-len(lines) :]
    # The end of the log (and the read-ahead) only
    assert remote.sent < 4 * 128 * 1024
    while earlier := reader.read_before():
        lines = earlier + lines
    assert lines == LINES
    # Each block fetched once
    fetched = sorted(remote.requests)
    for (offset, length), (next_offset, _) in zip(fetched, fetched[1:]):
        assert offset + length <= next_offset


def test_remote_log_sparse(remote, tmp_path):
    """The parts of the log not read take no disk space (a log of 40 MiB:
    APFS fills the holes smaller than 16 MiB)"""
    line = "x" * 1023 + "\n"
    write(remote, line * (40 * 1024))
    log = source(remote, tmp_path)
    LogFile(log).read_tail()
    stat = log.local_path.stat()
    assert stat.st_size == 40 * 1024 * 1024
    assert stat.st_blocks * 512 < 1024 * 1024


def test_remote_log_follows_growth(remote, tmp_path):
    write(remote, "start\n\rprogress 10%")
    reader = LogFile(source(remote, tmp_path))
    assert reader.read_tail() == (["start"], "progress 10%")
    write(remote, "\rprogress 20%\ndone\n", mode="a")
    assert reader.read_new_content() == (["progress 20%", "done"], "")


@pytest.mark.parametrize("rotate", [False, True], ids=["truncated", "rotated"])
def test_remote_log_replaced(remote, tmp_path, rotate):
    """A log rewritten (restarted job) is read again, even when it is longer
    than the first one when checked"""
    write(remote, "first run\n")
    log = source(remote, tmp_path)
    reader = LogFile(log)
    assert reader.read_tail() == (["first run"], "")
    generation = log.generation
    if rotate:
        path = remote.server.workspace_path / "job" / "task.err"
        path.rename(path.with_suffix(".old.err"))
    write(remote, "second run, writing a longer log\n")
    assert reader.read_new_content() == (["second run, writing a longer log"], "")
    assert log.generation != generation


def test_local_log_rewritten_in_place(tmp_path):
    path = tmp_path / "task.err"
    path.write_text("first run\n")
    reader = LogFile(LocalLogSource(path))
    assert reader.read_tail() == (["first run"], "")
    with path.open("r+") as f:
        f.write("second run, writing a longer log\n")
    assert reader.read_new_content() == (["second run, writing a longer log"], "")


def test_read_file_stays_in_workspace(remote):
    for path in ("../outside", "/etc/passwd"):
        with pytest.raises(PermissionError):
            remote.read_file(path, 0, 10, 0, None)


def test_probe_delay_grows_while_unchanged():
    delay = ProbeDelay()
    for _ in range(20):
        delay.probed(changed=False)
    assert delay.delay == ProbeDelay.MAX
    delay.probed(changed=True)
    assert delay.delay == ProbeDelay.MIN


def test_stale_cache_dirs_removed(tmp_path):
    """Only the marked cache directories of the monitors gone are removed"""

    def cache(name: str, pid: int | None, host=socket.gethostname()):
        path = tmp_path / f"{PREFIX}{name}"
        path.mkdir()
        if pid is not None:
            (path / MARKER).write_text(json.dumps({"pid": pid, "host": host}))
        return path

    # A process id that is not running
    gone = max(psutil.pids()) + 1
    stale = cache("gone", gone)
    kept = [
        cache("running", os.getppid()),
        cache("unmarked", None),
        cache("other-host", gone, host="another-host"),
    ]
    (tmp_path / f"{PREFIX}file").write_text("not a directory")

    assert remove_stale_cache_dirs(tmp_path) == [stale]
    assert all(path.is_dir() for path in kept)


def test_read_file_absolute_path(tmp_path):
    """An absolute path in the workspace is read, also when the workspace is
    given through a symbolic link"""
    (tmp_path / "real" / "job").mkdir(parents=True)
    (tmp_path / "real" / "job" / "task.err").write_text("hello\n")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    for workspace in ("real", "link"):
        remote = Remote(tmp_path / workspace)
        for other in ("real", "link"):
            path = str(tmp_path / other / "job" / "task.err")
            answer = remote.read_file(path, 0, 100, 0, None)
            assert base64.b64decode(answer["data"]) == b"hello\n"


class Renderer:
    """Renders the logs of a workspace as the remote server does, counting the
    bytes sent"""

    def __init__(self, workspace):
        self.server = SSHStateProviderServer(workspace)
        self.requests = 0
        self.sent = 0

    def read_log(self, path, offset, before, generation):
        answer = self.server._handle_read_log(
            {"path": path, "offset": offset, "before": before, "generation": generation}
        )
        self.requests += 1
        self.sent += len(json.dumps(answer))
        return answer


def progress_log(updates: int) -> tuple[list[str], str, str]:
    """Lines, then a progress bar rewritten ``updates`` times (no new line);
    returns the lines, the last state of the bar and the log"""
    lines = [f"INFO line {i:05d} {'x' * 50}" for i in range(5_000)]

    def bar(i: int) -> str:
        filled = 10 * i // updates
        return f"Indexing: {i:7d}|{'█' * filled}{' ' * (10 - filled)}| docs/s"

    text = "".join(f"{line}\n" for line in lines)
    text += "".join(f"\r{bar(i)}" for i in range(updates))
    return lines, bar(updates - 1), text


def read_whole(reader: LogFile) -> tuple[list[str], str]:
    lines, partial = reader.read_tail()
    while earlier := reader.read_before():
        lines = earlier + lines
    return lines, partial


def test_server_rendered_log(tmp_path):
    """A log rendered by the remote server: the lines before a long progress
    bar, and the bar as last drawn, without sending the bar's rewrites"""
    (tmp_path / "job").mkdir()
    lines, bar, text = progress_log(100_000)
    (tmp_path / "job" / "task.err").write_text(text)
    renderer = Renderer(tmp_path)
    reader = LogFile(ServerLogSource(renderer, "job/task.err"))

    shown, partial = reader.read_tail()
    assert partial == bar
    assert shown and shown == lines[-len(shown) :]
    assert renderer.sent < 2 * 128 * 1024 < len(text.encode()) // 10
    while earlier := reader.read_before():
        shown = earlier + shown
    assert shown == lines

    # Following: the new content only
    with open(tmp_path / "job" / "task.err", "a") as f:
        f.write("\ndone\n")
    assert reader.read_new_content() == ([bar, "done"], "")


def test_server_rendered_log_not_asked_while_probe_not_due(tmp_path, monkeypatch):
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "task.err").write_text("a\n")
    renderer = Renderer(tmp_path)
    reader = LogFile(ServerLogSource(renderer, "job/task.err"))
    assert reader.read_tail() == (["a"], "")
    monkeypatch.setattr(ProbeDelay, "due", lambda self: False)
    requests = renderer.requests
    assert reader.read_new_content() == ([], "")
    assert renderer.requests == requests


def test_remote_log_progress_bar_by_blocks(remote, tmp_path):
    """Older servers (logs read by blocks): the lines before a progress bar
    are found while it is not too long"""
    lines, bar, text = progress_log(20_000)
    assert len(text.encode()) - 5_000 * 67 < REMOTE_LINE_SEARCH
    write(remote, text)
    assert read_whole(LogFile(source(remote, tmp_path))) == (lines, bar)


def test_remote_log_never_returns_unfetched_bytes(remote, tmp_path):
    """The server may report a size beyond the data it sends (NFS): the part
    not fetched is not read (as zeros)"""
    write(remote, "abc\n")
    log = source(remote, tmp_path)
    read_file = remote.read_file

    def short_read(*args):
        answer = read_file(*args)
        if answer["exists"]:
            answer["size"] += 100
        return answer

    remote.read_file = short_read
    size = log.size()
    assert size == 104
    assert log.read(0, size) == b"abc\n"
