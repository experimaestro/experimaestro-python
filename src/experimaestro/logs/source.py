"""Where the bytes of a log file are read from

A :class:`LogSource` gives the size and the content of a log file, whether it
is local (:class:`LocalLogSource`) or on the remote host of an SSH workspace:
read by blocks through the remote server (:class:`RemoteLogSource`), or copied
with rsync when the remote server is too old to send them
(:class:`RsyncLogSource`). The log readers (``logs/reader.py``), used by
the TUI and the web UI, work on any of them.

Reading may wait for the network: the UIs read outside of their event loop.

A log file can be replaced while it is read (a restarted job rotates or
truncates its logs): each source tells it with its :attr:`~LogSource.generation`,
which changes when the file is not the one read before (other inode, or other
start).

Remote sources check the size of the file from time to time (the *probe*),
less and less often while it does not change (see :class:`ProbeDelay`).

To see how the UIs behave on a slow network, the ``XPM_LOG_READ_DELAY``
environment variable slows down each read of the log files by that many
seconds.
"""

import base64
import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("xpm.logs")

#: Size of the start of a file used to tell it apart from a file rewritten in
#: place (must match the remote server's ``READ_FILE_HEAD_SIZE``)
HEAD_SIZE = 4096
#: Size of the groups of filesystem blocks fetched together by
#: :class:`RemoteLogSource` (rounded to a multiple of the block size)
GROUP_SIZE = 64 * 1024
#: Groups fetched before and after the ones read (read-ahead)
READ_AHEAD_GROUPS = 1
#: Delay added to each read (seconds), to try the UIs on a slow network
READ_DELAY = float(os.environ.get("XPM_LOG_READ_DELAY") or 0)


def _slow_down() -> None:
    if READ_DELAY:
        time.sleep(READ_DELAY)


class ProbeDelay:
    """Delay between two probes of a remote file: it grows while the file does
    not change, and is reset when it changes or is read by the user"""

    MIN = 1.0
    MAX = 30.0
    FACTOR = 1.5

    def __init__(self):
        self.delay = self.MIN
        #: When the last probe was made (time.monotonic)
        self.last: float | None = None

    def due(self) -> bool:
        return self.last is None or time.monotonic() - self.last >= self.delay

    def probed(self, changed: bool) -> None:
        self.last = time.monotonic()
        self.delay = self.MIN if changed else min(self.delay * self.FACTOR, self.MAX)

    def reset(self) -> None:
        """Probe again as soon as possible"""
        self.delay = self.MIN
        self.last = None


class LogSource(ABC):
    """The bytes of a log file"""

    #: Changes when the file was replaced: what was read before is stale
    generation: int

    @abstractmethod
    def size(self) -> int | None:
        """Size of the file (None when it does not exist), checked (probed)
        when due"""
        ...

    @abstractmethod
    def read(self, offset: int, length: int) -> bytes:
        """Read a range of the file (shorter at its end)"""
        ...

    def touch(self) -> None:
        """The user reads the log (opened, scrolled): check it again soon"""


class _Identity:
    """Tells whether a file is still the one read before: same inode, and same
    start (the first bytes read before are unchanged)"""

    def __init__(self):
        self.inode: int | None = None
        #: Length and hash of the start of the file known
        self.head: tuple[int, str] | None = None

    @property
    def fingerprint_length(self) -> int:
        return 0 if self.head is None else self.head[0]

    def update(
        self, inode: int, fingerprint: str | None, head: tuple[int, str]
    ) -> bool:
        """Record the identity of the file; returns whether it changed

        ``fingerprint`` is the hash of the first :attr:`fingerprint_length`
        bytes of the file (None when shorter), ``head`` the length and hash of
        its start now.
        """
        changed = self.inode is not None and (
            inode != self.inode
            or (self.head is not None and fingerprint != self.head[1])
        )
        self.inode = inode
        self.head = head if head[0] else None
        return changed

    def clear(self) -> None:
        self.inode = None
        self.head = None


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalLogSource(LogSource):
    """A log file of this host"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.generation = 0
        self._identity = _Identity()

    def size(self) -> int | None:
        _slow_down()
        try:
            with self.path.open("rb") as f:
                stat = os.fstat(f.fileno())
                head = f.read(HEAD_SIZE)
        except FileNotFoundError:
            if self._identity.inode is not None:
                self._replaced()
            return None
        length = self._identity.fingerprint_length
        fingerprint = _hash(head[:length]) if len(head) >= length else None
        if self._identity.update(stat.st_ino, fingerprint, (len(head), _hash(head))):
            self.generation += 1
        return stat.st_size

    def _replaced(self) -> None:
        self._identity.clear()
        self.generation += 1

    def read(self, offset: int, length: int) -> bytes:
        _slow_down()
        try:
            with self.path.open("rb") as f:
                f.seek(offset)
                return f.read(length)
        except FileNotFoundError:
            return b""


class RsyncLogSource(LocalLogSource):
    """A remote log file copied (with its directory) by rsync

    Used when the remote server cannot send files by blocks (older
    experimaestro). The copy is made when probing.
    """

    def __init__(self, path: Path | str, sync: Callable[[], object]):
        super().__init__(path)
        self._sync = sync
        self._probe = ProbeDelay()
        self._size: int | None = None
        self._lock = threading.Lock()

    def size(self) -> int | None:
        with self._lock:
            if self._probe.due():
                self._sync()
                generation = self.generation
                size = super().size()
                self._probe.probed(size != self._size or generation != self.generation)
                self._size = size
            return self._size

    def touch(self) -> None:
        self._probe.reset()


class FileReader(Protocol):
    """Reads files of the remote workspace (see the ``read_file`` RPC)"""

    def read_file(
        self,
        path: str,
        offset: int,
        length: int,
        fingerprint_length: int,
        unless_size: int | None,
    ) -> dict: ...


class RemoteLogSource(LogSource):
    """A remote log file read by blocks

    The blocks read are written in a sparse local file (the parts not read
    take no disk space) and fetched only once: the file is read in groups of
    filesystem blocks (about :data:`GROUP_SIZE` bytes), with :data:`READ_AHEAD_GROUPS`
    groups before and after the ones needed. The last group of the file
    (still being written) is valid up to the size at which it was read, and
    read again (by the probe) when the file grows.
    """

    def __init__(self, remote: FileReader, remote_path: str, local_path: Path):
        self.remote = remote
        self.remote_path = remote_path
        self.local_path = local_path
        self.generation = 0
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Groups of blocks of the filesystem holding the local file, so that
        # the parts not read are holes
        # (f_frsize: the allocation unit, f_bsize being the preferred I/O size)
        block = max(os.statvfs(local_path.parent).f_frsize, 512)
        self.group_size = block * max(GROUP_SIZE // block, 1)
        local_path.write_bytes(b"")

        self._lock = threading.Lock()
        self._identity = _Identity()
        self._probe = ProbeDelay()
        self._size: int | None = None
        #: Complete groups held locally
        self._groups: set[int] = set()
        #: The last (incomplete) group held locally, and up to where
        self._tail: tuple[int, int] | None = None

    def _request(self, offset: int, length: int, unless_size: int | None = None):
        """Ask the remote server for a range of the file, and record what it
        sends"""
        _slow_down()
        answer = self.remote.read_file(
            self.remote_path,
            offset,
            length,
            self._identity.fingerprint_length,
            unless_size,
        )
        if not answer["exists"]:
            if self._size is not None:
                self._replaced()
            self._size = None
            return

        if self._identity.update(
            answer["inode"],
            answer["fingerprint"],
            (answer["head_length"], answer["head"]),
        ):
            logger.info("%s was replaced: reading it again", self.remote_path)
            self._replaced()
        self._size = answer["size"]
        data = base64.b64decode(answer["data"])

        with self.local_path.open("r+b") as f:
            if os.fstat(f.fileno()).st_size < self._size:
                # Extends the file with a hole
                f.truncate(self._size)
            if data:
                os.pwrite(f.fileno(), data, offset)
        end = offset + len(data)
        for group in range(offset // self.group_size, -(-end // self.group_size)):
            if (group + 1) * self.group_size <= end:
                self._groups.add(group)
        if data and end == self._size and end % self.group_size:
            self._tail = (end // self.group_size, end)

    def _replaced(self) -> None:
        self.generation += 1
        self._identity.clear()
        self._groups.clear()
        self._tail = None
        self.local_path.write_bytes(b"")

    def size(self) -> int | None:
        with self._lock:
            if self._probe.due():
                size, generation = self._size, self.generation
                if self._tail is None:
                    self._request(0, 0)
                else:
                    # The new content of the file comes with the answer
                    group = self._tail[0]
                    self._request(
                        group * self.group_size,
                        (1 + READ_AHEAD_GROUPS) * self.group_size,
                        unless_size=self._size,
                    )
                self._probe.probed(size != self._size or generation != self.generation)
            return self._size

    def _available(self, group: int, end: int) -> bool:
        """Whether the bytes of a group (up to ``end``) are held locally"""
        if group in self._groups:
            return True
        return (
            self._tail is not None
            and self._tail[0] == group
            and min(end, (group + 1) * self.group_size) <= self._tail[1]
        )

    def read(self, offset: int, length: int) -> bytes:
        with self._lock:
            while True:
                if self._size is None:
                    return b""
                end = min(offset + length, self._size)
                if end <= offset:
                    return b""
                needed = range(
                    offset // self.group_size, (end - 1) // self.group_size + 1
                )
                if all(self._available(group, end) for group in needed):
                    break
                last = (self._size - 1) // self.group_size
                window = range(
                    max(needed.start - READ_AHEAD_GROUPS, 0),
                    min(needed.stop + READ_AHEAD_GROUPS, last + 1),
                )
                missing = [g for g in window if not self._available(g, self._size)]
                start = missing[0] * self.group_size
                held = (len(self._groups), self._tail, self.generation)
                self._request(start, (missing[-1] + 1) * self.group_size - start)
                if held == (len(self._groups), self._tail, self.generation):
                    # Nothing new (the file changed meanwhile): read again
                    # what is held
                    break
            with self.local_path.open("rb") as f:
                return os.pread(f.fileno(), end - offset, offset)

    def touch(self) -> None:
        self._probe.reset()
