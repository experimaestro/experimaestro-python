#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["watchdog>=4.0"]
# ///
"""Cross-host filesystem change-notification benchmark.

Measures how quickly (and whether at all) a *server* host notices file
creations and appends made by a *client* host on a shared filesystem
(GPFS/Lustre/NFS).

It reproduces experimaestro's actual event-file workload:

  * many entity files, one per "job", in a flat directory
  * names of the form ev-{entity}-{count}.jsonl, mirroring the real
    {hash8}-{job_id}-{count}.jsonl scheme
  * rotation: after --rotate-every lines an entity starts a NEW file, so the
    watcher must discover files appearing mid-run
  * one JSON line per event, flush()+fsync() per line, matching
    scheduler/state_status.py:626

Two things are therefore measured separately:

  1. APPEND latency  -- how long until a line written elsewhere is readable
  2. DISCOVERY latency -- how long until a newly created file is noticed at all

Strategies (server side):

  wd-auto          watchdog default Observer (InotifyObserver on Linux)
  wd-polling       watchdog PollingObserver (experimaestro WatcherType.POLLING)
  stat-slow/fast   glob the dir, stat each file, open/seek/read/close on growth
  fd-fstat-*       hold an FD per file, poll os.fstat(fd), read from the FD
  fd-read-fast     hold an FD per file, just os.read() it, no stat at all
  kqueue-fd        register the held FDs with kqueue (macOS/BSD only)

The fd-* and kqueue-fd strategies are the TailedFilePool question: does a
long-lived FD ever see a remote appender's writes, or does client-side caching
pin it at a stale EOF? Note that NO fd-based strategy can discover a new file
on its own -- discovery always needs a directory scan or a directory event,
which is why the discovery table is reported separately.

MEASUREMENT HAZARD -- read before interpreting results
------------------------------------------------------
watchdog's inotify mask includes IN_OPEN and IN_CLOSE_NOWRITE (see
watchdog/observers/inotify_c.py). A *read-only* open therefore emits inotify
events. If a polling strategy runs in the same process, its local reads will
trigger the inotify observer, which then drains and appears to have detected
the remote write. To measure any wd-* strategy honestly, run it alone::

    uv run dev/fswatch_bench.py server $SCRATCH/fsbench --only wd-auto

kqueue is immune to this: NOTE_WRITE|NOTE_EXTEND fire only on modification, so
kqueue-fd stays honest even in a combined run.

Usage
-----
Start the SERVER first (it arms before anything is written)::

    uv run dev/fswatch_bench.py server $SCRATCH/fsbench

Then, on ANOTHER host, the CLIENT::

    uv run dev/fswatch_bench.py client $SCRATCH/fsbench

Both must point at the same shared directory with the same --tag. The script is
PEP 723 self-contained, so `uv run` fetches watchdog itself.

Clock skew: latencies are (server clock at detection) - (client clock at
write), so they include any NTP offset. The report prints the minimum observed
latency as a skew floor and warns if it goes negative.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

CONTROL = "control.json"
DONE = "client-done"
GLOB = "ev-*.jsonl"
CHUNK = 1 << 16


# =============================================================================
# Per-file state
# =============================================================================


@dataclass
class FileState:
    path: Path
    pos: int = 0
    buf: bytes = b""
    fd: int | None = None


@dataclass
class Hit:
    seq: int
    write_time: float
    detect_time: float

    @property
    def latency_ms(self) -> float:
        return (self.detect_time - self.write_time) * 1000.0


# =============================================================================
# Strategies
# =============================================================================


class Strategy:
    """Base: tracks many files, each with an independent read position."""

    name = "base"

    def __init__(self, directory: Path):
        self.dir = directory
        self.states: dict[Path, FileState] = {}
        self.hits: dict[int, Hit] = {}
        self.recovered: dict[int, Hit] = {}  # only seen by the final drain
        self.file_first_seen: dict[Path, float] = {}
        self.file_seen_live: set[Path] = set()  # excludes final-drain discoveries
        self.writer_hosts: set[str] = set()
        self.errors: list[str] = []
        self._lock = threading.RLock()
        self._final = False

    # -- to override --------------------------------------------------------

    def arm(self) -> None:
        """Start watching. Called before any file exists."""

    def close(self) -> None:
        """Release resources."""

    def _read_new(self, st: FileState) -> bytes:
        """Bytes appended to st.path since our last read (b'' if none)."""
        raise NotImplementedError

    # -- shared -------------------------------------------------------------

    def _glob(self) -> list[Path]:
        try:
            return sorted(self.dir.glob(GLOB))
        except OSError:
            return []

    def track(self, path: Path, when: float | None = None) -> FileState:
        """Register a file, recording when this strategy first saw it."""
        with self._lock:
            st = self.states.get(path)
            if st is None:
                st = FileState(path=path)
                self.states[path] = st
                self.file_first_seen[path] = when if when is not None else time.time()
                if not self._final:
                    self.file_seen_live.add(path)
            return st

    def discover(self) -> None:
        """Directory scan. fd-based strategies still need this for creation."""
        for p in self._glob():
            self.track(p)

    def drain_file(self, st: FileState) -> None:
        with self._lock:
            try:
                data = self._read_new(st)
            except FileNotFoundError:
                return
            except OSError as e:
                self.errors.append(repr(e))
                return
            if not data:
                return
            now = time.time()

            st.buf += data
            while b"\n" in st.buf:
                raw, st.buf = st.buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    seq, wt = int(rec["seq"]), float(rec["t"])
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
                host = rec.get("host")
                if host:
                    self.writer_hosts.add(str(host))
                if seq not in self.hits and seq not in self.recovered:
                    target = self.recovered if self._final else self.hits
                    target[seq] = Hit(seq=seq, write_time=wt, detect_time=now)

    def drain_all(self) -> None:
        with self._lock:
            states = list(self.states.values())
        for st in states:
            self.drain_file(st)

    def final_drain(self) -> None:
        """Post-run: proves data was readable even if never notified."""
        self._final = True
        self.discover()
        self.drain_all()


class _PollingStrategy(Strategy):
    """Rescans the directory and drains every tracked file on an interval."""

    def __init__(self, directory: Path, interval: float):
        super().__init__(directory)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def arm(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name=f"bench-{self.name}", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.discover()
            self.drain_all()
            self._stop.wait(self.interval)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


class StatReopen(_PollingStrategy):
    """stat() each path; on growth, open/seek/read/close.

    This is what experimaestro's PolledFile + non-pooled read path does today.
    """

    def __init__(self, directory: Path, interval: float, name: str = "stat-reopen"):
        super().__init__(directory, interval)
        self.name = name

    def _read_new(self, st: FileState) -> bytes:
        size = os.stat(st.path).st_size
        if size < st.pos:  # truncated / replaced
            st.pos, st.buf = 0, b""
        if size <= st.pos:
            return b""
        with open(st.path, "rb") as f:
            f.seek(st.pos)
            data = f.read(size - st.pos)
        st.pos += len(data)
        return data


class _HeldFd(_PollingStrategy):
    """Base for strategies keeping one long-lived FD per file."""

    def _ensure_fd(self, st: FileState) -> None:
        if st.fd is None:
            st.fd = os.open(st.path, os.O_RDONLY)  # raises if absent

    def close(self) -> None:
        super().close()
        with self._lock:
            for st in self.states.values():
                if st.fd is not None:
                    try:
                        os.close(st.fd)
                    except OSError:
                        pass
                    st.fd = None


class FdFstat(_HeldFd):
    """Hold an FD per file; poll os.fstat(fd); read from the held FD.

    The TailedFilePool model. If the shared-FS client pins a long-lived FD at a
    stale size, this is where it shows up.
    """

    def __init__(self, directory: Path, interval: float, name: str = "fd-fstat"):
        super().__init__(directory, interval)
        self.name = name

    def _read_new(self, st: FileState) -> bytes:
        self._ensure_fd(st)
        assert st.fd is not None
        size = os.fstat(st.fd).st_size
        if size < st.pos:
            os.lseek(st.fd, 0, os.SEEK_SET)
            st.pos, st.buf = 0, b""
        if size <= st.pos:
            return b""
        out = b""
        while st.pos < size:
            chunk = os.read(st.fd, min(CHUNK, size - st.pos))
            if not chunk:
                break
            out += chunk
            st.pos += len(chunk)
        return out


class FdRead(_HeldFd):
    """Hold an FD per file and just os.read() it -- no stat syscall at all."""

    def __init__(self, directory: Path, interval: float, name: str = "fd-read"):
        super().__init__(directory, interval)
        self.name = name

    def _read_new(self, st: FileState) -> bytes:
        self._ensure_fd(st)
        assert st.fd is not None
        out = b""
        while True:
            chunk = os.read(st.fd, CHUNK)
            if not chunk:
                break
            out += chunk
            st.pos += len(chunk)
        return out


class KqueueFd(FdFstat):
    """Register the ALREADY-OPEN FDs with kqueue (EVFILT_VNODE). macOS/BSD.

    This is the proposed experimaestro design. TailedFilePool already holds the
    file FDs, so kevent registration costs no extra descriptor beyond the kqueue
    object itself, and closing an FD deregisters it automatically.

    The watched DIRECTORY is also registered (NOTE_WRITE fires when entries are
    added), which is how rotation to a new ev-*-{n}.jsonl gets discovered
    without any polling.

    Immune to the read-contamination hazard: NOTE_WRITE|NOTE_EXTEND fire only on
    modification, so a co-running reader cannot trip them. `interval` is used
    only as a slow safety re-scan for entries kqueue might have coalesced.
    """

    def __init__(self, directory: Path, interval: float = 2.0, name: str = "kqueue-fd"):
        super().__init__(directory, interval, name=name)
        self.kq = None
        self.dir_fd: int | None = None
        self.file_events = 0
        self.dir_events = 0
        self._registered: set[Path] = set()

    def _kevent(self, fd: int):
        import select

        return select.kevent(
            fd,
            filter=select.KQ_FILTER_VNODE,
            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND,
        )

    def _register_new_files(self) -> None:
        """Open + register any file we are not yet watching."""
        for p in self._glob():
            if p in self._registered:
                continue
            st = self.track(p)
            try:
                self._ensure_fd(st)
            except OSError:
                continue
            assert st.fd is not None
            try:
                self.kq.control([self._kevent(st.fd)], 0, 0)
            except OSError as e:
                self.errors.append(repr(e))
                continue
            self._registered.add(p)
            self.drain_file(st)  # content written before we registered

    def _loop(self) -> None:
        import select

        self.kq = select.kqueue()

        while not self._stop.is_set():  # wait for the directory
            try:
                self.dir_fd = os.open(self.dir, os.O_RDONLY)
                break
            except OSError:
                self._stop.wait(0.25)
        if self._stop.is_set() or self.dir_fd is None:
            return

        self.kq.control([self._kevent(self.dir_fd)], 0, 0)
        self._register_new_files()

        while not self._stop.is_set():
            try:
                events = self.kq.control(None, 16, self.interval)
            except OSError as e:
                self.errors.append(repr(e))
                break

            fd_to_state = {
                st.fd: st for st in list(self.states.values()) if st.fd is not None
            }
            for ev in events:
                if ev.ident == self.dir_fd:
                    self.dir_events += 1
                    self._register_new_files()
                else:
                    self.file_events += 1
                    st = fd_to_state.get(ev.ident)
                    if st is not None:
                        self.drain_file(st)
            if not events:
                # Slow safety net; also catches a directory event we coalesced.
                self._register_new_files()

    def close(self) -> None:
        super().close()
        if self.dir_fd is not None:
            try:
                os.close(self.dir_fd)
            except OSError:
                pass
            self.dir_fd = None
        if self.kq is not None:
            try:
                self.kq.close()
            except OSError:
                pass
            self.kq = None


class WatchdogStrategy(Strategy):
    """Event-driven: drain whenever watchdog reports activity in the directory.

    Pure events -- no directory scan -- so it also measures whether creation of
    a remote file is noticed at all.
    """

    def __init__(self, directory: Path, kind: str, poll_interval: float):
        super().__init__(directory)
        self.name = kind
        self.kind = kind
        self.poll_interval = poll_interval
        self.observer = None
        self.event_count = 0
        self.observer_class = "n/a"

    def arm(self) -> None:
        from watchdog.events import FileSystemEventHandler

        if self.kind == "wd-polling":
            from watchdog.observers.polling import PollingObserver

            self.observer = PollingObserver(timeout=self.poll_interval)
        else:
            from watchdog.observers import Observer

            self.observer = Observer()
        self.observer_class = type(self.observer).__name__

        outer = self

        class _H(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if not p.name.startswith("ev-") or not p.name.endswith(".jsonl"):
                    return
                outer.event_count += 1
                st = outer.track(p)
                outer.drain_file(st)

        self.observer.schedule(_H(), str(self.dir), recursive=False)
        self.observer.start()

    def discover(self) -> None:
        # Event-driven by design: only scan during the final drain, so the
        # live numbers reflect what the event stream actually delivered.
        if self._final:
            super().discover()

    def _read_new(self, st: FileState) -> bytes:
        try:
            size = os.stat(st.path).st_size
        except FileNotFoundError:
            return b""
        if size <= st.pos:
            return b""
        with open(st.path, "rb") as f:
            f.seek(st.pos)
            data = f.read(size - st.pos)
        st.pos += len(data)
        return data

    def close(self) -> None:
        if self.observer is not None:
            try:
                self.observer.stop()
                self.observer.join(timeout=5)
            except Exception:
                pass
            self.observer = None


# =============================================================================
# Reporting
# =============================================================================


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def _fmt(v: float) -> str:
    return "        n/a" if v != v else f"{v:8.1f} ms"


@dataclass
class Truth:
    """Authoritative ground truth, read from disk after the run."""

    total_lines: int = 0
    file_first_write: dict[Path, float] = field(default_factory=dict)


def read_truth(session: Path) -> Truth:
    t = Truth()
    for p in sorted(session.glob(GLOB)):
        try:
            with p.open("rb") as f:
                for raw in f:
                    if not raw.endswith(b"\n"):
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t.total_lines += 1
                    if p not in t.file_first_write:
                        t.file_first_write[p] = float(rec["t"])
        except OSError:
            continue
    return t


def print_report(entries: list[tuple[Strategy, Truth]], extra: dict) -> None:
    strategies = [s for s, _ in entries]
    print()
    print("=" * 94)
    print("RESULTS")
    print("=" * 94)
    for k, v in extra.items():
        print(f"  {k}: {v}")

    hosts: set[str] = set()
    for s in strategies:
        hosts |= s.writer_hosts
    me = platform.node()
    print(f"  client host(s): {', '.join(sorted(hosts)) or 'unknown'}")

    if hosts and hosts <= {me}:
        print()
        print("  " + "!" * 70)
        print("  !! CLIENT RAN ON THE SAME HOST AS THE SERVER.")
        print("  !! This measures LOCAL filesystem behaviour only. inotify sees")
        print("  !! local VFS activity, so it will look fast here and still fail")
        print("  !! completely against a remote writer.")
        print("  " + "!" * 70)

    if extra.get("mode") == "combined":
        print()
        print("  " + "!" * 70)
        print("  !! COMBINED MODE -- strategies contaminated each other:")
        print("  !!  * a local read emits IN_OPEN/IN_CLOSE_NOWRITE, tripping the")
        print("  !!    inotify observer, so wd-auto may reflect LOCAL reads")
        print("  !!  * whichever strategy polls first warms the client cache for")
        print("  !!    all the others, so per-strategy latencies are NOT")
        print("  !!    independent measurements")
        print("  !! Drop --combined to run each strategy alone on fresh files.")
        print("  " + "!" * 70)

    # -- append latency ----------------------------------------------------
    lines = entries[0][1].total_lines if entries else 0
    print()
    print(f"APPEND latency  (ground truth: {lines} lines per phase)")
    hdr = (
        f"{'strategy':<20} {'live':>6} {'late':>6} {'never':>6} "
        f"{'median':>11} {'p90':>11} {'max':>11} {'err':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s, truth in entries:
        lats = [h.latency_ms for h in s.hits.values()]
        never = truth.total_lines - len(s.hits) - len(s.recovered)
        print(
            f"{s.name:<20} {len(s.hits):>6} {len(s.recovered):>6} {never:>6} "
            f"{_fmt(_pct(lats, 0.5))} {_fmt(_pct(lats, 0.9))} "
            f"{_fmt(max(lats) if lats else float('nan'))} {len(s.errors):>4}"
        )

    # -- discovery latency -------------------------------------------------
    files = len(entries[0][1].file_first_write) if entries else 0
    print()
    print(f"DISCOVERY latency  (ground truth: {files} files per phase)")
    hdr2 = (
        f"{'strategy':<20} {'found':>6} {'missed':>6} "
        f"{'median':>11} {'p90':>11} {'max':>11}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for s, truth in entries:
        dl = [
            (s.file_first_seen[p] - t0) * 1000.0
            for p, t0 in truth.file_first_write.items()
            if p in s.file_seen_live
        ]
        # A file discovered only by the final drain is not a live discovery.
        found = len(dl)
        print(
            f"{s.name:<20} {found:>6} {len(truth.file_first_write) - found:>6} "
            f"{_fmt(_pct(dl, 0.5))} {_fmt(_pct(dl, 0.9))} "
            f"{_fmt(max(dl) if dl else float('nan'))}"
        )

    print()
    print("  live  = detected during the run (counts toward latency stats)")
    print("  late  = invisible during the run, readable by the final drain")
    print("          -> the data arrived; the notification mechanism did not")
    print("  never = not present even at the end of the run")

    all_lats = [h.latency_ms for s in strategies for h in s.hits.values()]
    if all_lats:
        lo = min(all_lats)
        print(f"\n  minimum latency observed anywhere: {lo:.1f} ms (clock-skew floor)")
        if lo < -50:
            print(
                "  !! NEGATIVE -- the nodes' clocks disagree by at least "
                f"{-lo:.0f} ms. Subtract this offset from every number above."
            )

    print()
    for s in strategies:
        if isinstance(s, WatchdogStrategy):
            print(
                f"  {s.name}: observer={s.observer_class}, "
                f"raw callbacks={s.event_count}"
            )
        elif isinstance(s, KqueueFd):
            print(
                f"  {s.name}: file kevents={s.file_events}, "
                f"dir kevents={s.dir_events}, registered={len(s._registered)}"
            )
        if s.errors:
            print(f"  {s.name}: {len(s.errors)} errors, first={s.errors[0]}")


def write_csv(path: Path, strategies: list[Strategy]) -> None:
    with path.open("w") as f:
        f.write("strategy,seq,phase,write_time,detect_time,latency_ms\n")
        for s in strategies:
            for phase, d in (("live", s.hits), ("late", s.recovered)):
                for h in sorted(d.values(), key=lambda x: x.seq):
                    f.write(
                        f"{s.name},{h.seq},{phase},{h.write_time:.6f},"
                        f"{h.detect_time:.6f},{h.latency_ms:.3f}\n"
                    )


# =============================================================================
# Roles
# =============================================================================


def write_batch(args, pdir: Path) -> int:
    """Write one full batch of rotating entity files into pdir."""
    handles: dict[int, tuple[Path, object]] = {}
    written: dict[int, int] = {e: 0 for e in range(args.files)}
    created: list[str] = []

    try:
        for i in range(args.count):
            e = i % args.files
            rot = written[e] // args.rotate_every
            path = pdir / f"ev-{e:02d}-{rot}.jsonl"

            cur = handles.get(e)
            if cur is None or cur[0] != path:
                if cur is not None:
                    cur[1].close()
                # buffering=1, matching scheduler/state_status.py:572
                handles[e] = (path, open(path, "a", buffering=1))
                created.append(path.name)

            f = handles[e][1]
            rec = {
                "seq": i,
                "t": time.time(),
                "host": platform.node(),
                "entity": e,
                "file": path.name,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if not args.no_fsync:
                os.fsync(f.fileno())
            written[e] += 1

            if (i + 1) % 20 == 0 or i == 0:
                print(f"[client]   {i + 1}/{args.count}", flush=True)
            if i < args.count - 1:
                time.sleep(args.interval)
    finally:
        for _, h in handles.values():
            try:
                h.close()
            except OSError:
                pass

    (pdir / DONE).write_text(json.dumps({"count": args.count, "files": created}))
    return len(created)


def run_client(args) -> int:
    """The writer: follows the server's phase plan, one fresh batch per phase."""
    session = Path(args.path) / args.tag
    control = session / CONTROL

    print(f"[client] host={platform.node()} session={session}")
    print(f"[client] waiting for the server ({CONTROL}) ...", flush=True)

    deadline = time.time() + args.handshake_timeout
    last = -1

    while True:
        if time.time() > deadline:
            print(
                f"[client] ERROR: no phase from the server after "
                f"{args.handshake_timeout}s. Start the server first; check both "
                "hosts see the same path and --tag.",
                file=sys.stderr,
            )
            return 1
        try:
            ctl = json.loads(control.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(0.5)
            continue

        phase = int(ctl.get("phase", -1))
        if phase < 0:
            break
        if phase <= last:
            time.sleep(0.25)
            continue

        pdir = session / ctl["dir"]
        pdir.mkdir(parents=True, exist_ok=True)
        print(
            f"[client] phase {phase + 1}/{ctl.get('total', '?')} "
            f"[{ctl.get('label', '')}] -> {pdir.name}: {args.count} lines, "
            f"{args.files} entities, rotate/{args.rotate_every}, "
            f"{args.interval}s apart (fsync={not args.no_fsync})",
            flush=True,
        )
        n = write_batch(args, pdir)
        print(
            f"[client] phase {phase + 1} done: {args.count} lines in {n} files",
            flush=True,
        )
        last = phase
        deadline = time.time() + args.handshake_timeout

    print("[client] all phases complete")
    return 0


def run_server(args) -> int:
    """The watcher: arms every strategy, then reports detection latency."""
    session = Path(args.path) / args.tag
    if session.exists() and not args.keep:
        shutil.rmtree(session)
    session.mkdir(parents=True, exist_ok=True)
    control = session / CONTROL

    print(f"[server] host={platform.node()} session={session}")
    print(f"[server] platform={platform.platform()}")

    factories = {
        "wd-auto": lambda d: WatchdogStrategy(d, "wd-auto", args.poll),
        "wd-polling": lambda d: WatchdogStrategy(d, "wd-polling", args.poll),
        "stat-slow": lambda d: StatReopen(d, args.poll, f"stat-reopen@{args.poll}"),
        "stat-fast": lambda d: StatReopen(
            d, args.fast_poll, f"stat-reopen@{args.fast_poll}"
        ),
        "fd-fstat-slow": lambda d: FdFstat(d, args.poll, f"fd-fstat@{args.poll}"),
        "fd-fstat-fast": lambda d: FdFstat(
            d, args.fast_poll, f"fd-fstat@{args.fast_poll}"
        ),
        "fd-read-fast": lambda d: FdRead(
            d, args.fast_poll, f"fd-read@{args.fast_poll}"
        ),
        "kqueue-fd": lambda d: KqueueFd(d, args.kqueue_rescan),
    }

    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        unknown = [k for k in keys if k not in factories]
        if unknown:
            print(
                f"[server] ERROR: unknown strategy {unknown}. "
                f"Choose from: {', '.join(factories)}",
                file=sys.stderr,
            )
            return 2
    else:
        keys = list(factories)

    import select

    if "kqueue-fd" in keys and not hasattr(select, "kqueue"):
        if args.only:
            print(
                "[server] ERROR: kqueue is unavailable on this platform.",
                file=sys.stderr,
            )
            return 2
        keys.remove("kqueue-fd")
        print("[server] kqueue unavailable on this platform - skipping kqueue-fd")

    if any(k.startswith("wd-") for k in keys):
        try:
            import watchdog  # noqa: F401
        except ImportError:
            print("[server] watchdog not installed - skipping observer strategies")
            keys = [k for k in keys if not k.startswith("wd-")]

    # One phase per strategy by default: each runs ALONE, on its own fresh
    # directory, so neither inotify contamination nor client-cache warming
    # from a sibling strategy can leak into its numbers.
    phases = [keys] if args.combined else [[k] for k in keys]

    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())

    per_phase = args.count * args.interval + args.grace
    print(
        f"[server] {len(phases)} phase(s), ~{per_phase:.0f}s each "
        f"-> ~{len(phases) * per_phase / 60:.1f} min total"
    )

    entries: list[tuple[Strategy, Truth]] = []

    for idx, pkeys in enumerate(phases):
        label = ",".join(pkeys)
        suffix = pkeys[0] if len(pkeys) == 1 else "all"
        pdir = session / f"phase-{idx:02d}-{suffix}"
        if pdir.exists():
            shutil.rmtree(pdir)  # fresh inodes -> cold client cache
        pdir.mkdir(parents=True)

        strategies = [factories[k](pdir) for k in pkeys]
        for s in strategies:
            s.arm()
        print(
            f"\n[server] phase {idx + 1}/{len(phases)}: {label} -> {pdir.name}",
            flush=True,
        )

        control.write_text(
            json.dumps(
                {
                    "phase": idx,
                    "dir": pdir.name,
                    "label": label,
                    "total": len(phases),
                }
            )
        )

        done = pdir / DONE
        start = time.time()
        grace_until: float | None = None
        while not interrupted.is_set():
            if time.time() - start > args.timeout:
                print(f"[server] phase timeout after {args.timeout}s")
                break
            if grace_until is None and done.exists():
                grace_until = time.time() + args.grace
                print(
                    f"[server] client finished; draining {args.grace}s ...", flush=True
                )
            if grace_until is not None and time.time() > grace_until:
                break
            time.sleep(0.25)

        for s in strategies:
            s.final_drain()
        for s in strategies:
            s.close()

        truth = read_truth(pdir)
        entries += [(s, truth) for s in strategies]

        if interrupted.is_set():
            print("\n[server] interrupted - reporting what we have")
            break

    control.write_text(json.dumps({"phase": -1}))

    print_report(
        entries,
        {
            "server host": platform.node(),
            "session": str(session),
            "mode": "combined" if args.combined else "sequential (isolated)",
            "poll interval": f"{args.poll}s",
            "fast poll interval": f"{args.fast_poll}s",
        },
    )

    csv = Path(args.csv) if args.csv else session / "results.csv"
    write_csv(csv, [s for s, _ in entries])
    print(f"\n[server] per-line CSV: {csv}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cross-host filesystem notification benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start the server first, then the client on another host.",
    )
    p.add_argument(
        "role",
        choices=["server", "client"],
        help="server watches (start first); client writes",
    )
    p.add_argument("path", help="Shared directory visible from BOTH hosts")
    p.add_argument(
        "--tag", default="run1", help="Session subdirectory; must match on both hosts"
    )
    p.add_argument("--count", type=int, default=60, help="Total lines to write")
    p.add_argument(
        "--files", type=int, default=5, help="Concurrent entity files (simulated jobs)"
    )
    p.add_argument(
        "--rotate-every",
        type=int,
        default=10,
        help="Lines per entity before rotating to a new file",
    )
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between writes")
    p.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="Baseline poll interval (experimaestro min_poll_interval)",
    )
    p.add_argument(
        "--fast-poll",
        type=float,
        default=0.05,
        help="Fast poll interval, to probe the attribute-cache floor",
    )
    p.add_argument(
        "--kqueue-rescan",
        type=float,
        default=2.0,
        help="kqueue safety rescan interval (not the detection path)",
    )
    p.add_argument(
        "--no-fsync",
        action="store_true",
        help="Skip fsync (experimaestro does fsync; this tests without)",
    )
    p.add_argument("--timeout", type=float, default=900.0, help="Server hard timeout")
    p.add_argument(
        "--grace",
        type=float,
        default=30.0,
        help="Server drain window after the client finishes",
    )
    p.add_argument(
        "--handshake-timeout",
        type=float,
        default=300.0,
        help="Client wait for the server-ready marker",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Run every strategy at once in one directory (fast, but "
        "the strategies contaminate each other: local reads trip "
        "the inotify observer, and whichever polls first warms "
        "the client cache for the rest). Default is one phase "
        "per strategy on fresh files.",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated strategy keys to run in ISOLATION. "
        "Required to measure a wd-* strategy honestly: local "
        "reads emit IN_OPEN/IN_CLOSE_NOWRITE and trigger the "
        "inotify observer. Keys: wd-auto, wd-polling, stat-slow, "
        "stat-fast, fd-fstat-slow, fd-fstat-fast, fd-read-fast, "
        "kqueue-fd",
    )
    p.add_argument(
        "--keep", action="store_true", help="Server: do not wipe an existing session"
    )
    p.add_argument("--csv", default=None, help="Where to write per-line results")
    args = p.parse_args()

    return run_server(args) if args.role == "server" else run_client(args)


if __name__ == "__main__":
    sys.exit(main())
