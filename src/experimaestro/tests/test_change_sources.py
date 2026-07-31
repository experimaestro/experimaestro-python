"""Tests for the ChangeSource abstraction.

Rather than depending on how a particular OS notification backend behaves on
whatever machine CI happens to run on, these tests inject *simulated* sources
into DirectoryWatch and reproduce the behaviours actually measured against real
filesystems:

  BlindSource       never reports anything -- inotify watching a Lustre/GPFS
                    directory written to by another host, which delivers no
                    events at all for remote creations or appends
  CoalescingSource  reports one event per N changes -- macOS FSEvents, which
                    merges rapid appends into a single notification
  PerfectSource     reports every change immediately -- kqueue on a local FS
  CreateOnlySource  reports creations but never modifications -- FSEvents
                    again, whose two channels fail independently

The invariant under test is the one the whole design rests on: whatever the
event source does, the PollSource backstop keeps the watch correct, and the
measured reliability moves in the right direction so polling adapts.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from experimaestro.filewatcher import (
    AsyncEventBridge,
    ChangeSource,
    FileWatcherService,
    KqueueSource,
    PollSource,
    WatchdogSource,
)


@pytest.fixture(autouse=True)
def reset_service():
    FileWatcherService.reset()
    AsyncEventBridge.reset()
    FileWatcherService.configure(testing_mode=True, polling_interval=0.01)
    yield
    FileWatcherService.reset()
    AsyncEventBridge.reset()


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# =============================================================================
# Simulated sources
# =============================================================================


class ScriptedSource(ChangeSource):
    """An event source the test drives by hand."""

    name = "scripted"

    def __init__(self, watch):
        super().__init__(watch)
        self.started = False
        self.armed: list[Path] = []
        self.disarmed: list[Path] = []

    def start(self) -> None:
        self.started = True

    def add_file(self, path: Path) -> None:
        self.armed.append(path)

    def remove_file(self, path: Path) -> None:
        self.disarmed.append(path)

    # -- what the OS would call --

    def emit_change(self, path: Path) -> None:
        watch = self._watch
        if watch is not None:
            watch._source_changed(self, path)

    def emit_created(self, path: Path) -> None:
        watch = self._watch
        if watch is not None:
            watch._source_created(self, path)

    def emit_deleted(self, path: Path) -> None:
        watch = self._watch
        if watch is not None:
            watch._source_deleted(self, path)


class BlindSource(ScriptedSource):
    """Reports nothing, ever (inotify against a remote writer on Lustre)."""

    name = "blind"

    def emit_change(self, path: Path) -> None:
        pass

    def emit_created(self, path: Path) -> None:
        pass


class CoalescingSource(ScriptedSource):
    """Reports only every Nth change (FSEvents merging rapid appends)."""

    name = "coalescing"

    def __init__(self, watch, every: int = 3):
        super().__init__(watch)
        self.every = every
        self.seen = 0

    def emit_change(self, path: Path) -> None:
        self.seen += 1
        if self.seen % self.every == 0:
            super().emit_change(path)


class CreateOnlySource(ScriptedSource):
    """Reports creations but never modifications (the FSEvents asymmetry)."""

    name = "create-only"

    def emit_change(self, path: Path) -> None:
        pass


def factory_for(cls, captured: list, **kwargs):
    """Build a source factory that records the instance it creates."""

    def factory(watch):
        source = cls(watch, **kwargs)
        captured.append(source)
        return source

    return factory


def make_watch(
    tmp_path: Path,
    captured: list,
    cls=ScriptedSource,
    max_poll: float = 5.0,
    **kwargs,
):
    """A watch whose only event source is a simulated one.

    `max_poll` matters for any test that waits on the backstop *after* the
    event source has fired: a successful report raises reliability, which
    legitimately stretches the poll interval toward max_poll_interval. Tests
    that assert "polling still catches it" must cap that, or they are really
    asserting how long they were willing to wait.
    """
    changes: list[Path] = []
    creations: list[Path] = []
    deletions: list[Path] = []
    svc = FileWatcherService.instance()
    watch = svc.watch_directory(
        tmp_path,
        on_change=changes.append,
        on_created=creations.append,
        on_deleted=deletions.append,
        min_poll_interval=0.05,
        max_poll_interval=max_poll,
        source_factories=[factory_for(cls, captured, **kwargs)],
    )
    return watch, changes, creations, deletions


# =============================================================================
# Wiring
# =============================================================================


class TestSourceWiring:
    def test_poll_source_is_always_present(self, tmp_path):
        """Even with no event sources at all, the backstop is installed."""
        svc = FileWatcherService.instance()
        with svc.watch_directory(tmp_path, source_factories=[]) as watch:
            assert [type(s) for s in watch._sources] == [PollSource]
            assert watch._poll_source.is_event_source is False

    def test_defaults_are_backstop_plus_one_event_source(self, tmp_path):
        """kqueue where available, watchdog otherwise -- never both."""
        svc = FileWatcherService.instance()
        with svc.watch_directory(tmp_path) as watch:
            assert isinstance(watch._sources[0], PollSource)
            event_sources = [s for s in watch._sources if s.is_event_source]
            assert len(event_sources) == 1

            expected = KqueueSource if KqueueSource.available(svc) else WatchdogSource
            assert isinstance(event_sources[0], expected)

    def test_injected_source_replaces_watchdog(self, tmp_path):
        captured: list[ScriptedSource] = []
        watch, *_ = make_watch(tmp_path, captured)
        with watch:
            assert len(captured) == 1
            assert captured[0].started
            assert not any(isinstance(s, WatchdogSource) for s in watch._sources)

    def test_add_file_arms_every_source(self, tmp_path):
        captured: list[ScriptedSource] = []
        watch, *_ = make_watch(tmp_path, captured)
        with watch:
            target = tmp_path / "a.txt"
            target.write_text("x")
            watch.add_file(target)

            assert captured[0].armed == [target]
            assert target in watch._files  # the poll source armed it too

            watch.remove_file(target)
            assert captured[0].disarmed == [target]
            assert target not in watch._files


# =============================================================================
# Behaviour under each simulated backend
# =============================================================================


class TestBlindSource:
    """Lustre/GPFS: the event source never fires, polling carries the watch."""

    def test_appends_still_detected(self, tmp_path):
        captured: list[BlindSource] = []
        watch, changes, _, _ = make_watch(tmp_path, captured, cls=BlindSource)
        with watch:
            target = tmp_path / "events.jsonl"
            target.write_text("one\n")
            watch.add_file(target)

            target.write_text("one\ntwo\n")
            captured[0].emit_change(target)  # blind: does nothing

            assert wait_for(lambda: target in changes), (
                "poll backstop must detect the append the event source missed"
            )

    def test_creations_still_detected(self, tmp_path):
        captured: list[BlindSource] = []
        watch, _, creations, _ = make_watch(tmp_path, captured, cls=BlindSource)
        with watch:
            watch._dir_poller.min_interval = 0.05
            watch._dir_poller.poll_interval = 0.05
            watch._dir_poller.schedule_next()

            (tmp_path / "new.jsonl").write_text("hello\n")

            assert wait_for(lambda: creations), (
                "directory scan must discover files the event source missed"
            )

    def test_reliability_falls_so_polling_stays_fast(self, tmp_path):
        """A blind source must not be allowed to slow the poller down."""
        captured: list[BlindSource] = []
        # max_poll is capped here because AdaptivePoller's floor is really
        # `estimated_change_interval * 0.5`, which starts at 2.5s regardless of
        # min_poll_interval -- so an uncapped watch polls far slower than
        # min_poll_interval suggests even when the source is known blind.
        watch, changes, _, _ = make_watch(
            tmp_path, captured, cls=BlindSource, max_poll=0.3
        )
        with watch:
            target = tmp_path / "events.jsonl"
            target.write_text("")
            watch.add_file(target)
            initial = watch._files[target].watchdog_reliability

            for i in range(6):
                with target.open("a") as f:
                    f.write(f"line {i}\n")
                assert wait_for(lambda n=i: len(changes) > n, timeout=3.0)

            assert watch._files[target].watchdog_reliability < initial


class TestReliableSource:
    """Local FS with a working backend: polling should get out of the way."""

    def test_reliability_rises_and_poll_backs_off(self, tmp_path):
        captured: list[ScriptedSource] = []
        watch, changes, _, _ = make_watch(tmp_path, captured)
        with watch:
            target = tmp_path / "events.jsonl"
            target.write_text("")
            watch.add_file(target)

            polled = watch._files[target]
            initial_reliability = polled.watchdog_reliability
            initial_interval = polled.poll_interval

            for _ in range(6):
                captured[0].emit_change(target)

            assert polled.watchdog_reliability > initial_reliability
            assert polled.poll_interval > initial_interval
            assert len(changes) >= 6

    def test_beats_a_blind_source_on_poll_interval_once_cold(self, tmp_path):
        """The whole point: reliability, not configuration, sets the cadence.

        Only once the files leave the hot window -- inside it both are pinned
        to the hot ceiling regardless of who reported the change.
        """
        good: list[ScriptedSource] = []
        watch_good, *_ = make_watch(tmp_path / "good", good)
        blind: list[BlindSource] = []
        watch_blind, *_ = make_watch(tmp_path / "blind", blind, cls=BlindSource)

        with watch_good, watch_blind:
            good_file = tmp_path / "good" / "e.jsonl"
            blind_file = tmp_path / "blind" / "e.jsonl"
            good_file.write_text("")
            blind_file.write_text("")
            watch_good.add_file(good_file)
            watch_blind.add_file(blind_file)

            for _ in range(8):
                good[0].emit_change(good_file)
                # The blind watch learns the same number of changes by polling
                watch_blind._source_changed(watch_blind._poll_source, blind_file)

            good_poller = watch_good._files[good_file].poller
            blind_poller = watch_blind._files[blind_file].poller
            assert good_poller.watchdog_reliability > blind_poller.watchdog_reliability

            # Age both past the hot window, then recompute.
            for poller in (good_poller, blind_poller):
                poller.last_change_time -= poller.hot_window * 2
                poller._compute_poll_interval()

            assert not good_poller.is_hot
            assert good_poller.poll_interval > blind_poller.poll_interval


class TestHotSet:
    """A recently-changed file must stay fast regardless of the adaptive drift."""

    def test_hot_file_is_capped_at_the_hot_interval(self, tmp_path):
        captured: list[ScriptedSource] = []
        watch, *_ = make_watch(tmp_path, captured)
        with watch:
            target = tmp_path / "e.jsonl"
            target.write_text("")
            watch.add_file(target)
            poller = watch._files[target].poller

            # A perfectly reliable source would otherwise push this toward
            # max_poll_interval (5s here).
            for _ in range(10):
                captured[0].emit_change(target)

            assert poller.is_hot
            assert poller.poll_interval <= watch._hot_poll_interval

    def test_cold_file_is_allowed_to_drift(self, tmp_path):
        captured: list[ScriptedSource] = []
        watch, *_ = make_watch(tmp_path, captured)
        with watch:
            target = tmp_path / "e.jsonl"
            target.write_text("")
            watch.add_file(target)
            poller = watch._files[target].poller

            for _ in range(10):
                captured[0].emit_change(target)
            hot_interval = poller.poll_interval

            poller.last_change_time -= poller.hot_window * 2
            poller._compute_poll_interval()

            assert poller.poll_interval > hot_interval

    def test_min_poll_interval_is_not_the_real_floor_without_the_cap(self, tmp_path):
        """Documents why the hot cap exists at all.

        A fresh poller's interval comes from estimated_change_interval (5.0),
        halved -- 2.5s -- so min_interval alone never delivers fast polling.
        """
        from experimaestro.filewatcher import AdaptivePoller

        uncapped = AdaptivePoller(min_interval=0.05, max_interval=30.0)
        uncapped._compute_poll_interval()
        assert uncapped.poll_interval > 1.0  # nowhere near min_interval

        capped = AdaptivePoller(
            min_interval=0.05, max_interval=30.0, hot_max_interval=0.2
        )
        capped._compute_poll_interval()
        assert capped.poll_interval == pytest.approx(0.2)


class TestCoalescingSource:
    """macOS FSEvents: merges rapid appends into one late notification."""

    def test_every_append_is_still_observed(self, tmp_path):
        captured: list[CoalescingSource] = []
        watch, changes, _, _ = make_watch(
            tmp_path, captured, cls=CoalescingSource, max_poll=0.3, every=3
        )
        with watch:
            target = tmp_path / "events.jsonl"
            target.write_text("")
            watch.add_file(target)

            for i in range(5):
                with target.open("a") as f:
                    f.write(f"line {i}\n")
                captured[0].emit_change(target)  # swallowed 2 times out of 3
                assert wait_for(lambda n=i: len(changes) > n, timeout=3.0), (
                    f"append {i} was coalesced away and polling did not catch it"
                )


class TestChannelIndependence:
    """Creation and modification reliability are tracked separately."""

    def test_create_only_source_does_not_credit_the_file_channel(self, tmp_path):
        captured: list[CreateOnlySource] = []
        watch, changes, creations, _ = make_watch(
            tmp_path, captured, cls=CreateOnlySource, max_poll=0.3
        )
        with watch:
            target = tmp_path / "events.jsonl"
            target.write_text("first\n")

            dir_reliability_before = watch._dir_poller.watchdog_reliability
            captured[0].emit_created(target)

            # The creation channel is credited...
            assert watch._dir_poller.watchdog_reliability > dir_reliability_before
            assert creations == [target]

            # ...while appends still have to be found by polling.
            file_reliability = watch._files[target].watchdog_reliability
            with target.open("a") as f:
                f.write("second\n")
            assert wait_for(lambda: len(changes) >= 2, timeout=3.0)
            assert watch._files[target].watchdog_reliability < file_reliability


class TestStoppedSource:
    def test_stopped_source_reports_nothing(self, tmp_path):
        """Regression: releasing the OS resource can be asynchronous.

        WatchdogSource.stop() only *queues* the unschedule for the poll thread,
        so events kept arriving after stop() and were still being reported.
        """
        captured: list[ScriptedSource] = []
        watch, changes, creations, deletions = make_watch(tmp_path, captured)
        with watch:
            target = tmp_path / "events.jsonl"
            target.write_text("x")
            watch.add_file(target)

            source = captured[0]
            source.stop()
            assert source._watch is None

            source.emit_change(target)
            source.emit_created(tmp_path / "other.jsonl")
            source.emit_deleted(target)

            assert changes == []
            assert creations == []
            assert deletions == []

    def test_source_is_silent_once_the_watch_closes(self, tmp_path):
        captured: list[ScriptedSource] = []
        watch, changes, _, _ = make_watch(tmp_path, captured)
        target = tmp_path / "events.jsonl"
        target.write_text("x")
        watch.add_file(target)
        watch.close()

        captured[0].emit_change(target)
        assert changes == []
