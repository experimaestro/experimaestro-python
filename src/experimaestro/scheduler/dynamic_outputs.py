"""Handles dynamic task outputs

This module provides support for tasks that produce dynamic outputs during
execution. These outputs can trigger callbacks that submit new tasks.

The task process appends one JSON line per output to
``<job>/.experimaestro/task-outputs.jsonl`` (see
:meth:`~experimaestro.core.objects.Config.register_task_output`). In the
experiment process:

- a :class:`TaskOutputsReader` (one per job and experiment, owned by the
  experiment) reads new lines on the scheduler loop; watchdog only pokes it;
- for each line, the user's output method and callbacks run in the
  experiment's callback executor (a single thread, so they stay serial), and
  each run is tracked by the experiment;
- when the job is over, :meth:`TaskOutputsReader.drain` reads up to the end of
  the file and waits for the callbacks it produced.
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from watchdog.events import FileSystemEventHandler

from experimaestro.filewatcher import FileWatcherService
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.core.objects import WatchedOutput
    from experimaestro.scheduler.experiment import experiment as Experiment

#: How long :meth:`TaskOutputsReader.drain` waits for lines announced in
#: ``.done`` but not visible yet (e.g. NFS caching), in seconds
DRAIN_TIMEOUT = 60.0

#: Polling interval while waiting for those lines, in seconds
DRAIN_POLL_INTERVAL = 0.5


def count_task_outputs(path: Path) -> int:
    """Number of outputs (non-empty lines) in a task outputs file"""
    if not path.is_file():
        return 0
    with path.open("rb") as fp:
        return sum(1 for line in fp if line.strip())


class TaskOutputWatcher:
    """Watches a specific output method for a configuration within a job

    All its methods run in the experiment's callback executor (a single
    thread), so no locking is needed.
    """

    def __init__(self, key: str, method: Callable, job: "Job"):
        self.key = key
        self.method = method
        self.job = job  # The job that produces the dynamic outputs

        #: The callbacks to call (ordered, without duplicates)
        self.callbacks: list[Callable] = []

        #: Processed Config objects, replayed to callbacks added later
        self.processed_events: list[object] = []

    def add_callback(self, callback: Callable):
        """Add a callback and replay the events already processed"""
        if callback in self.callbacks:
            return
        self.callbacks.append(callback)
        for event in self.processed_events:
            self._call(callback, event)

    def process_event(self, raw_event: dict):
        """Convert a raw event to a configuration and dispatch it to callbacks"""
        # The method signature is: method(dep, *args, **kwargs) -> Config
        # We need to provide a marker function that marks the output as
        # a dynamic output of the producing task. This ensures:
        # 1. The checkpoint identifier includes the task (different learning_rates
        #    produce different checkpoints)
        # 2. The dependency is non-blocking (Evaluate can run while Learn runs)
        task_config = self.job.config

        def mark_output(config):
            """Marker function that marks config as dynamic output of the task"""
            return task_config.__xpm__.mark_output(config, dynamic=True)

        try:
            result = self.method(mark_output, *raw_event["args"], **raw_event["kwargs"])
        except Exception:
            logger.exception("Error processing task output event %s", raw_event)
            return

        self.processed_events.append(result)
        for callback in list(self.callbacks):
            self._call(callback, result)

    def _call(self, callback: Callable, event):
        try:
            logger.debug("Calling callback %s with event %s", callback, event)
            callback(event)
        except Exception:
            logger.exception("Error in task output callback")


class TaskOutputsReader(FileSystemEventHandler):
    """Reads the dynamic outputs of one job, for one experiment

    Created by: :meth:`experiment.watch_output`

    Apart from the watchdog callbacks (which only schedule :meth:`poke` on the
    scheduler loop), all methods run on the scheduler loop.
    """

    def __init__(self, xp: "Experiment", job: "Job"):
        super().__init__()
        self.xp = xp
        self.job = job
        self.path = job.task_outputs_path
        self._loop = xp.scheduler.loop
        self._watch = None

        #: Byte offset of the first line not read yet
        self._offset = 0

        #: Number of outputs (non-empty lines) read so far
        self.lines_read = 0

        #: Raw events by key, replayed to watchers added later
        self.events: dict[str, list[dict]] = defaultdict(list)

        #: Map from key (config_id/method_name) to TaskOutputWatcher
        self.watchers: dict[str, TaskOutputWatcher] = {}

        #: Callback runs this reader produced that are not finished yet
        self._pending: set[asyncio.Future] = set()

    def add_watcher(self, watched: "WatchedOutput"):
        """Add a watcher for a specific output method"""
        # watched.config is actually a Config object (method.__self__), not a
        # ConfigInformation, despite the type annotation
        config_id = watched.config.__xpm__.identifier.all.hex()
        key = f"{config_id}/{watched.method_name}"
        logger.debug("Adding watcher for key: %s", key)

        watcher = self.watchers.get(key)
        if watcher is None:
            watcher = self.watchers[key] = TaskOutputWatcher(
                key, watched.method, self.job
            )
            for event in self.events[key]:
                self._dispatch(watcher.process_event, event)

        # Queued after the events above: the callback gets every one of them
        self._dispatch(watcher.add_callback, watched.callback)

        if self._watch is None:
            self._start_watching()
        self.poke()

    def _start_watching(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._watch = FileWatcherService.instance().fswatch(
            self, self.path.parent, False
        )
        logger.debug("Started watching directory %s", self.path.parent)

    def close(self):
        """Stop watching the file"""
        if self._watch is not None:
            try:
                FileWatcherService.instance().fsunwatch(self._watch)
            except KeyError:
                pass  # Already unwatched
            self._watch = None

    def _dispatch(self, fn: Callable, *args):
        future = self.xp.run_callback(fn, *args)
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)

    def poke(self):
        """Read the complete lines appended since the last call"""
        # Re-open each time: this forces NFS to revalidate the file, and
        # outputs are rare enough for the cost not to matter
        try:
            with self.path.open("rb") as fp:
                fp.seek(self._offset)
                data = fp.read()
        except FileNotFoundError:
            return

        # Only consume complete lines: the task may be writing the last one
        end = data.rfind(b"\n") + 1
        self._offset += end

        for line in data[:end].splitlines():
            line = line.strip()
            if not line:
                continue
            self.lines_read += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in task output: %s", line)
                continue

            key = event.get("key")
            self.events[key].append(event)
            if watcher := self.watchers.get(key):
                logger.debug("New task output %s", event)
                self._dispatch(watcher.process_event, event)

    async def drain(self, expected: int | None = None):
        """Read the file to its end and wait for the callbacks it produced

        Only meaningful once the job's process is over, since the end of the
        file is final then.

        :param expected: number of outputs the task wrote (from ``.done``);
            when given, wait (up to :data:`DRAIN_TIMEOUT`) until that many
            lines have been read, since they might not be visible yet
        """
        self.poke()
        if expected is not None and self.lines_read < expected:
            deadline = self._loop.time() + DRAIN_TIMEOUT
            while self.lines_read < expected and self._loop.time() < deadline:
                await asyncio.sleep(DRAIN_POLL_INTERVAL)
                self.poke()
            if self.lines_read < expected:
                logger.warning(
                    "Only %d of the %d outputs of %s could be read",
                    self.lines_read,
                    expected,
                    self.job,
                )

        while self._pending:
            await asyncio.wait(list(self._pending))

    # FileSystemEventHandler methods (watchdog thread)

    def _on_change(self, event):
        if Path(event.src_path) == self.path:
            self._loop.call_soon_threadsafe(self.poke)

    on_modified = _on_change
    on_created = _on_change
