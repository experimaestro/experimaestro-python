"""Handles dynamic task outputs

This module provides support for tasks that produce dynamic outputs during
execution. These outputs can trigger callbacks that submit new tasks.

Key concepts:
- TaskOutputs: Monitors a task's output file for events
- TaskOutputsWorker: Processes events and calls registered callbacks
"""

import asyncio
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Set, TYPE_CHECKING

import concurrent
from watchdog.events import FileSystemEventHandler

from experimaestro.ipc import ipcom
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.core.objects import WatchedOutput
    from experimaestro.scheduler.experiment import experiment


@dataclass
class CallbackItem:
    """A callback to be executed on the worker thread"""

    callback: Callable
    event: object
    update_count: bool

    def run(self, worker: "TaskOutputsWorker") -> None:
        """Execute the callback"""
        try:
            logger.debug("Calling callback %s with event %s", self.callback, self.event)
            self.callback(self.event)
        except Exception:
            logger.exception("Error in task output callback")
        finally:
            if self.update_count:
                asyncio.run_coroutine_threadsafe(
                    worker.xp.update_task_output_count(-1),
                    worker.xp.scheduler.loop,
                ).result()


@dataclass
class RawEventItem:
    """A raw event that needs method processing on the worker thread

    This moves the potentially slow method call to the worker thread
    instead of blocking the file watcher thread.
    """

    watcher: "TaskOutputWatcher"
    raw_event: dict

    def run(self, worker: "TaskOutputsWorker") -> None:
        """Execute the event: call method and dispatch to callbacks"""
        try:
            self.watcher.execute_event(self.raw_event)
        except Exception:
            logger.exception("Error processing raw event")


@dataclass
class CompletionSignal:
    """Signal that job output processing is complete"""

    completion_event: threading.Event

    def run(self, worker: "TaskOutputsWorker") -> None:
        """Signal completion"""
        self.completion_event.set()


# Queue item type
QueueItem = CallbackItem | RawEventItem | CompletionSignal | None


class TaskOutputWatcher:
    """Watches a specific output method for a configuration within a job"""

    def __init__(
        self,
        key: str,
        method: Callable,
        worker: "TaskOutputsWorker",
    ):
        self.key = key
        self.method = method
        self.worker = worker
        self.callbacks: Set[Callable] = set()
        self.processed_events: List[dict] = []
        self.update_xp_futures: list[concurrent.futures.Future] = []

    def add_callback(self, callback: Callable):
        """Add a callback and replay any existing events"""
        # Replay processed events to new callback (don't update count for replays)
        for event in self.processed_events:
            self.worker.add(callback, event, update_count=False)
        self.callbacks.add(callback)

    def process_event(self, raw_event: dict):
        """Queue a raw event for processing on the worker thread

        This queues the event to be processed by the worker thread, which
        will call the method and dispatch to callbacks. This avoids blocking
        the file watcher thread with potentially slow method calls.
        """
        logger.info("Adding raw event %s", raw_event)
        # Update task output count for this event
        self.update_xp_futures.append(
            asyncio.run_coroutine_threadsafe(
                self.worker.xp.update_task_output_count(1),
                self.worker.xp.scheduler.loop,
            )
        )

        # Queue the raw event for processing on worker thread
        self.worker.queue.put(RawEventItem(watcher=self, raw_event=raw_event))

    def execute_event(self, raw_event: dict):
        """Execute a raw event on the worker thread

        This is called by the worker thread to process the event:
        1. Call the method to convert raw event to configuration
        2. Store in processed_events for replay
        3. Dispatch to all callbacks
        """
        try:
            # The method signature is: method(dep, *args, **kwargs) -> Config
            # We need to provide a marker function that marks the output
            def mark_output(config):
                """Marker function that just returns the config"""
                return config

            result = self.method(mark_output, *raw_event["args"], **raw_event["kwargs"])
            self.processed_events.append(result)

            # Dispatch to all callbacks (don't update count, already counted above)
            for callback in self.callbacks:
                self.worker.add(callback, result, update_count=False)
        except Exception:
            logger.exception("Error processing task output event")


class TaskOutputs(FileSystemEventHandler):
    """Monitors dynamic outputs generated by one task"""

    #: Global dictionary mapping paths to TaskOutputs instances
    HANDLERS: Dict[Path, "TaskOutputs"] = {}

    #: Global lock for accessing HANDLERS
    LOCK = threading.Lock()

    @staticmethod
    def get_or_create(path: Path, worker: "TaskOutputsWorker") -> "TaskOutputs":
        """Get or create a TaskOutputs instance for the given path"""
        with TaskOutputs.LOCK:
            if path in TaskOutputs.HANDLERS:
                instance = TaskOutputs.HANDLERS[path]
                # Update worker reference in case this is a new experiment
                instance.worker = worker
                # Clear old watchers - new ones will be added and replay events
                instance.watchers.clear()
                return instance

            instance = TaskOutputs(path, worker)
            TaskOutputs.HANDLERS[path] = instance
            return instance

    def __init__(self, path: Path, worker: "TaskOutputsWorker"):
        """Initialize monitoring for a task output path"""
        super().__init__()
        logger.debug("Creating TaskOutputs monitor for %s", path)
        self.path = path
        self.worker = worker
        self._watch_handle = None
        self._file_handle = None
        self._lock = threading.Lock()

        # Map from key (config_id/method_name) to TaskOutputWatcher
        self.watchers: Dict[str, TaskOutputWatcher] = {}

    def start_watching(self):
        """Start watching the task output file"""
        logger.debug("Starting to watch task outputs at %s", self.path)
        with self._lock:
            if self._watch_handle is not None:
                return  # Already watching

            # Ensure the directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Start file system watching
            self._watch_handle = ipcom().fswatch(self, self.path.parent, False)
            logger.debug("Started watching directory %s", self.path.parent)

            # Process any existing content
            self._process_file()

    def stop_watching(self):
        """Stop watching the task output file"""
        with self._lock:
            if self._watch_handle is not None:
                try:
                    ipcom().fsunwatch(self._watch_handle)
                except KeyError:
                    pass  # Already unwatched
                self._watch_handle = None

            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None

    def add_watcher(self, watched: "WatchedOutput"):
        """Add a watcher for a specific output method"""
        # Use the identifier from the config - watched.config is actually a Config object
        # (method.__self__), not a ConfigInformation, despite the type annotation
        config_id = watched.config.__xpm__.identifier.all.hex()
        key = f"{config_id}/{watched.method_name}"
        logger.debug("Adding watcher for key: %s", key)

        with self._lock:
            is_new = key not in self.watchers
            if is_new:
                self.watchers[key] = TaskOutputWatcher(key, watched.method, self.worker)

            # If this is a new watcher and the file already exists, replay events from file
            if is_new and self.path.exists():
                self._replay_events_for_key(key)

            self.watchers[key].add_callback(watched.callback)

    def _replay_events_for_key(self, key: str):
        """Replay events from the file for a specific key"""
        if not self.path.exists():
            return

        with self.path.open("rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    if event.get("key") == key:
                        self.watchers[key].process_event(event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in task output: %s", line)
                except Exception:
                    logger.exception("Error processing task output line")

    def _process_file(self):
        """Process the task output file"""
        if not self.path.exists():
            return

        if self._file_handle is None:
            self._file_handle = self.path.open("rt")

        while line := self._file_handle.readline():
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                key = event.get("key")
                if key and key in self.watchers:
                    logger.info("Adding one event %s", event)
                    self.watchers[key].process_event(event)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in task output: %s", line)
            except Exception:
                logger.exception("Error processing task output line")

    # FileSystemEventHandler methods
    def on_modified(self, event):
        if Path(event.src_path) == self.path:
            with self._lock:
                self._process_file()

    def on_created(self, event):
        if Path(event.src_path) == self.path:
            with self._lock:
                self._process_file()


class TaskOutputsWorker(threading.Thread):
    """Worker thread that processes task output callbacks for one experiment"""

    def __init__(self, xp: "experiment"):
        super().__init__(name="task-outputs-worker", daemon=True)
        self.queue: queue.Queue = queue.Queue()
        self.xp = xp
        self._monitors: Dict[Path, TaskOutputs] = {}
        self._lock = threading.Lock()

    def watch_output(self, watched: "WatchedOutput"):
        """Register a watched output

        :param watched: The watched output specification
        """
        # Get the job's task output path
        job = watched.job
        if job is None:
            logger.warning("Cannot watch output without job: %s", watched)
            return

        path = job.task_outputs_path
        logger.debug("Registering task output listener at %s", path)

        with self._lock:
            if path not in self._monitors:
                monitor = TaskOutputs.get_or_create(path, self)
                self._monitors[path] = monitor
                monitor.start_watching()
            else:
                monitor = self._monitors[path]

        monitor.add_watcher(watched)

    def add(self, callback: Callable, event, update_count: bool = True):
        """Add an event to the processing queue

        :param callback: The callback to call with the event
        :param event: The event data
        :param update_count: Whether to update the task output count (False for replays)
        """

        async def add_one():
            total = await self.xp.update_task_output_count(1)
            logger.info(
                "Added one task output to be processed to experiment (total %d)", total
            )

        if update_count:
            asyncio.run_coroutine_threadsafe(
                add_one(),
                self.xp.scheduler.loop,
            ).result()

        self.queue.put(
            CallbackItem(callback=callback, event=event, update_count=update_count)
        )

    def run(self):
        """Main worker loop"""
        logger.debug("Starting task outputs worker")
        while True:
            item = self.queue.get()

            if item is None:
                # Shutdown signal
                break

            logger.info("Processing event %s", item)
            try:
                item.run(self)
            except Exception:
                logger.exception("Got an error while processing event %s", item)
            self.queue.task_done()
            logger.info("Processed event %s", item)

        logger.debug("Task outputs worker stopped")

    def process_job_outputs(self, job) -> None:
        """Explicitly process any remaining task outputs for a completed job.

        This is called when a job finishes to ensure all task outputs written
        by the job are processed before the experiment considers exiting.
        This is necessary because file system watchers may have latency.

        :param job: The job that has finished
        """
        path = job.task_outputs_path
        with self._lock:
            monitor = self._monitors.get(path)

        if monitor is not None:
            with monitor._lock:
                monitor._process_file()

    async def aio_process_job_outputs(self, job: "Job"):
        """Process job outputs and return a future that resolves when complete.

        This method:
        1. Processes the task outputs file to queue any remaining events
        3. Returns a future that resolves when the completion signal is processed

        The returned future can be awaited to ensure all callbacks have completed.

        :param job: The job that has finished
        :return: Future that resolves when processing is complete, or None if no monitor
        """
        path = job.task_outputs_path
        with self._lock:
            monitor = self._monitors.get(path)

        if monitor is None:
            return None

        # Process file to queue any remaining events
        with monitor._lock:
            monitor._process_file()

        for key, watcher in monitor.watchers.items():
            futures = watcher.update_xp_futures
            logger.info(
                "Waiting for completion of %d task outputs workers for %s",
                len(futures),
                key,
            )
            await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
            logger.info("Task outputs processed for %s", self)

    def shutdown(self):
        """Stop the worker and all monitors"""
        # Stop all monitors
        with self._lock:
            for monitor in self._monitors.values():
                monitor.stop_watching()
            self._monitors.clear()

        # Signal the worker to stop
        self.queue.put(None)
