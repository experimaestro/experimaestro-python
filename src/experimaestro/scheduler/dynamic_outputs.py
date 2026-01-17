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

from watchdog.events import FileSystemEventHandler

from experimaestro.ipc import ipcom
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.core.objects import WatchedOutput
    from experimaestro.scheduler.experiment import experiment


@dataclass
class CallbackItem:
    """A callback to be executed on the worker thread

    Created by: TaskOutputsWorker.add() when dispatching events to callbacks
    Processed by: TaskOutputsWorker.run() (main worker loop)
    """

    callback: Callable
    event: object

    def run(self, worker: "TaskOutputsWorker") -> None:
        """Execute the callback

        Called by: TaskOutputsWorker.run() (worker thread)
        """
        try:
            logger.debug("Calling callback %s with event %s", self.callback, self.event)
            self.callback(self.event)
        except Exception:
            logger.exception("Error in task output callback")
        finally:
            # Always decrease the value
            asyncio.run_coroutine_threadsafe(
                worker.xp.update_task_output_count(-1),
                worker.xp.scheduler.loop,
            ).result()


@dataclass
class RawEventItem:
    """A raw event that needs method processing on the worker thread

    Created by: TaskOutputWatcher.process_event() (file watcher thread)
    Processed by: TaskOutputsWorker.run() (worker thread)

    This moves the potentially slow method call to the worker thread
    instead of blocking the file watcher thread.
    """

    watcher: "TaskOutputWatcher"
    raw_event: dict

    def run(self, worker: "TaskOutputsWorker") -> None:
        """Execute the event: call method and dispatch to callbacks

        Called by: TaskOutputsWorker.run() (worker thread)

        Calls watcher.execute_event() which:
        1. Converts raw event to Config via the method
        2. Dispatches to all registered callbacks via worker.add()

        Decrements task_output_count after processing (balances the increment
        done in process_event()).
        """
        try:
            self.watcher.execute_event(self.raw_event)
        except Exception:
            logger.exception("Error processing raw event")
        finally:
            asyncio.run_coroutine_threadsafe(
                worker.xp.update_task_output_count(-1),
                worker.xp.scheduler.loop,
            ).result()


# Queue item type
QueueItem = CallbackItem | RawEventItem | None


class TaskOutputWatcher:
    """Watches a specific output method for a configuration within a job

    Created by: TaskOutputs.add_watcher()

    Flow:
    1. File watcher detects change -> TaskOutputs._process_file()
    2. _process_file() calls process_event() for each new event
    3. process_event() queues RawEventItem to worker thread
    4. Worker thread calls execute_event() via RawEventItem.run()
    5. execute_event() converts raw event to Config and dispatches to callbacks

    Thread safety:
    - add_callback() is called from the main thread
    - execute_event() is called from the worker thread
    - Both access callbacks and processed_events, so _lock protects them
    """

    def __init__(
        self,
        key: str,
        method: Callable,
        worker: "TaskOutputsWorker",
    ):
        self.key = key
        self.method = method
        self.worker = worker
        self._lock = threading.Lock()  # Protects callbacks and processed_events

        #: The callbacks to call
        self.callbacks: Set[Callable] = set()
        self.processed_events: List[object] = []  # Processed Config objects for replay

    def add_callback(self, callback: Callable):
        """Add a callback and replay any existing events

        Called by: TaskOutputs.add_watcher() (main thread)

        Thread safety: Must hold _lock to ensure atomicity between checking
        processed_events and adding to callbacks. This prevents a race where
        execute_event() could dispatch to callbacks between these operations,
        causing the callback to miss an event.
        """
        with self._lock:
            # Add callback first, then replay. This ensures that if execute_event
            # runs concurrently, the callback will receive new events directly.
            # Any events already in processed_events will be replayed below.
            self.callbacks.add(callback)
            # Replay processed events to new callback (with counting)
            for event in self.processed_events:
                self.worker.add(callback, event)

    def process_event(self, raw_event: dict):
        """Queue a raw event for processing on the worker thread

        Called by: TaskOutputs._process_file() (file watcher thread)

        This queues the event to be processed by the worker thread, which
        will call the method and dispatch to callbacks. This avoids blocking
        the file watcher thread with potentially slow method calls.

        Count management: Increments task_output_count here. RawEventItem.run()
        decrements it after processing. This ensures the experiment waits for
        all queued events to be processed before exiting.
        """
        logger.info("Adding raw event %s", raw_event)

        # Increment count before queuing to ensure experiment waits
        self.worker.update_count(1)

        # Queue the raw event for processing on worker thread
        self.worker.queue.put(RawEventItem(watcher=self, raw_event=raw_event))

    def execute_event(self, raw_event: dict):
        """Execute a raw event on the worker thread

        Called by: RawEventItem.run() (worker thread)

        1. Call the method to convert raw event to configuration
        2. Store in processed_events for replay
        3. Dispatch to all callbacks via worker.add()

        Thread safety: Must hold _lock when modifying processed_events and
        reading callbacks to ensure atomicity with add_callback().
        """
        try:
            # The method signature is: method(dep, *args, **kwargs) -> Config
            # We need to provide a marker function that marks the output
            def mark_output(config):
                """Marker function that just returns the config"""
                return config

            result = self.method(mark_output, *raw_event["args"], **raw_event["kwargs"])

            # Hold lock while updating processed_events and reading callbacks
            # to ensure atomicity with add_callback()
            with self._lock:
                self.processed_events.append(result)
                # Take a snapshot of callbacks to dispatch to
                callbacks_snapshot = list(self.callbacks)

            # Dispatch to all callbacks (outside lock to avoid blocking add_callback)
            for callback in callbacks_snapshot:
                self.worker.add(callback, result)
        except Exception:
            logger.exception("Error processing task output event")


class TaskOutputs(FileSystemEventHandler):
    """Monitors dynamic outputs generated by one task

    Created by: TaskOutputsWorker.watch_output() via get_or_create()

    Watches a task-outputs.jsonl file for new events. When events are detected,
    dispatches them to the appropriate TaskOutputWatcher based on the event key.

    Thread safety: File watching callbacks run on the watchdog thread,
    but processing is queued to the TaskOutputsWorker thread.
    """

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
    """Worker thread that processes all task output callbacks for one experiment

    Created by: experiment.__enter__()

    Main responsibilities:
    1. Manage TaskOutputs monitors for each job with watched outputs
    2. Process RawEventItems and CallbackItems from the queue
    3. Track task_output_count for experiment exit synchronization

    Thread model:
    - File watchers run on watchdog thread, queue items to this worker
    - This worker processes items sequentially from the queue
    - Count updates are synchronized with the scheduler's event loop
    """

    def __init__(self, xp: "experiment"):
        super().__init__(name="task-outputs-worker", daemon=True)
        self.queue: queue.Queue = queue.Queue()
        self.xp = xp
        self._monitors: Dict[Path, TaskOutputs] = {}
        self._lock = threading.Lock()

    def update_count(self, delta: int):
        """Update task_output_count safely from any thread.

        Handles the case where we might be on the event loop thread
        (which would deadlock if we used run_coroutine_threadsafe().result()).
        """
        loop = self.xp.scheduler.loop
        try:
            running_loop = asyncio.get_running_loop()
            on_event_loop = running_loop is loop
        except RuntimeError:
            on_event_loop = False

        if on_event_loop:
            # We're on the event loop thread, create a task (don't block)
            asyncio.create_task(self.xp.update_task_output_count(delta))
        else:
            # We're on another thread, use run_coroutine_threadsafe and wait
            asyncio.run_coroutine_threadsafe(
                self.xp.update_task_output_count(delta),
                loop,
            ).result()

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

    def add(self, callback: Callable, event):
        """Add a callback event to the processing queue

        Called by: TaskOutputWatcher.execute_event() or add_callback()

        :param callback: The callback to call with the event
        :param event: The processed Config object
        """
        self.update_count(1)
        self.queue.put(CallbackItem(callback=callback, event=event))

    def run(self):
        """Main worker loop - processes items from queue until shutdown

        Called by: threading.Thread.start()

        Processes:
        - RawEventItem: calls execute_event() to convert and dispatch
        - CallbackItem: calls the user callback and updates count
        - None: shutdown signal
        """
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
        """Process any remaining task outputs for a completed job.

        Called by: Job.aio_done_handler() (scheduler event loop)

        This ensures all task outputs written by the job are queued for processing
        before the experiment considers exiting. File system watchers may have
        latency, so we explicitly read the file here.

        Note: This only queues the events. The actual callbacks will complete
        asynchronously and decrement task_output_count when done.

        :param job: The job that has finished
        """
        path = job.task_outputs_path
        with self._lock:
            monitor = self._monitors.get(path)

        if monitor is None:
            return

        # Process file to queue any remaining events
        with monitor._lock:
            monitor._process_file()

    def shutdown(self):
        """Stop the worker and all monitors"""
        # Stop all monitors
        with self._lock:
            for monitor in self._monitors.values():
                monitor.stop_watching()
            self._monitors.clear()

        # Signal the worker to stop
        self.queue.put(None)
