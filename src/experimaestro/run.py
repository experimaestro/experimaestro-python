"""Command line parsing"""

import os
from datetime import datetime
from pathlib import Path
import signal
import sys
import json
import threading
from typing import List, TYPE_CHECKING
import filelock
from experimaestro.notifications import progress, report_eoj, start_of_job
from experimaestro.utils.multiprocessing import delayed_shutdown
from experimaestro.exceptions import GracefulTimeout, TaskCancelled
from experimaestro.locking import JobDependencyLocks
from .core.types import ObjectType
from experimaestro.utils import logger
from experimaestro.core.objects import ConfigInformation
import experimaestro.taskglobals as taskglobals
import atexit

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import MockJob


def parse_commandline(argv=None):
    """Called when executing a task"""
    if argv is None:
        argv = sys.argv[1:]
    taskid, params = argv
    tasktype = ObjectType.REGISTERED[taskid]
    with open(params, "r") as fp:
        params = json.load(fp)
        ConfigInformation.LOADING = True
        task = tasktype(**params)
        ConfigInformation.LOADING = False
        task.execute()


def run(parameters: Path):
    with open(parameters, "r") as fp:
        params = json.load(fp)
        env = taskglobals.Env.instance()

        env.wspath = Path(params["workspace"])
        env.taskpath = parameters.parent

        task = ConfigInformation.fromParameters(params["objects"])
        task.__taskdir__ = Path.cwd()

        # Store task reference for signal handlers (graceful termination)
        env.current_task = task

        # Notify that the task has started
        progress(0)

        # Execute the task
        task.execute()


def rmfile(path: Path):
    if path.is_file():
        logger.debug("Removing file %s", path)
        path.unlink()


# =============================================================================
# Carbon Tracking Integration
# =============================================================================


def _init_carbon_tracking(
    workspace_path: Path,
    task_path: Path,
    job_id: str,
    carbon_settings: dict | None = None,
) -> tuple["CarbonTracker | None", "threading.Thread | None", "threading.Event | None"]:
    """Initialize carbon tracking for a job.

    Args:
        workspace_path: Path to the workspace.
        task_path: Path to the task directory.
        job_id: Job identifier.
        carbon_settings: Carbon settings dict from params.json (preferred), or None
            to use default settings.

    Returns:
        Tuple of (tracker, reporter_thread, stop_event) or (None, None, None) if unavailable.
    """
    try:
        from experimaestro.carbon import create_tracker, is_available
        from experimaestro.settings import CarbonSettings

        # Use provided settings or defaults
        if carbon_settings is None:
            carbon_settings = {}
        settings = CarbonSettings(**carbon_settings)

        if not settings.enabled:
            logger.debug("Carbon tracking disabled in settings")
            return None, None, None

        if not is_available():
            logger.debug("CodeCarbon not installed, carbon tracking unavailable")
            return None, None, None

        # Create tracker with settings
        tracker = create_tracker(
            country_iso_code=settings.country_iso_code,
            region=settings.region,
        )

        # Start tracking
        tracker.start()
        logger.info("Carbon tracking started")

        # Set up in environment
        env = taskglobals.Env.instance()
        env.carbon_tracker = tracker

        # Create periodic reporter thread
        stop_event = threading.Event()
        reporter_thread = threading.Thread(
            target=_carbon_reporter_loop,
            args=(
                tracker,
                workspace_path,
                task_path,
                job_id,
                settings.report_interval_s,
                stop_event,
            ),
            daemon=True,
            name="carbon-reporter",
        )
        reporter_thread.start()

        return tracker, reporter_thread, stop_event

    except Exception as e:
        logger.warning("Failed to initialize carbon tracking: %s", e)
        return None, None, None


def _carbon_reporter_loop(
    tracker,
    workspace_path: Path,
    task_path: Path,
    job_id: str,
    interval_s: float,
    stop_event: threading.Event,
):
    """Background thread that periodically emits carbon metrics events.

    Args:
        tracker: Carbon tracker instance.
        workspace_path: Path to the workspace.
        task_path: Path to the task directory.
        job_id: Job identifier.
        interval_s: Reporting interval in seconds.
        stop_event: Event to signal thread termination.
    """
    from experimaestro.scheduler.state_status import CarbonMetricsEvent, JobEventWriter

    task_id = task_path.parent.name

    # Create event writer for this job
    try:
        event_writer = JobEventWriter(
            workspace_path, task_id, job_id, 0, job_path=task_path
        )
    except Exception as e:
        logger.warning("Failed to create carbon event writer: %s", e)
        return

    try:
        while not stop_event.wait(timeout=interval_s):
            try:
                metrics = tracker.get_current_metrics()
                event = CarbonMetricsEvent(
                    job_id=job_id,
                    co2_kg=metrics.co2_kg,
                    energy_kwh=metrics.energy_kwh,
                    cpu_power_w=metrics.cpu_power_w,
                    gpu_power_w=metrics.gpu_power_w,
                    ram_power_w=metrics.ram_power_w,
                    duration_s=metrics.duration_s,
                    region=metrics.region,
                    is_final=False,
                )
                event_writer.write_event(event)
                logger.debug(
                    "Carbon metrics: %.4f kg CO2, %.4f kWh",
                    metrics.co2_kg,
                    metrics.energy_kwh,
                )
            except Exception as e:
                logger.debug("Failed to emit carbon metrics: %s", e)

        # Write one final periodic update before exiting (before the main thread writes is_final=True)
        try:
            metrics = tracker.get_current_metrics()
            event = CarbonMetricsEvent(
                job_id=job_id,
                co2_kg=metrics.co2_kg,
                energy_kwh=metrics.energy_kwh,
                cpu_power_w=metrics.cpu_power_w,
                gpu_power_w=metrics.gpu_power_w,
                ram_power_w=metrics.ram_power_w,
                duration_s=metrics.duration_s,
                region=metrics.region,
                is_final=False,
            )
            event_writer.write_event(event)
            logger.debug("Final periodic carbon metrics written before shutdown")
        except Exception as e:
            logger.debug("Failed to emit final periodic carbon metrics: %s", e)
    finally:
        event_writer.close()


def _stop_carbon_tracking(
    tracker,
    reporter_thread: "threading.Thread | None",
    stop_event: "threading.Event | None",
    workspace_path: Path,
    task_path: Path,
    job_id: str,
    task_id: str,
    start_time: datetime,
    mock_job: "MockJob",
):
    """Stop carbon tracking and record final metrics.

    Args:
        tracker: Carbon tracker instance.
        reporter_thread: Periodic reporter thread.
        stop_event: Event to signal thread termination.
        workspace_path: Path to the workspace.
        task_path: Path to the task directory.
        job_id: Job identifier.
        task_id: Task type identifier.
        start_time: Job start time.
        mock_job: MockJob to apply the carbon event to.
    """
    if tracker is None:
        return

    # Stop reporter thread
    if stop_event is not None:
        stop_event.set()
    if reporter_thread is not None:
        reporter_thread.join(timeout=5.0)

    try:
        # Stop tracker and get final metrics
        metrics = tracker.stop()
        logger.info(
            "Carbon tracking stopped: %.4f kg CO2, %.4f kWh, %.0f s",
            metrics.co2_kg,
            metrics.energy_kwh,
            metrics.duration_s,
        )

        # Record to carbon storage first
        from experimaestro.carbon.storage import CarbonRecord, CarbonStorage

        record = CarbonRecord(
            job_id=job_id,
            task_id=task_id,
            started_at=start_time.isoformat(),
            ended_at=datetime.now().isoformat(),
            co2_kg=metrics.co2_kg,
            energy_kwh=metrics.energy_kwh,
            cpu_power_w=metrics.cpu_power_w,
            gpu_power_w=metrics.gpu_power_w,
            ram_power_w=metrics.ram_power_w,
            duration_s=metrics.duration_s,
            region=metrics.region,
        )

        written = False
        try:
            storage = CarbonStorage(workspace_path)
            storage.write_record(record)
            written = True
        except Exception as storage_error:
            logger.warning(
                "Failed to write carbon record to storage: %s", storage_error
            )

        # Emit final carbon event with written status
        from experimaestro.scheduler.state_status import (
            CarbonMetricsEvent,
            JobEventWriter,
        )

        task_id_from_path = task_path.parent.name
        event_writer = JobEventWriter(
            workspace_path, task_id_from_path, job_id, 0, job_path=task_path
        )
        try:
            event = CarbonMetricsEvent(
                job_id=job_id,
                co2_kg=metrics.co2_kg,
                energy_kwh=metrics.energy_kwh,
                cpu_power_w=metrics.cpu_power_w,
                gpu_power_w=metrics.gpu_power_w,
                ram_power_w=metrics.ram_power_w,
                duration_s=metrics.duration_s,
                region=metrics.region,
                is_final=True,
                written=written,
            )
            event_writer.write_event(event)

            # Apply event to MockJob so it's included in status.json
            mock_job.apply_event(event)
        finally:
            event_writer.close()

    except Exception as e:
        logger.warning("Failed to finalize carbon tracking: %s", e)

    # Clear from environment
    env = taskglobals.Env.instance()
    env.carbon_tracker = None


# Type hint for optional import
try:
    from experimaestro.carbon.base import CarbonTracker
except ImportError:
    CarbonTracker = None  # type: ignore


class TaskRunner:
    """Runs a task, after locking"""

    def __init__(self, scriptpath: str, lockfiles: List[str]):
        # Sets the working directory
        self.scriptpath = Path(scriptpath)
        self.pidfile = self.scriptpath.with_suffix(".pid")
        self.lockfiles = lockfiles
        self.donepath = self.scriptpath.with_suffix(".done")
        self.failedpath = self.scriptpath.with_suffix(".failed")
        self.started = False
        self.locks = []
        self.dynamic_locks = JobDependencyLocks()
        env = taskglobals.Env.instance()
        env.taskpath = self.scriptpath.parent

        self.cleaned = False

        # Set when SIGTERM/SIGINT received - prevents marking task as done
        # if task catches and suppresses TaskCancelled
        self._cancelled = False

        # Carbon tracking state
        self._carbon_tracker = None
        self._carbon_reporter_thread = None
        self._carbon_stop_event = None
        self._job_start_time: datetime | None = None

        # MockJob for tracking state (loaded from status.json)
        self._mock_job: "MockJob | None" = None

    def _load_mock_job(self) -> "MockJob":
        """Load MockJob from disk for state tracking."""
        from experimaestro.scheduler.state_provider import MockJob

        workdir = self.scriptpath.parent
        job_id = workdir.name
        task_id = workdir.parent.name

        return MockJob.from_disk(workdir, task_id, job_id)

    def _write_status(self) -> None:
        """Write status.json from MockJob state (while still holding locks)."""
        if self._mock_job is None:
            return

        try:
            self._mock_job.write_status()
        except Exception as e:
            logger.warning("Failed to write status.json: %s", e)

    def cleanup(self):
        if not self.cleaned:
            self.cleaned = True
            logger.info("Cleaning up")
            rmfile(self.pidfile)

            # Load MockJob for state tracking
            self._mock_job = self._load_mock_job()

            # Stop carbon tracking and record metrics
            if self._carbon_tracker is not None:
                workdir = self.scriptpath.parent
                task_path = workdir
                job_id = workdir.name
                task_id = workdir.parent.name

                # Load workspace path from params if available
                workspace_path = None
                params_path = workdir / "params.json"
                if params_path.exists():
                    try:
                        with params_path.open() as f:
                            params = json.load(f)
                        workspace_path = Path(params.get("workspace", ""))
                    except Exception:
                        pass

                if workspace_path and self._job_start_time:
                    _stop_carbon_tracking(
                        self._carbon_tracker,
                        self._carbon_reporter_thread,
                        self._carbon_stop_event,
                        workspace_path,
                        task_path,
                        job_id,
                        task_id,
                        self._job_start_time,
                        self._mock_job,
                    )
                self._carbon_tracker = None

            # Write status.json while still holding locks
            self._write_status()

            # Release IPC locks
            for lock in self.locks:
                try:
                    if lock.acquired:
                        logger.info("Releasing lock")
                        lock.release()
                        logger.info("Released lock")
                except Exception:
                    logger.error("Error while releasing lock %s", lock)

            # Note: dynamic dependency locks are released via context manager
            # in the run() method, not here

            if self.started:
                report_eoj()
            logger.info("Finished cleanup")

    def handle_error(self, code, frame_type, reason: str = "failed", message: str = ""):
        """Handle task error and write failure information.

        Args:
            code: Exit code
            frame_type: Signal frame type (unused)
            reason: Failure reason (e.g., "failed", "timeout")
            message: Optional message with details
        """
        logger.info("Error handler: finished with code %d, reason=%s", code, reason)
        failure_info = {"code": code, "reason": reason}
        if message:
            failure_info["message"] = message
        self.failedpath.write_text(json.dumps(failure_info))
        self.cleanup()
        logger.info("Exiting")
        delayed_shutdown(60, exit_code=code)
        sys.exit(1)

    def _background_cleanup(self, reason: str, message: str = ""):
        """Run framework cleanup in background thread."""
        try:
            logger.info("Background cleanup: reason=%s", reason)
            failure_info = {"code": signal.SIGTERM, "reason": reason}
            if message:
                failure_info["message"] = message
            self.failedpath.write_text(json.dumps(failure_info))
            self.cleanup()
            logger.info("Background cleanup finished")
        except Exception:
            logger.exception("Error during background cleanup")

    def handle_sigterm(self, signum, frame):
        """Handle SIGTERM signal for graceful termination.

        This is called when the job is cancelled (e.g., via scancel).
        Spawns a cleanup thread and raises TaskCancelled to allow the
        task to do its own cleanup if needed.
        """
        logger.warning("SIGTERM received, initiating graceful termination")

        # Mark as cancelled to prevent marking as done if task catches exception
        self._cancelled = True

        # Start framework cleanup in background thread
        # This runs regardless of whether the task catches TaskCancelled
        cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            args=("cancelled", "Job terminated by SIGTERM"),
            daemon=True,
        )
        cleanup_thread.start()

        # Get remaining time from launcher if available
        remaining_time = None
        env = taskglobals.Env.instance()
        if env.launcher_info is not None:
            try:
                remaining_time = env.launcher_info.remaining_time()
            except Exception:
                pass

        # Raise exception to interrupt main thread
        # Task can catch this to do its own cleanup
        raise TaskCancelled("Job cancelled by SIGTERM", remaining_time=remaining_time)

    def run(self):
        atexit.register(self.cleanup)
        # Both SIGTERM (scancel) and SIGINT (Ctrl+C) trigger graceful termination
        sigterm_handler = signal.signal(signal.SIGTERM, self.handle_sigterm)
        sigint_handler = signal.signal(signal.SIGINT, self.handle_sigterm)

        def remove_signal_handlers(remove_cleanup=True):
            """Removes cleanup in forked processes"""
            signal.signal(signal.SIGTERM, sigterm_handler)
            signal.signal(signal.SIGINT, sigint_handler)
            atexit.unregister(self.cleanup)

        if sys.platform != "win32":
            os.register_at_fork(after_in_child=remove_signal_handlers)

        try:
            workdir = self.scriptpath.parent
            os.chdir(workdir)
            os.getpid()
            logger.info("Working in directory %s", workdir)

            for lockfile in self.lockfiles:
                fullpath = str(Path(lockfile).resolve())
                logger.info("Locking %s", fullpath)
                lock = filelock.FileLock(fullpath)
                # MAYBE: should have a clever way to lock
                # Problem = slurm would have a job doing nothing...
                # Fix = maybe with two files
                lock.acquire()
                self.locks.append(lock)

            # Load and setup dynamic dependency locks from locks.json
            locks_path = workdir / "locks.json"
            if locks_path.exists():
                logger.info("Loading dynamic dependency locks from %s", locks_path)
                with locks_path.open() as f:
                    locks_data = json.load(f)
                self.dynamic_locks = JobDependencyLocks.from_json(
                    locks_data.get("dynamic_locks", [])
                )
                logger.info(
                    "Loaded %d dynamic dependency locks", len(self.dynamic_locks.locks)
                )

            # Check if failed/done have been generated by another job
            if self.donepath.is_file():
                logger.info("Job already completed")
            else:
                logger.info("Running task")
                rmfile(self.failedpath)
                self.started = True

                # Notify that the job has started
                start_of_job()

                # Initialize carbon tracking
                self._job_start_time = datetime.now()
                job_id = workdir.name

                # Load workspace path and carbon settings from params
                params_path = workdir / "params.json"
                workspace_path = None
                carbon_settings = None
                if params_path.exists():
                    try:
                        with params_path.open() as f:
                            params_data = json.load(f)
                        workspace_path = Path(params_data.get("workspace", ""))
                        carbon_settings = params_data.get("carbon")
                    except Exception as e:
                        logger.debug("Failed to load params for carbon tracking: %s", e)

                if workspace_path:
                    (
                        self._carbon_tracker,
                        self._carbon_reporter_thread,
                        self._carbon_stop_event,
                    ) = _init_carbon_tracking(
                        workspace_path,
                        workdir,
                        job_id,
                        carbon_settings=carbon_settings,
                    )

                # Acquire dynamic dependency locks while running the task
                with self.dynamic_locks.dependency_locks():
                    run(workdir / "params.json")

                # ... remove the handlers
                remove_signal_handlers(remove_cleanup=False)

                # Check if cancellation occurred (task caught and suppressed TaskCancelled)
                if self._cancelled:
                    logger.info(
                        "Task completed but was cancelled - not marking as done"
                    )
                    sys.exit(1)

                # Everything went OK
                logger.info("Task ended successfully")
                self.cleanup()
                sys.exit(0)
        except GracefulTimeout as e:
            logger.info("Task requested graceful timeout: %s", e.message)
            self.handle_error(1, None, reason="timeout", message=e.message)

        except TaskCancelled as e:
            # Cleanup is already running in background thread (started by signal handler)
            # Just log and exit - the background thread handles everything
            logger.info("Task cancelled: %s", e.message)
            sys.exit(1)

        except Exception:
            logger.exception("Got exception while running")
            self.handle_error(1, None)

        except SystemExit as e:
            if e.code == 0:
                # Normal exit, just create the ".done" file
                self.donepath.touch()

                # ... and finish the exit process
                raise
            elif self._cancelled:
                # Task was cancelled - .failed already written by background cleanup thread
                # Just exit without overwriting
                raise
            else:
                self.handle_error(e.code, None)
