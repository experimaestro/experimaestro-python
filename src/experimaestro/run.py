"""Command line parsing"""

import os
from datetime import datetime
from pathlib import Path
import signal
import sys
import json
import threading
from typing import List
import filelock
from experimaestro.notifications import progress, report_eoj, start_of_job
from experimaestro.utils.multiprocessing import delayed_shutdown
from experimaestro.exceptions import GracefulTimeout
from experimaestro.locking import JobDependencyLocks
from .core.types import ObjectType
from experimaestro.utils import logger
from experimaestro.core.objects import ConfigInformation
import experimaestro.taskglobals as taskglobals
import atexit


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
    report_interval_s: float = 60.0,
) -> tuple["CarbonTracker | None", "threading.Thread | None", "threading.Event | None"]:
    """Initialize carbon tracking for a job.

    Args:
        workspace_path: Path to the workspace.
        task_path: Path to the task directory.
        job_id: Job identifier.
        report_interval_s: How often to emit periodic carbon events.

    Returns:
        Tuple of (tracker, reporter_thread, stop_event) or (None, None, None) if unavailable.
    """
    try:
        from experimaestro.carbon import create_tracker, is_available
        from experimaestro.settings import get_settings

        settings = get_settings()
        carbon_settings = settings.carbon

        if not carbon_settings.enabled:
            logger.debug("Carbon tracking disabled in settings")
            return None, None, None

        if not is_available():
            logger.debug("CodeCarbon not installed, carbon tracking unavailable")
            return None, None, None

        # Create tracker with settings
        tracker = create_tracker(
            country_iso_code=carbon_settings.country_iso_code,
            region=carbon_settings.region,
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
                report_interval_s,
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

        # Emit final carbon event
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
            )
            event_writer.write_event(event)
        finally:
            event_writer.close()

        # Record to carbon storage
        from experimaestro.carbon.storage import CarbonRecord, CarbonStorage

        storage = CarbonStorage(workspace_path)
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
        storage.write_record(record)

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

        # Carbon tracking state
        self._carbon_tracker = None
        self._carbon_reporter_thread = None
        self._carbon_stop_event = None
        self._job_start_time: datetime | None = None

    def cleanup(self):
        if not self.cleaned:
            self.cleaned = True
            logger.info("Cleaning up")
            rmfile(self.pidfile)

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
                    )
                self._carbon_tracker = None

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

    def run(self):
        atexit.register(self.cleanup)
        sigterm_handler = signal.signal(signal.SIGTERM, self.handle_error)
        sigint_handler = signal.signal(signal.SIGINT, self.handle_error)

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

                # Load workspace path from params
                params_path = workdir / "params.json"
                workspace_path = None
                if params_path.exists():
                    try:
                        with params_path.open() as f:
                            params_data = json.load(f)
                        workspace_path = Path(params_data.get("workspace", ""))
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
                    )

                # Acquire dynamic dependency locks while running the task
                with self.dynamic_locks.dependency_locks():
                    run(workdir / "params.json")

                # ... remove the handlers
                remove_signal_handlers(remove_cleanup=False)

                # Everything went OK
                logger.info("Task ended successfully")
                self.cleanup()
                sys.exit(0)
        except GracefulTimeout as e:
            logger.info("Task requested graceful timeout: %s", e.message)
            self.handle_error(1, None, reason="timeout", message=e.message)

        except Exception:
            logger.exception("Got exception while running")
            self.handle_error(1, None)

        except SystemExit as e:
            if e.code == 0:
                # Normal exit, just create the ".done" file
                self.donepath.touch()

                # ... and finish the exit process
                raise
            else:
                self.handle_error(e.code, None)
