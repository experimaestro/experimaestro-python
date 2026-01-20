"""Unified workspace cleanup functionality

This module provides a centralized cleanup function used by both the workspace
state provider (for automatic cleanup on initialization) and the CLI (for
user-driven cleanup operations).

The cleanup function detects various cleanup scenarios and returns WarningEvent
objects for issues requiring user confirmation. Callbacks for executing cleanup
actions are also provided.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import MockJob
    from experimaestro.scheduler.state_status import WarningEvent
    from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

logger = logging.getLogger("xpm.cleanup")


def perform_workspace_cleanup(
    workspace_path: Path,
    *,
    auto_fix: bool = True,
    provider: "WorkspaceStateProvider | None" = None,
) -> tuple[list["WarningEvent"], dict[str, dict[str, Callable[[], None]]]]:
    """Perform workspace cleanup and return warnings for user confirmation

    This function scans the workspace for cleanup scenarios:
    1. Orphaned experiment events (events without events_count in status.json)
    2. Orphaned job events (events without events_count in status.json)
    3. Stray jobs (running jobs not in latest run of any experiment)
    4. Orphan jobs (finished jobs not in any run)
    5. Orphan partials (partial directories not referenced by any job)

    Args:
        workspace_path: Path to workspace directory
        auto_fix: If True, automatically fix safe issues (consolidate events
                 with events_count present). If False, return warnings for everything.
        provider: Optional WorkspaceStateProvider instance. If None, skip checks
                 that require a provider (stray jobs, orphan jobs, orphan partials).
                 This prevents circular dependency when called from provider initialization.

    Returns:
        Tuple of (warnings, callbacks):
        - warnings: List of WarningEvent objects for issues requiring user confirmation
        - callbacks: Dict mapping warning_key to {action_key -> callback}
    """

    warnings: list[WarningEvent] = []
    callbacks: dict[str, dict[str, Callable[[], None]]] = {}

    workspace_path = Path(workspace_path).resolve()

    # 1. Check for orphaned experiment events
    experiment_warnings, experiment_callbacks = _check_orphaned_experiment_events(
        workspace_path, auto_fix=auto_fix
    )
    warnings.extend(experiment_warnings)
    callbacks.update(experiment_callbacks)

    # 2. Check for orphaned job events
    job_warnings, job_callbacks = _check_orphaned_job_events(
        workspace_path, auto_fix=auto_fix
    )
    warnings.extend(job_warnings)
    callbacks.update(job_callbacks)

    # Skip provider-dependent checks if no provider given (avoids circular dependency)
    if provider is not None:
        # 3. Check for stray jobs (running jobs not in latest run)
        stray_warnings, stray_callbacks = _check_stray_jobs(provider)
        warnings.extend(stray_warnings)
        callbacks.update(stray_callbacks)

        # 4. Check for orphan jobs (finished jobs not in any run)
        orphan_warnings, orphan_callbacks = _check_orphan_jobs(provider)
        warnings.extend(orphan_warnings)
        callbacks.update(orphan_callbacks)

        # 5. Check for orphan partials
        partial_warnings, partial_callbacks = _check_orphan_partials(provider)
        warnings.extend(partial_warnings)
        callbacks.update(partial_callbacks)

    return warnings, callbacks


def _check_orphaned_experiment_events(
    workspace_path: Path, auto_fix: bool
) -> tuple[list["WarningEvent"], dict[str, dict[str, Callable[[], None]]]]:
    """Check for orphaned experiment events

    Events are orphaned when:
    - Event files exist in .events/experiments/{experiment_id}/
    - status.json exists but has NO events_count field

    If auto_fix=True, events WITH events_count are consolidated automatically.

    Returns:
        Tuple of (warnings, callbacks)
    """
    from experimaestro.locking import OrphanedEventsError
    from experimaestro.scheduler.state_status import WarningEvent
    import filelock

    warnings: list[WarningEvent] = []
    callbacks: dict[str, dict[str, Callable[[], None]]] = {}

    experiments_dir = workspace_path / ".events" / "experiments"
    if not experiments_dir.exists():
        return warnings, callbacks

    for exp_events_dir in experiments_dir.iterdir():
        if not exp_events_dir.is_dir():
            continue

        experiment_id = exp_events_dir.name

        # Find event files
        event_files = list(exp_events_dir.glob("events-*.jsonl"))
        if not event_files:
            continue  # No orphaned events

        # Get current run_id from symlink
        symlink = exp_events_dir / "current"
        if not symlink.is_symlink():
            logger.debug(f"No 'current' symlink for experiment {experiment_id}")
            continue

        try:
            run_dir = symlink.resolve()
            run_id = run_dir.name
        except OSError:
            logger.warning(
                f"Could not resolve 'current' symlink for experiment {experiment_id}"
            )
            continue

        # Check status.json for events_count
        status_path = run_dir / "status.json"
        events_count = None

        if status_path.exists():
            try:
                with status_path.open("r") as f:
                    exp_data = json.load(f)
                events_count = exp_data.get("events_count")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read status.json for {experiment_id}: {e}")

        # Try to acquire experiment lock
        experiment_base = run_dir.parent
        lock_path = experiment_base / "lock"

        try:
            lock = filelock.FileLock(lock_path, timeout=0.1)
            lock.acquire()
        except filelock.Timeout:
            # Experiment is locked (running), skip
            logger.debug(
                f"Experiment {experiment_id} is locked (running), skipping cleanup"
            )
            continue

        try:
            if events_count is not None and auto_fix:
                # Auto-fix: consolidate events with events_count
                _consolidate_experiment_events_with_count(
                    workspace_path, experiment_id, run_dir, exp_events_dir, event_files
                )
            elif events_count is None:
                # Orphaned events without events_count - needs user confirmation
                # Raise OrphanedEventsError to create warning
                try:
                    from experimaestro.scheduler.state_provider import MockExperiment

                    # Load experiment
                    exp = MockExperiment.from_disk(run_dir, workspace_path)
                    if exp is not None:
                        exp._cleanup_experiment_event_files()
                except OrphanedEventsError as e:
                    # Create warning event
                    warning_event = WarningEvent(
                        experiment_id=e.context.get("experiment_id", experiment_id),
                        run_id=e.context.get("run_id", run_id),
                        warning_key=e.warning_key,
                        description=e.description,
                        actions=e.actions,
                        context=e.context,
                    )
                    warnings.append(warning_event)
                    callbacks[e.warning_key] = e.callbacks
        finally:
            lock.release()

    return warnings, callbacks


def _consolidate_experiment_events_with_count(
    workspace_path: Path,
    experiment_id: str,
    run_dir: Path,
    exp_events_dir: Path,
    event_files: list[Path],
) -> None:
    """Consolidate experiment events when events_count is present"""
    from experimaestro.scheduler.state_provider import MockExperiment

    # Load experiment
    exp = MockExperiment.from_disk(run_dir, workspace_path)
    if exp is None:
        return

    # Use the experiment's own cleanup method
    try:
        exp._cleanup_experiment_event_files()

        # Write status.json without events_count field
        status_dict = exp.state_dict()
        status_dict.pop("events_count", None)

        status_path = run_dir / "status.json"
        status_path.write_text(json.dumps(status_dict, indent=2))

        logger.info(
            f"Consolidated experiment {experiment_id}, status={exp.status.value}"
        )
    except Exception as e:
        logger.warning(f"Failed to consolidate experiment {experiment_id}: {e}")


def _check_orphaned_job_events(
    workspace_path: Path, auto_fix: bool
) -> tuple[list["WarningEvent"], dict[str, dict[str, Callable[[], None]]]]:
    """Check for orphaned job events

    Events are orphaned when:
    - Event files exist in .events/jobs/{task_id}/event-{job_id}-*.jsonl
    - status.json exists but has NO events_count field (or events_count is None)

    If auto_fix=True, events WITH events_count are consolidated automatically.

    Returns:
        Tuple of (warnings, callbacks)
    """

    warnings: list[WarningEvent] = []
    callbacks: dict[str, dict[str, Callable[[], None]]] = {}

    jobs_dir = workspace_path / ".events" / "jobs"
    if not jobs_dir.exists():
        return warnings, callbacks

    for task_dir in jobs_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_id = task_dir.name
        job_event_files = list(task_dir.glob("event-*-*.jsonl"))

        # Group files by job_id
        job_files_map: dict[str, list[Path]] = {}
        for event_file in job_event_files:
            # Extract job_id from filename: event-{job_id}-{count}.jsonl
            filename = event_file.name
            parts = filename.split("-")
            if len(parts) >= 3:
                job_id = parts[1]
                if job_id not in job_files_map:
                    job_files_map[job_id] = []
                job_files_map[job_id].append(event_file)

        # Consolidate events for each job
        for job_id, files in job_files_map.items():
            job_path = workspace_path / "jobs" / task_id / job_id
            if not job_path.exists():
                # Job directory doesn't exist, just delete orphaned event files
                if auto_fix:
                    for event_file in files:
                        try:
                            event_file.unlink()
                            logger.debug(f"Deleted orphaned event file: {event_file}")
                        except OSError:
                            pass
                continue

            # Load job from disk
            from experimaestro.scheduler.state_provider import MockJob

            job = MockJob.from_disk(
                job_path=job_path,
                task_id=task_id,
                job_id=job_id,
                workspace_path=workspace_path,
            )

            if job.events_count is None:
                # Already consolidated, just delete event files if auto_fix
                if auto_fix:
                    for event_file in files:
                        try:
                            event_file.unlink()
                            logger.debug(
                                f"Deleted already-consolidated event file: {event_file}"
                            )
                        except OSError:
                            pass
            elif auto_fix:
                # Auto-consolidate events with events_count
                _consolidate_job_events(workspace_path, task_id, job_id, job, job_path)

    return warnings, callbacks


def _consolidate_job_events(
    workspace_path: Path,
    task_id: str,
    job_id: str,
    job: "MockJob",
    job_path: Path,
) -> None:
    """Consolidate orphaned job event files"""
    # Use the job's own cleanup method
    job._cleanup_event_files()

    # Write updated status.json without events_count
    status_dict = job.state_dict()
    status_dict.pop("events_count", None)

    status_path = job_path / ".experimaestro" / "status.json"
    try:
        status_path.write_text(json.dumps(status_dict, indent=2))
        logger.debug(f"Consolidated and updated status.json for job {task_id}/{job_id}")
    except OSError as e:
        logger.warning(f"Failed to write status.json for {task_id}/{job_id}: {e}")


def _check_stray_jobs(
    provider: "WorkspaceStateProvider",
) -> tuple[list["WarningEvent"], dict[str, dict[str, Callable[[], None]]]]:
    """Check for stray jobs (running jobs not in latest run of any experiment)

    Returns:
        Tuple of (warnings, callbacks)
    """
    from experimaestro.scheduler.state_status import WarningEvent

    warnings: list[WarningEvent] = []
    callbacks: dict[str, dict[str, Callable[[], None]]] = {}

    # Get stray jobs from the provider
    stray_jobs = provider.get_stray_jobs()

    if not stray_jobs:
        return warnings, callbacks

    # Create warning event for stray jobs
    warning_key = f"stray_jobs_{len(stray_jobs)}"
    description = f"Found {len(stray_jobs)} stray running job(s) not in any active experiment.\n\n"
    description += (
        "Stray jobs occur when an experiment is relaunched with different parameters.\n"
    )
    description += "These jobs are still running but are not part of the current experiment runs.\n\n"
    description += "You can either:\n"
    description += "- Kill them (stop the running processes)\n"
    description += "- Leave them running"

    job_list = []
    for job in stray_jobs[:10]:  # Show first 10
        job_list.append(
            {
                "task_id": job.task_id,
                "job_id": job.identifier,
                "state": job.state.name if job.state else "UNKNOWN",
            }
        )
    if len(stray_jobs) > 10:
        job_list.append(
            {
                "task_id": "...",
                "job_id": f"and {len(stray_jobs) - 10} more",
                "state": "",
            }
        )

    context = {
        "title": "Stray Running Jobs Detected",
        "stray_job_count": len(stray_jobs),
        "stray_jobs": job_list,
    }

    # Create callbacks
    def kill_stray_jobs():
        """Kill all stray jobs"""
        killed_count = 0
        for job in stray_jobs:
            try:
                provider.kill_job(job, perform=True)
                killed_count += 1
                logger.info(f"Killed stray job: {job.task_id}/{job.identifier}")
            except Exception as e:
                logger.warning(
                    f"Failed to kill job {job.task_id}/{job.identifier}: {e}"
                )
        logger.info(f"Killed {killed_count}/{len(stray_jobs)} stray jobs")

    actions = {
        "kill": "Kill All Stray Jobs",
        "dismiss": "Leave Running",
    }

    action_callbacks = {
        "kill": kill_stray_jobs,
        "dismiss": lambda: None,  # No-op
    }

    warning_event = WarningEvent(
        experiment_id="",
        run_id="",
        warning_key=warning_key,
        description=description,
        actions=actions,
        context=context,
    )

    warnings.append(warning_event)
    callbacks[warning_key] = action_callbacks

    return warnings, callbacks


def _check_orphan_jobs(
    provider: "WorkspaceStateProvider",
) -> tuple[list["WarningEvent"], dict[str, dict[str, Callable[[], None]]]]:
    """Check for orphan jobs (finished jobs not in any run)

    Returns:
        Tuple of (warnings, callbacks)
    """
    from experimaestro.scheduler.state_status import WarningEvent

    warnings: list[WarningEvent] = []
    callbacks: dict[str, dict[str, Callable[[], None]]] = {}

    # Get orphan jobs from the provider
    all_orphan_jobs = provider.get_orphan_jobs()

    # Filter to only finished jobs (exclude stray/running)
    orphan_jobs = [j for j in all_orphan_jobs if j.state and not j.state.running()]

    if not orphan_jobs:
        return warnings, callbacks

    # Create warning event for orphan jobs
    warning_key = f"orphan_jobs_{len(orphan_jobs)}"
    description = f"Found {len(orphan_jobs)} orphan finished job(s) not in any experiment run.\n\n"
    description += (
        "Orphan jobs are finished jobs that are not referenced by any experiment.\n"
    )
    description += (
        "They may be leftover from deleted experiments or experimental changes.\n\n"
    )
    description += "You can either:\n"
    description += "- Delete them (remove job directories and data)\n"
    description += "- Keep them for reference"

    job_list = []
    for job in orphan_jobs[:10]:  # Show first 10
        job_list.append(
            {
                "task_id": job.task_id,
                "job_id": job.identifier,
                "state": job.state.name if job.state else "UNKNOWN",
            }
        )
    if len(orphan_jobs) > 10:
        job_list.append(
            {
                "task_id": "...",
                "job_id": f"and {len(orphan_jobs) - 10} more",
                "state": "",
            }
        )

    context = {
        "title": "Orphan Finished Jobs Detected",
        "orphan_job_count": len(orphan_jobs),
        "orphan_jobs": job_list,
    }

    # Create callbacks
    def delete_orphan_jobs():
        """Delete all orphan jobs"""
        deleted_count = 0
        for job in orphan_jobs:
            try:
                success, msg = provider.delete_job_safely(job, perform=True)
                if success:
                    deleted_count += 1
                    logger.info(f"Deleted orphan job: {job.task_id}/{job.identifier}")
                else:
                    logger.warning(
                        f"Failed to delete job {job.task_id}/{job.identifier}: {msg}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to delete job {job.task_id}/{job.identifier}: {e}"
                )
        logger.info(f"Deleted {deleted_count}/{len(orphan_jobs)} orphan jobs")

        # Clean up orphan partials after deleting jobs
        provider.cleanup_orphan_partials(perform=True)

    actions = {
        "delete": "Delete All Orphan Jobs",
        "dismiss": "Keep Jobs",
    }

    action_callbacks = {
        "delete": delete_orphan_jobs,
        "dismiss": lambda: None,  # No-op
    }

    warning_event = WarningEvent(
        experiment_id="",
        run_id="",
        warning_key=warning_key,
        description=description,
        actions=actions,
        context=context,
    )

    warnings.append(warning_event)
    callbacks[warning_key] = action_callbacks

    return warnings, callbacks


def _check_orphan_partials(
    provider: "WorkspaceStateProvider",
) -> tuple[list["WarningEvent"], dict[str, dict[str, Callable[[], None]]]]:
    """Check for orphan partial directories

    Returns:
        Tuple of (warnings, callbacks)
    """
    from experimaestro.scheduler.state_status import WarningEvent

    warnings: list[WarningEvent] = []
    callbacks: dict[str, dict[str, Callable[[], None]]] = {}

    # Get orphan partials from the provider
    orphan_paths = provider.cleanup_orphan_partials(perform=False)

    if not orphan_paths:
        return warnings, callbacks

    # Create warning event for orphan partials
    warning_key = f"orphan_partials_{len(orphan_paths)}"
    description = f"Found {len(orphan_paths)} orphan partial director{'y' if len(orphan_paths) == 1 else 'ies'}.\n\n"
    description += "Partial directories are shared checkpoint locations.\n"
    description += (
        "When all jobs using a partial are deleted, the partial becomes orphaned.\n\n"
    )
    description += "You can either:\n"
    description += "- Delete them (free up disk space)\n"
    description += "- Keep them for reference"

    partial_list = []
    for path in orphan_paths[:10]:  # Show first 10
        partial_list.append({"path": str(path)})
    if len(orphan_paths) > 10:
        partial_list.append({"path": f"... and {len(orphan_paths) - 10} more"})

    context = {
        "title": "Orphan Partial Directories Detected",
        "orphan_partial_count": len(orphan_paths),
        "orphan_partials": partial_list,
    }

    # Create callbacks
    def delete_orphan_partials():
        """Delete all orphan partials"""
        deleted_paths = provider.cleanup_orphan_partials(perform=True)
        logger.info(f"Deleted {len(deleted_paths)} orphan partial directories")

    actions = {
        "delete": "Delete All Orphan Partials",
        "dismiss": "Keep Partials",
    }

    action_callbacks = {
        "delete": delete_orphan_partials,
        "dismiss": lambda: None,  # No-op
    }

    warning_event = WarningEvent(
        experiment_id="",
        run_id="",
        warning_key=warning_key,
        description=description,
        actions=actions,
        context=context,
    )

    warnings.append(warning_event)
    callbacks[warning_key] = action_callbacks

    return warnings, callbacks
