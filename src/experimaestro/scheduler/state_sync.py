"""Disk-based state synchronization for workspace database

This module implements synchronization from disk state (marker files) to the
workspace database. It includes locking and throttling mechanisms to prevent
excessive disk scanning and conflicts with running experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import fasteners

from experimaestro.scheduler.state_db import (
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    JobTagModel,
    WorkspaceSyncMetadata,
)

logger = logging.getLogger("xpm.state_sync")


def read_jobs_jsonl(exp_dir: Path) -> Dict[str, Dict]:
    """Read jobs.jsonl file and return a mapping of job_id -> record

    Args:
        exp_dir: Path to the experiment directory

    Returns:
        Dictionary mapping job_id to record (with tags, task_id, timestamp)
    """
    jobs_jsonl_path = exp_dir / "jobs.jsonl"
    job_records = {}

    if not jobs_jsonl_path.exists():
        logger.debug("No jobs.jsonl found in %s", exp_dir)
        return job_records

    try:
        with jobs_jsonl_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    job_id = record.get("job_id")
                    if job_id:
                        job_records[job_id] = record
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse line in jobs.jsonl: %s", e)
    except Exception as e:
        logger.warning("Failed to read jobs.jsonl from %s: %s", jobs_jsonl_path, e)

    logger.debug("Read %d job records from jobs.jsonl", len(job_records))
    return job_records


def acquire_sync_lock(
    workspace_path: Path, blocking: bool = True
) -> Optional[fasteners.InterProcessLock]:
    """Acquire exclusive lock for workspace synchronization

    Args:
        workspace_path: Path to the workspace directory
        blocking: If True, wait for lock; if False, return None if unavailable

    Returns:
        Lock object if acquired, None if not acquired (only in non-blocking mode)
    """
    lock_path = workspace_path / ".sync.lock"
    lock = fasteners.InterProcessLock(str(lock_path))

    if lock.acquire(blocking=blocking):
        logger.debug("Acquired sync lock: %s", lock_path)
        return lock
    else:
        logger.debug("Could not acquire sync lock (already held): %s", lock_path)
        return None


def get_last_sync_time(workspace_path: Path) -> Optional[datetime]:
    """Get the timestamp of the last successful sync

    Args:
        workspace_path: Path to workspace directory

    Returns:
        datetime of last sync, or None if never synced
    """
    try:
        from .state_provider import WorkspaceStateProvider

        provider = WorkspaceStateProvider.get_instance(
            workspace_path, read_only=True, sync_on_start=False
        )
        with provider.workspace_db.bind_ctx([WorkspaceSyncMetadata]):
            metadata = WorkspaceSyncMetadata.get_or_none(
                WorkspaceSyncMetadata.id == "workspace"
            )
            if metadata and metadata.last_sync_time:
                return metadata.last_sync_time
    except Exception as e:
        logger.warning("Failed to get last sync time: %s", e)

    return None


def update_last_sync_time(workspace_path: Path) -> None:
    """Update the last sync timestamp to now

    Args:
        workspace_path: Path to workspace directory
    """
    try:
        from .state_provider import WorkspaceStateProvider

        provider = WorkspaceStateProvider.get_instance(
            workspace_path, read_only=False, sync_on_start=False
        )
        with provider.workspace_db.bind_ctx([WorkspaceSyncMetadata]):
            WorkspaceSyncMetadata.insert(
                id="workspace", last_sync_time=datetime.now()
            ).on_conflict(
                conflict_target=[WorkspaceSyncMetadata.id],
                update={WorkspaceSyncMetadata.last_sync_time: datetime.now()},
            ).execute()
            logger.debug("Updated last sync time")
    except Exception as e:
        logger.error("Failed to update last sync time: %s", e)


def should_sync(
    workspace_path: Path, min_interval_minutes: int = 5
) -> Tuple[bool, Optional[fasteners.InterProcessLock]]:
    """Determine if sync should be performed based on locking and timing

    Args:
        workspace_path: Path to workspace directory
        min_interval_minutes: Minimum minutes between syncs (default: 5)

    Returns:
        Tuple of (should_sync: bool, lock: Optional[Lock])
        If should_sync is True, lock is acquired and must be released after sync
        If should_sync is False, lock is None
    """
    # Try to acquire exclusive lock (non-blocking)
    lock = acquire_sync_lock(workspace_path, blocking=False)
    if lock is None:
        # Other experiments running - skip sync
        logger.info("Skipping sync: other experiments are running")
        return False, None

    # Check last sync time
    last_sync_time = get_last_sync_time(workspace_path)
    if last_sync_time is None:
        # First sync ever
        logger.info("Performing first sync")
        return True, lock

    time_since_last_sync = datetime.now() - last_sync_time
    if time_since_last_sync.total_seconds() > min_interval_minutes * 60:
        # Enough time has passed
        logger.info(
            "Performing sync (%.1f minutes since last sync)",
            time_since_last_sync.total_seconds() / 60,
        )
        return True, lock

    # Recently synced, skip
    logger.info(
        "Skipping sync: last sync was %.1f minutes ago (threshold: %d minutes)",
        time_since_last_sync.total_seconds() / 60,
        min_interval_minutes,
    )
    lock.release()
    return False, None


def scan_job_state_from_disk(job_path: Path, scriptname: str) -> Optional[Dict]:
    """Scan a job directory to determine state from disk files

    Reads job state from .xpm_metadata.json (primary) or marker files (fallback).

    Args:
        job_path: Path to the job directory
        scriptname: Name of the script (for marker file names)

    Returns:
        Dictionary with job state information, or None if no state found:
        {
            'job_id': str,
            'task_id': str,
            'state': str,
            'submitted_time': float or None,
            'started_time': float or None,
            'ended_time': float or None,
            'exit_code': int or None,
            'failure_reason': str or None,
            'retry_count': int,
            'process': dict or None  # Process spec from metadata
        }
    """
    # Try reading .xpm_metadata.json first (primary source)
    metadata_path = job_path / ".xpm_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r") as f:
                metadata = json.load(f)

            logger.debug("Read metadata from %s", metadata_path)
            return {
                "job_id": metadata.get("job_id"),
                "task_id": metadata.get("task_id"),
                "state": metadata.get("state", "unscheduled"),
                "submitted_time": metadata.get("submitted_time"),
                "started_time": metadata.get("started_time"),
                "ended_time": metadata.get("ended_time"),
                "exit_code": metadata.get("exit_code"),
                "failure_reason": metadata.get("failure_reason"),
                "retry_count": metadata.get("retry_count", 0),
                "process": metadata.get("process"),  # Process spec with launcher info
            }
        except Exception as e:
            logger.warning("Failed to read metadata from %s: %s", metadata_path, e)
            # Fall through to marker file fallback

    # Fallback: Infer from marker files and params.json
    try:
        # Check marker files for state
        done_file = job_path / f"{scriptname}.done"
        failed_file = job_path / f"{scriptname}.failed"
        pid_file = job_path / f"{scriptname}.pid"

        state = "unscheduled"
        exit_code = None

        if done_file.is_file():
            state = "done"
            exit_code = 0
        elif failed_file.is_file():
            state = "error"
            try:
                exit_code = int(failed_file.read_text().strip())
            except (ValueError, OSError):
                exit_code = 1
        elif pid_file.is_file():
            state = "running"

        # Use directory structure to infer job_id and task_id
        # job_path structure: {workspace}/jobs/{task_id}/{hash}/
        if job_path.parent and job_path.parent.parent:
            job_id = job_path.name  # Hash hex
            task_id = job_path.parent.name

        return {
            "job_id": job_id,
            "task_id": task_id,
            "state": state,
            "submitted_time": None,  # Not available from marker files
            "started_time": None,
            "ended_time": None,
            "exit_code": exit_code,
            "failure_reason": None,
            "retry_count": 0,
            "process": None,
        }

    except Exception as e:
        logger.warning("Failed to scan job state from %s: %s", job_path, e)
        return None


def sync_workspace_from_disk(  # noqa: C901
    workspace_path: Path,
    write_mode: bool = True,
    force: bool = False,
    blocking: bool = True,
    sync_interval_minutes: int = 5,
) -> None:
    """Synchronize workspace database from disk state

    Scans job directories and experiment symlinks to update the database.
    Uses exclusive locking and time-based throttling to prevent conflicts.

    Args:
        workspace_path: Path to workspace directory
        write_mode: If True, update database; if False, dry-run mode
        force: If True, bypass time throttling (still requires lock)
        blocking: If True, wait for lock; if False, fail if lock unavailable
        sync_interval_minutes: Minimum minutes between syncs (default: 5)

    Raises:
        RuntimeError: If lock unavailable in non-blocking mode
    """
    # Normalize path
    if not isinstance(workspace_path, Path):
        workspace_path = Path(workspace_path)
    workspace_path = workspace_path.absolute()

    # Check if sync should proceed (unless force=True)
    if not force:
        should_proceed, lock = should_sync(workspace_path, sync_interval_minutes)
        if not should_proceed:
            return
    else:
        # Force mode: skip time check but still require lock
        lock = acquire_sync_lock(workspace_path, blocking=blocking)
        if lock is None:
            if blocking:
                raise RuntimeError("Failed to acquire sync lock in blocking mode")
            else:
                raise RuntimeError("Sync lock unavailable (other experiments running)")

    try:
        logger.info("Starting workspace sync from disk: %s", workspace_path)

        # Get the workspace state provider to access the database
        from .state_provider import WorkspaceStateProvider

        provider = WorkspaceStateProvider.get_instance(
            workspace_path,
            read_only=not write_mode,
            sync_on_start=False,  # Don't sync recursively
        )

        experiments_found = 0
        runs_found = 0
        jobs_scanned = 0
        jobs_updated = 0

        # Use database context binding for all queries
        with provider.workspace_db.bind_ctx(
            [
                ExperimentModel,
                ExperimentRunModel,
                JobModel,
                JobTagModel,
                WorkspaceSyncMetadata,
            ]
        ):
            # Scan experiments directory - this is the source of truth
            xp_dir = workspace_path / "xp"

            if not xp_dir.exists():
                logger.info("No experiments directory found")
                return

            for exp_dir in xp_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                experiment_id = exp_dir.name
                experiments_found += 1

                # Read jobs.jsonl to get tags for each job
                job_records = read_jobs_jsonl(exp_dir)

                if write_mode:
                    # Ensure experiment exists in database
                    ExperimentModel.insert(
                        experiment_id=experiment_id, workdir_path=str(exp_dir)
                    ).on_conflict(
                        conflict_target=[ExperimentModel.experiment_id],
                        update={ExperimentModel.workdir_path: str(exp_dir)},
                    ).execute()

                # Determine or create run_id for experiment
                existing_runs = list(
                    ExperimentRunModel.select()
                    .where(ExperimentRunModel.experiment_id == experiment_id)
                    .order_by(ExperimentRunModel.started_at.desc())
                )

                if existing_runs:
                    # Use the most recent run as current
                    current_run_id = existing_runs[0].run_id
                    runs_found += len(existing_runs)
                else:
                    # Create initial run
                    current_run_id = "initial"
                    runs_found += 1

                    if write_mode:
                        ExperimentRunModel.insert(
                            experiment_id=experiment_id,
                            run_id=current_run_id,
                            status="active",
                        ).on_conflict_ignore().execute()

                        # Update experiment's current_run_id
                        ExperimentModel.update(current_run_id=current_run_id).where(
                            ExperimentModel.experiment_id == experiment_id
                        ).execute()

                logger.debug(
                    "Experiment %s: current_run_id=%s", experiment_id, current_run_id
                )

                # Scan jobs linked from this experiment
                jobs_dir = exp_dir / "jobs"
                if not jobs_dir.exists():
                    continue

                # Find all symlinks in experiment jobs directory
                for symlink_path in jobs_dir.rglob("*"):
                    if not symlink_path.is_symlink():
                        continue

                    jobs_scanned += 1

                    # Try to resolve symlink to actual job directory
                    try:
                        job_path = symlink_path.resolve()
                        job_exists = job_path.is_dir()
                    except (OSError, RuntimeError):
                        # Broken symlink
                        job_path = None
                        job_exists = False

                    # Read job state from disk if job exists
                    job_state = None
                    if job_exists and job_path:
                        # Try to determine scriptname
                        scriptname = None
                        for suffix in [".done", ".failed", ".pid"]:
                            for file in job_path.glob(f"*{suffix}"):
                                scriptname = file.name[: -len(suffix)]
                                break
                            if scriptname:
                                break

                        if not scriptname:
                            # Infer from job_path name
                            scriptname = job_path.name

                        job_state = scan_job_state_from_disk(job_path, scriptname)

                    # If we couldn't read state, create minimal entry from symlink
                    if not job_state:
                        # Extract job_id and task_id from symlink structure
                        # Symlink structure: xp/{exp}/jobs/{task_id}/{hash}
                        parts = symlink_path.parts
                        try:
                            jobs_idx = parts.index("jobs")
                            if jobs_idx + 2 < len(parts):
                                task_id = parts[jobs_idx + 1]
                                job_id = parts[jobs_idx + 2]
                            else:
                                # Fallback
                                task_id = "unknown"
                                job_id = symlink_path.name
                        except (ValueError, IndexError):
                            task_id = "unknown"
                            job_id = symlink_path.name

                        job_state = {
                            "job_id": job_id,
                            "task_id": task_id,
                            "state": "unscheduled",  # Unknown state for broken links
                            "submitted_time": None,
                            "started_time": None,
                            "ended_time": None,
                            "exit_code": None,
                            "failure_reason": None,
                            "retry_count": 0,
                            "process": None,
                        }
                        job_path_str = str(job_path) if job_path else str(symlink_path)
                    else:
                        job_path_str = str(job_path)

                    # Update database
                    if write_mode and job_state and job_state["job_id"]:
                        JobModel.insert(
                            job_id=job_state["job_id"],
                            experiment_id=experiment_id,
                            run_id=current_run_id,
                            task_id=job_state["task_id"],
                            locator="",  # Not available from disk
                            path=job_path_str,
                            state=job_state["state"],
                            failure_reason=job_state.get("failure_reason"),
                            submitted_time=job_state.get("submitted_time"),
                            started_time=job_state.get("started_time"),
                            ended_time=job_state.get("ended_time"),
                            progress="[]",
                        ).on_conflict(
                            conflict_target=[
                                JobModel.job_id,
                                JobModel.experiment_id,
                                JobModel.run_id,
                            ],
                            update={
                                JobModel.state: job_state["state"],
                                JobModel.failure_reason: job_state.get(
                                    "failure_reason"
                                ),
                                JobModel.submitted_time: job_state.get(
                                    "submitted_time"
                                ),
                                JobModel.started_time: job_state.get("started_time"),
                                JobModel.ended_time: job_state.get("ended_time"),
                                JobModel.path: job_path_str,
                            },
                        ).execute()

                        jobs_updated += 1

                        # Sync tags from jobs.jsonl
                        job_id = job_state["job_id"]
                        if job_id in job_records:
                            tags = job_records[job_id].get("tags", {})
                            if tags:
                                # Delete existing tags for this job+experiment+run
                                JobTagModel.delete().where(
                                    (JobTagModel.job_id == job_id)
                                    & (JobTagModel.experiment_id == experiment_id)
                                    & (JobTagModel.run_id == current_run_id)
                                ).execute()

                                # Insert new tags
                                for tag_key, tag_value in tags.items():
                                    JobTagModel.insert(
                                        job_id=job_id,
                                        experiment_id=experiment_id,
                                        run_id=current_run_id,
                                        tag_key=tag_key,
                                        tag_value=str(tag_value),
                                    ).on_conflict_ignore().execute()

                                logger.debug(
                                    "Synced %d tags for job %s", len(tags), job_id
                                )

                        logger.debug(
                            "Synced job %s for experiment %s run %s: state=%s",
                            job_state["job_id"],
                            experiment_id,
                            current_run_id,
                            job_state["state"],
                        )

            logger.info(
                "Sync complete: %d experiments, %d runs, %d jobs scanned, %d jobs updated",
                experiments_found,
                runs_found,
                jobs_scanned,
                jobs_updated,
            )

            # Update last sync time if in write mode
            if write_mode:
                update_last_sync_time(workspace_path)

    finally:
        # Always release lock
        if lock:
            lock.release()
            logger.debug("Released sync lock")
