"""File-based progress tracking system for experimaestro tasks."""

import threading
from pathlib import Path
from typing import Optional


class FileBasedProgressReporter:
    """File-based progress reporter that writes to job event files.

    Writes JobProgressEvent objects to:
    workspace/.events/jobs/{task_id}/event-{job_id}-*.jsonl

    These files are watched by the scheduler's EventReader to forward
    progress updates to listeners.
    """

    def __init__(self, task_path: Path):
        """Initialize file-based progress reporter

        Args:
            task_path: Path to the task directory (workspace/jobs/task_id/job_id/)
        """
        from experimaestro.scheduler.state_status import JobEventWriter

        self.task_path = task_path
        # Extract workspace, task_id, and job_id from task_path
        # task_path is typically: workspace/jobs/task_id/job_id/
        self.job_id = task_path.name
        self.task_id = task_path.parent.name
        workspace_path = task_path.parent.parent.parent

        # Create event writer for this job's events
        # Pass job_path for permanent storage of events
        self.event_writer = JobEventWriter(
            workspace_path, self.task_id, self.job_id, 0, job_path=task_path
        )
        self.current_progress: dict[int, tuple[float | None, str | None]] = {}
        self.lock = threading.Lock()

    def set_progress(self, progress: float, level: int = 0, desc: Optional[str] = None):
        """Set progress for a specific level

        Args:
            progress: Progress value between 0.0 and 1.0
            level: Progress level (0 is top level)
            desc: Optional description
        """
        from experimaestro.scheduler.state_status import JobProgressEvent

        with self.lock:
            # Check if progress has changed significantly
            current = self.current_progress.get(level, (None, None))
            if (
                current[0] is None
                or abs(progress - current[0]) > 0.01
                or desc != current[1]
            ):
                self.current_progress[level] = (progress, desc)

                # Write to event file (EventWriter handles periodic flushing)
                event = JobProgressEvent(
                    job_id=self.job_id,
                    level=level,
                    progress=progress,
                    desc=desc,
                )
                self.event_writer.write_event(event)

    def start_of_job(self):
        """Start of job notification - called when job execution begins"""
        from experimaestro.scheduler.state_status import JobStateChangedEvent
        from datetime import datetime

        with self.lock:
            # Write JobStateChangedEvent with state="running" to event file
            event = JobStateChangedEvent(
                job_id=self.job_id,
                state="running",
                started_time=datetime.now().isoformat(),
            )
            self.event_writer.write_event(event)
            self.event_writer.flush()

    def eoj(self):
        """End of job notification"""
        from experimaestro.scheduler.state_status import JobStateChangedEvent
        from datetime import datetime

        with self.lock:
            # Write JobStateChangedEvent with state="done" to event file
            event = JobStateChangedEvent(
                job_id=self.job_id,
                state="done",
                ended_time=datetime.now().isoformat(),
            )
            self.event_writer.write_event(event)
            self.event_writer.flush()

            # Archive events from .events/ to job directory
            self.event_writer.archive_events()
