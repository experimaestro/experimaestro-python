"""Orphan jobs tab widget for the TUI

Displays orphan jobs: Jobs on disk not referenced by any experiment.
Running orphan jobs (stray) are highlighted differently and can be killed.
Non-running orphan jobs can be deleted.
"""

import logging
from typing import Optional
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import DataTable, Static, Button
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tui.utils import get_status_icon
from experimaestro.tui.dialogs import DeleteConfirmScreen, KillConfirmScreen
from experimaestro.tui.messages import SizeCalculated

logger = logging.getLogger("xpm.tui.orphan_jobs")


class OrphanJobsTab(Vertical):
    """Tab widget for viewing and managing orphan jobs

    Orphan jobs: Jobs on disk not referenced by any experiment.
    - Running orphan jobs (stray) are shown in yellow and can be killed (ctrl+k)
    - Non-running orphan jobs can be deleted (ctrl+d)
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+d", "delete_selected", "Delete", show=False),
        Binding("ctrl+k", "kill_selected", "Kill", show=False),
        Binding("T", "sort_by_task", "Sort Task", show=False),
        Binding("Z", "sort_by_size", "Sort Size", show=False),
    ]

    _size_cache: dict = {}  # Class-level cache (formatted strings)
    _size_bytes_cache: dict = {}  # Class-level cache (raw bytes for sorting)

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.orphan_jobs = []  # All orphan jobs
        self._pending_jobs = []  # Jobs waiting for size calculation
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", id="orphan-warning", classes="warning-banner hidden")
        with Horizontal(id="orphan-controls", classes="controls-bar"):
            yield Button("Refresh", id="orphan-refresh-btn")
            yield Button("Kill All", id="orphan-kill-all-btn", variant="error")
            yield Button("Delete All", id="orphan-delete-all-btn", variant="warning")
        yield DataTable(id="orphan-table", cursor_type="row")
        yield Static("", id="orphan-job-info")

    def on_mount(self) -> None:
        """Initialize the orphan jobs table"""
        table = self.query_one("#orphan-table", DataTable)
        table.add_column("", key="status", width=3)
        table.add_column("Job ID", key="job_id", width=10)
        table.add_column("Task", key="task")
        table.add_column("Size", key="size", width=10)
        self.refresh_orphan_jobs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "orphan-refresh-btn":
            self.action_refresh()
        elif event.button.id == "orphan-kill-all-btn":
            self.action_kill_all()
        elif event.button.id == "orphan-delete-all-btn":
            self.action_delete_all()

    def action_sort_by_task(self) -> None:
        """Sort by task name"""
        if self._sort_column == "task":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "task"
            self._sort_reverse = False
        self._rebuild_table()
        order = "desc" if self._sort_reverse else "asc"
        self.notify(f"Sorted by task ({order})", severity="information")

    def action_sort_by_size(self) -> None:
        """Sort by size"""
        if self._sort_column == "size":
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = "size"
            self._sort_reverse = True  # Default: largest first
        self._rebuild_table()
        order = "largest first" if self._sort_reverse else "smallest first"
        self.notify(f"Sorted by size ({order})", severity="information")

    def refresh_orphan_jobs(self) -> None:
        """Refresh the orphan jobs list"""
        # Check if remote provider
        if self.state_provider.is_remote:
            self._show_warning(
                "Orphan job detection not available for remote workspaces"
            )
            return

        # Get all orphan jobs (only those with existing folders)
        all_orphans = self.state_provider.get_orphan_jobs()
        self.orphan_jobs = [j for j in all_orphans if j.path and j.path.exists()]

        # Count running jobs
        running_count = sum(
            1 for j in self.orphan_jobs if j.state and j.state.running()
        )

        # Update warning based on scheduler status
        self._update_scheduler_warning(running_count)

        # Update tab title in parent app
        self._update_tab_title()

        # Collect jobs needing size calculation
        self._pending_jobs = [
            j for j in self.orphan_jobs if j.identifier not in self._size_cache
        ]

        # Rebuild table
        self._rebuild_table()

        # Start calculating sizes
        if self._pending_jobs:
            self._calculate_next_size()

    def _update_scheduler_warning(self, running_count: int) -> None:
        """Update warning banner based on scheduler status"""
        warning = self.query_one("#orphan-warning", Static)

        # Check if any experiments are running (ended_at is None means still running)
        running_experiments = [
            e
            for e in self.state_provider.get_experiments()
            if e.run_id and getattr(e, "ended_at", None) is None
        ]

        if running_experiments or self.state_provider.is_live:
            warning.update(
                "WARNING: At least one experiment is running. "
                "Killing stray jobs or deleting orphans may cause issues!"
            )
            warning.remove_class("hidden")
        elif running_count > 0:
            warning.update(
                f"{running_count} orphan jobs are still running (stray). "
                "Use ctrl+k to kill them."
            )
            warning.remove_class("hidden")
        else:
            warning.add_class("hidden")

    def _show_warning(self, message: str) -> None:
        """Show a warning message"""
        warning = self.query_one("#orphan-warning", Static)
        warning.update(f"{message}")
        warning.remove_class("hidden")

    def _update_tab_title(self) -> None:
        """Update the tab title with orphan job count"""
        try:
            self.app.update_orphan_tab_title()
        except Exception:
            pass

    @property
    def orphan_count(self) -> int:
        """Number of all orphan jobs"""
        return len(self.orphan_jobs)

    @property
    def running_count(self) -> int:
        """Number of running orphan jobs (stray)"""
        return sum(1 for j in self.orphan_jobs if j.state and j.state.running())

    @property
    def finished_count(self) -> int:
        """Number of non-running orphan jobs"""
        return len(self.orphan_jobs) - self.running_count

    def _get_sorted_jobs(self):
        """Return jobs sorted by current sort column"""
        jobs = self.orphan_jobs[:]
        if self._sort_column == "task":
            jobs.sort(key=lambda j: j.task_id or "", reverse=self._sort_reverse)
        elif self._sort_column == "size":
            # Sort by raw bytes, jobs not in cache go to end
            jobs.sort(
                key=lambda j: self._size_bytes_cache.get(j.identifier, -1),
                reverse=self._sort_reverse,
            )
        return jobs

    def _rebuild_table(self) -> None:
        """Rebuild the table with current sort order"""
        from rich.text import Text

        table = self.query_one("#orphan-table", DataTable)
        table.clear()

        for job in self._get_sorted_jobs():
            failure_reason = getattr(job, "failure_reason", None)
            transient = getattr(job, "transient", None)
            status_icon = get_status_icon(
                job.state.name if job.state else "unknown", failure_reason, transient
            )

            # Use different styling for running vs finished jobs
            is_running = job.state and job.state.running()
            if is_running:
                # Running jobs (stray) in yellow/bold
                job_id_text = Text(job.identifier[:7], style="bold yellow")
                task_text = Text(job.task_id or "", style="yellow")
            else:
                # Finished jobs in normal style
                job_id_text = Text(job.identifier[:7])
                task_text = Text(job.task_id or "")

            if job.identifier in self._size_cache:
                size_text = self._size_cache[job.identifier]
            else:
                size_text = "waiting"

            table.add_row(
                status_icon,
                job_id_text,
                task_text,
                size_text,
                key=job.identifier,
            )

    def _calculate_next_size(self) -> None:
        """Calculate size for the next pending job using a worker"""
        if not self._pending_jobs:
            return

        job = self._pending_jobs.pop(0)
        # Update to "calc..."
        self._update_size_cell(job.identifier, "calc...")
        # Run calculation in worker thread
        self.run_worker(
            self._calc_size_worker(job.identifier, job.path),
            thread=True,
        )

    async def _calc_size_worker(self, job_id: str, path):
        """Worker to calculate folder size"""
        size_bytes = await self._get_folder_size_async(path)
        size_str = self._format_size(size_bytes)
        self._size_cache[job_id] = size_str
        self._size_bytes_cache[job_id] = size_bytes
        self.post_message(SizeCalculated(job_id, size_str, size_bytes))

    def on_size_calculated(self, message: SizeCalculated) -> None:
        """Handle size calculation completion"""
        self._size_bytes_cache[message.job_id] = message.size_bytes
        self._update_size_cell(message.job_id, message.size)
        # Calculate next one
        self._calculate_next_size()

    @staticmethod
    async def _get_folder_size_async(path) -> int:
        """Calculate total size of a folder using du command if available"""
        import asyncio
        import shutil
        import sys

        # Try using du command for better performance
        if shutil.which("du"):
            try:
                if sys.platform == "darwin":
                    # macOS: du -sk gives size in KB
                    proc = await asyncio.create_subprocess_exec(
                        "du",
                        "-sk",
                        str(path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode == 0 and stdout:
                        # Output format: "SIZE\tPATH"
                        size_kb = int(stdout.decode().split()[0])
                        return size_kb * 1024
                else:
                    # Linux: du -sb gives size in bytes
                    proc = await asyncio.create_subprocess_exec(
                        "du",
                        "-sb",
                        str(path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode == 0 and stdout:
                        # Output format: "SIZE\tPATH"
                        return int(stdout.decode().split()[0])
            except (asyncio.TimeoutError, ValueError, IndexError, OSError):
                pass  # Fall back to Python implementation

        # Fallback: Python implementation
        return OrphanJobsTab._get_folder_size_sync(path)

    @staticmethod
    def _get_folder_size_sync(path) -> int:
        """Calculate total size of a folder using Python (fallback)"""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except (OSError, PermissionError):
            pass
        return total

    @staticmethod
    def _format_size(size: int) -> str:
        """Format size in human-readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def _update_size_cell(self, job_id: str, value: str = None) -> None:
        """Update the size cell for a job"""
        try:
            table = self.query_one("#orphan-table", DataTable)
            size_text = (
                value if value is not None else self._size_cache.get(job_id, "-")
            )
            table.update_cell(job_id, "size", size_text)
        except Exception:
            pass  # Table may have changed

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show job details when a row is selected"""
        self._update_job_info()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show job details when cursor moves"""
        self._update_job_info()

    def _update_job_info(self) -> None:
        """Update the job info display"""
        table = self.query_one("#orphan-table", DataTable)
        info = self.query_one("#orphan-job-info", Static)

        if table.cursor_row is None:
            info.update("")
            return

        try:
            row_key = list(table.rows.keys())[table.cursor_row]
        except IndexError:
            info.update("")
            return

        if row_key:
            job_id = str(row_key.value)
            job = next((j for j in self.orphan_jobs if j.identifier == job_id), None)
            if job and job.path:
                size = self._size_cache.get(job.identifier, "calculating...")
                state = job.state.name if job.state else "unknown"
                is_running = job.state and job.state.running()
                hint = "(ctrl+k to kill)" if is_running else "(ctrl+d to delete)"
                info.update(
                    f"Path: {job.path}  |  Size: {size}  |  State: {state} {hint}"
                )
            else:
                info.update("")

    def _get_selected_job(self):
        """Get the currently selected job"""
        table = self.query_one("#orphan-table", DataTable)
        if table.cursor_row is None:
            return None

        try:
            row_key = list(table.rows.keys())[table.cursor_row]
        except IndexError:
            return None

        if row_key:
            job_id = str(row_key.value)
            return next((j for j in self.orphan_jobs if j.identifier == job_id), None)
        return None

    def action_refresh(self) -> None:
        """Refresh the orphan jobs list"""
        self.refresh_orphan_jobs()
        self.notify("Refreshed orphan jobs list", severity="information")

    def action_delete_selected(self) -> None:
        """Delete the selected orphan job (if not running)"""
        job = self._get_selected_job()
        if not job:
            return

        if job.state and job.state.running():
            self.notify(
                "Cannot delete a running job - kill it first (ctrl+k)",
                severity="warning",
            )
            return

        self._delete_job(job)

    def action_kill_selected(self) -> None:
        """Kill the selected running orphan job"""
        job = self._get_selected_job()
        if not job:
            return

        if not job.state or not job.state.running():
            self.notify("Job is not running", severity="warning")
            return

        self._kill_job(job)

    def _delete_job(self, job) -> None:
        """Delete a single orphan job with confirmation"""

        def handle_delete(confirmed: bool) -> None:
            if confirmed:
                success, msg = self.state_provider.delete_job_safely(job)
                if success:
                    self.notify(msg, severity="information")
                    self.refresh_orphan_jobs()
                else:
                    self.notify(msg, severity="error")

        self.app.push_screen(
            DeleteConfirmScreen("orphan job", job.identifier),
            handle_delete,
        )

    def _kill_job(self, job) -> None:
        """Kill a running orphan job with confirmation"""

        def handle_kill(confirmed: bool) -> None:
            if confirmed:
                success = self.state_provider.kill_job(job, perform=True)
                if success:
                    self.notify(f"Job {job.identifier} killed", severity="information")
                    self.refresh_orphan_jobs()
                else:
                    self.notify("Failed to kill job", severity="error")

        self.app.push_screen(
            KillConfirmScreen("orphan job", job.identifier),
            handle_kill,
        )

    def action_kill_all(self) -> None:
        """Kill all running orphan jobs"""
        running_jobs = [j for j in self.orphan_jobs if j.state and j.state.running()]

        if not running_jobs:
            self.notify("No running orphan jobs to kill", severity="warning")
            return

        def handle_kill_all(confirmed: bool) -> None:
            if confirmed:
                killed = 0
                for job in running_jobs:
                    if self.state_provider.kill_job(job, perform=True):
                        killed += 1

                self.notify(
                    f"Killed {killed} of {len(running_jobs)} running jobs",
                    severity="information",
                )
                self.refresh_orphan_jobs()

        self.app.push_screen(
            KillConfirmScreen("all running orphan jobs", f"{len(running_jobs)} jobs"),
            handle_kill_all,
        )

    def action_delete_all(self) -> None:
        """Delete all non-running orphan jobs"""
        deletable_jobs = [
            j for j in self.orphan_jobs if not j.state or not j.state.running()
        ]

        if not deletable_jobs:
            self.notify(
                "No deletable orphan jobs (all are running)", severity="warning"
            )
            return

        def handle_delete_all(confirmed: bool) -> None:
            if confirmed:
                deleted = 0
                for job in deletable_jobs:
                    success, _ = self.state_provider.delete_job_safely(
                        job, cascade_orphans=False
                    )
                    if success:
                        deleted += 1

                # Clean up orphan partials once at the end
                self.state_provider.cleanup_orphan_partials(perform=True)

                self.notify(f"Deleted {deleted} orphan jobs", severity="information")
                self.refresh_orphan_jobs()

        self.app.push_screen(
            DeleteConfirmScreen(
                "all finished orphan jobs",
                f"{len(deletable_jobs)} jobs",
                "This action cannot be undone",
            ),
            handle_delete_all,
        )


# Keep old name for backwards compatibility
StrayJobsTab = OrphanJobsTab
