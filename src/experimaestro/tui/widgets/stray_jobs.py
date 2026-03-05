"""Stray and orphan jobs tab widget for the TUI

Displays two sub-tabs:
- Stray jobs: Running jobs from old experiment runs (not in the latest run)
- Orphan jobs: Jobs on disk not referenced by any experiment run
"""

import logging
from typing import Optional
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import DataTable, Static, Button, TabbedContent, TabPane
from textual.binding import Binding

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.tui.utils import get_status_icon
from experimaestro.tui.dialogs import DeleteConfirmScreen, KillConfirmScreen
from experimaestro.tui.messages import SizeCalculated

logger = logging.getLogger("xpm.tui.orphan_jobs")


class _JobListPanel(Vertical):
    """Base panel for displaying a list of jobs in a DataTable

    Subclasses override _fetch_jobs() to provide the appropriate job list.
    """

    _size_cache: dict = {}  # Class-level cache (formatted strings)
    _size_bytes_cache: dict = {}  # Class-level cache (raw bytes for sorting)

    def __init__(self, state_provider: StateProvider, panel_id: str) -> None:
        super().__init__()
        self.state_provider = state_provider
        self.jobs: list = []
        self._pending_jobs: list = []
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False
        self._panel_id = panel_id

    @property
    def table_id(self) -> str:
        return f"{self._panel_id}-table"

    @property
    def info_id(self) -> str:
        return f"{self._panel_id}-info"

    @property
    def warning_id(self) -> str:
        return f"{self._panel_id}-warning"

    def compose(self) -> ComposeResult:
        yield Static("", id=self.warning_id, classes="warning-banner hidden")
        with Horizontal(classes="controls-bar"):
            yield Button("Refresh", id=f"{self._panel_id}-refresh-btn")
            yield from self._extra_buttons()
        yield DataTable(id=self.table_id, cursor_type="row")
        yield Static("", id=self.info_id)

    def _extra_buttons(self) -> ComposeResult:
        """Override to add panel-specific buttons"""
        return
        yield  # make it a generator

    def on_mount(self) -> None:
        table = self.query_one(f"#{self.table_id}", DataTable)
        table.add_column("", key="status", width=3)
        table.add_column("Job ID", key="job_id", width=10)
        table.add_column("Task", key="task")
        table.add_column("Size", key="size", width=10)
        self.refresh_jobs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"{self._panel_id}-refresh-btn":
            self.refresh_jobs()
            self.notify("Refreshed", severity="information")

    def _fetch_jobs(self) -> list:
        """Override to fetch the appropriate job list"""
        raise NotImplementedError

    def refresh_jobs(self) -> None:
        """Refresh the job list"""
        self.jobs = self._fetch_jobs()

        # Only calculate sizes for local paths that exist
        self._pending_jobs = [
            j
            for j in self.jobs
            if j.identifier not in self._size_cache and j.path and j.path.exists()
        ]

        self._rebuild_table()
        self._update_parent_title()

        if self._pending_jobs:
            self._calculate_next_size()

    def _update_parent_title(self) -> None:
        """Update the parent tab title"""
        try:
            self.app.update_orphan_tab_title()
        except Exception:
            pass

    def _get_sorted_jobs(self):
        jobs = self.jobs[:]
        if self._sort_column == "task":
            jobs.sort(key=lambda j: j.task_id or "", reverse=self._sort_reverse)
        elif self._sort_column == "size":
            jobs.sort(
                key=lambda j: self._size_bytes_cache.get(j.identifier, -1),
                reverse=self._sort_reverse,
            )
        return jobs

    def _rebuild_table(self) -> None:
        from rich.text import Text

        table = self.query_one(f"#{self.table_id}", DataTable)
        table.clear()

        for job in self._get_sorted_jobs():
            failure_reason = job.state.failure_reason if job.state else None
            status_icon = get_status_icon(
                job.state.name if job.state else "unknown", failure_reason
            )

            is_running = job.state and job.state.running()
            if is_running:
                job_id_text = Text(job.identifier[:7], style="bold yellow")
                task_text = Text(job.task_id or "", style="yellow")
            else:
                job_id_text = Text(job.identifier[:7])
                task_text = Text(job.task_id or "")

            size_text = self._size_cache.get(job.identifier, "waiting")

            table.add_row(
                status_icon, job_id_text, task_text, size_text, key=job.identifier
            )

    def _calculate_next_size(self) -> None:
        if not self._pending_jobs:
            return
        job = self._pending_jobs.pop(0)
        self._update_size_cell(job.identifier, "calc...")
        self.run_worker(self._calc_size_worker(job.identifier, job.path), thread=True)

    async def _calc_size_worker(self, job_id: str, path):
        size_bytes = await self._get_folder_size_async(path)
        size_str = self._format_size(size_bytes)
        self._size_cache[job_id] = size_str
        self._size_bytes_cache[job_id] = size_bytes
        self.post_message(SizeCalculated(job_id, size_str, size_bytes))

    def on_size_calculated(self, message: SizeCalculated) -> None:
        self._size_bytes_cache[message.job_id] = message.size_bytes
        self._update_size_cell(message.job_id, message.size)
        self._calculate_next_size()

    @staticmethod
    async def _get_folder_size_async(path) -> int:
        import asyncio
        import shutil
        import sys

        if shutil.which("du"):
            try:
                if sys.platform == "darwin":
                    proc = await asyncio.create_subprocess_exec(
                        "du",
                        "-sk",
                        str(path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode == 0 and stdout:
                        return int(stdout.decode().split()[0]) * 1024
                else:
                    proc = await asyncio.create_subprocess_exec(
                        "du",
                        "-sb",
                        str(path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode == 0 and stdout:
                        return int(stdout.decode().split()[0])
            except (asyncio.TimeoutError, ValueError, IndexError, OSError):
                pass

        return _JobListPanel._get_folder_size_sync(path)

    @staticmethod
    def _get_folder_size_sync(path) -> int:
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
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def _update_size_cell(self, job_id: str, value: str = None) -> None:
        try:
            table = self.query_one(f"#{self.table_id}", DataTable)
            size_text = (
                value if value is not None else self._size_cache.get(job_id, "-")
            )
            table.update_cell(job_id, "size", size_text)
        except Exception:
            pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_job_info()

    def _update_job_info(self) -> None:
        table = self.query_one(f"#{self.table_id}", DataTable)
        info = self.query_one(f"#{self.info_id}", Static)

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
            job = next((j for j in self.jobs if j.identifier == job_id), None)
            if job and job.path:
                size = self._size_cache.get(job.identifier, "calculating...")
                state = job.state.name if job.state else "unknown"
                display_path = self.state_provider.get_display_path(job)
                info.update(f"Path: {display_path}  |  Size: {size}  |  State: {state}")
            else:
                info.update("")

    def action_copy_path(self) -> None:
        """Copy the selected job's path to clipboard"""
        from experimaestro.tui.clipboard import copy

        job = self._get_selected_job()
        if not job or not job.path:
            self.notify("No path available", severity="warning")
            return
        display_path = self.state_provider.get_display_path(job)
        if copy(display_path):
            self.notify(f"Path copied: {display_path}", severity="information")
        else:
            self.notify("Failed to copy path", severity="error")

    def _get_selected_job(self):
        table = self.query_one(f"#{self.table_id}", DataTable)
        if table.cursor_row is None:
            return None
        try:
            row_key = list(table.rows.keys())[table.cursor_row]
        except IndexError:
            return None
        if row_key:
            job_id = str(row_key.value)
            return next((j for j in self.jobs if j.identifier == job_id), None)
        return None


class StrayJobsPanel(_JobListPanel):
    """Panel for stray jobs (running jobs from old experiment runs)"""

    BINDINGS = [
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("ctrl+k", "kill_selected", "Kill", show=False),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__(state_provider, "stray")

    def _extra_buttons(self) -> ComposeResult:
        yield Button("Kill All", id="stray-kill-all-btn", variant="error")

    def _fetch_jobs(self) -> list:
        return self.state_provider.get_stray_jobs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        super().on_button_pressed(event)
        if event.button.id == "stray-kill-all-btn":
            self._action_kill_all()

    def action_kill_selected(self) -> None:
        job = self._get_selected_job()
        if not job:
            return
        if not job.state or not job.state.running():
            self.notify("Job is not running", severity="warning")
            return
        self._kill_job(job)

    def _kill_job(self, job) -> None:
        def handle_kill(confirmed: bool) -> None:
            if confirmed:
                try:
                    self.state_provider.kill_job(job, perform=True)
                    self.notify(f"Job {job.identifier} killed", severity="information")
                    self.refresh_jobs()
                except Exception as e:
                    self.notify(f"Failed to kill job: {e}", severity="error")

        self.app.push_screen(
            KillConfirmScreen("stray job", job.identifier), handle_kill
        )

    def _action_kill_all(self) -> None:
        running_jobs = [j for j in self.jobs if j.state and j.state.running()]
        if not running_jobs:
            self.notify("No stray jobs to kill", severity="warning")
            return

        def handle_kill_all(confirmed: bool) -> None:
            if confirmed:
                killed = failed = 0
                for job in running_jobs:
                    try:
                        self.state_provider.kill_job(job, perform=True)
                        killed += 1
                    except Exception:
                        failed += 1
                msg = f"Killed {killed}/{len(running_jobs)} jobs"
                if failed:
                    msg += f" ({failed} failed)"
                self.notify(msg, severity="information" if not failed else "warning")
                self.refresh_jobs()

        self.app.push_screen(
            KillConfirmScreen("all stray jobs", f"{len(running_jobs)} jobs"),
            handle_kill_all,
        )


class OrphanJobsPanel(_JobListPanel):
    """Panel for orphan jobs (not referenced by any experiment)"""

    BINDINGS = [
        Binding("f", "copy_path", "Copy Path", show=False),
        Binding("ctrl+d", "delete_selected", "Delete", show=False),
        Binding("ctrl+k", "kill_selected", "Kill", show=False),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__(state_provider, "orphan")

    def _extra_buttons(self) -> ComposeResult:
        yield Button("Kill All", id="orphan-kill-all-btn", variant="error")
        yield Button("Delete All", id="orphan-delete-all-btn", variant="warning")

    def _fetch_jobs(self) -> list:
        return self.state_provider.get_orphan_jobs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        super().on_button_pressed(event)
        if event.button.id == "orphan-kill-all-btn":
            self._action_kill_all()
        elif event.button.id == "orphan-delete-all-btn":
            self._action_delete_all()

    def action_delete_selected(self) -> None:
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
        job = self._get_selected_job()
        if not job:
            return
        if not job.state or not job.state.running():
            self.notify("Job is not running", severity="warning")
            return
        self._kill_job(job)

    def _delete_job(self, job) -> None:
        def handle_delete(confirmed: bool) -> None:
            if confirmed:
                success, msg = self.state_provider.delete_job_safely(job)
                if success:
                    self.notify(msg, severity="information")
                    self.refresh_jobs()
                else:
                    self.notify(msg, severity="error")

        self.app.push_screen(
            DeleteConfirmScreen("orphan job", job.identifier), handle_delete
        )

    def _kill_job(self, job) -> None:
        def handle_kill(confirmed: bool) -> None:
            if confirmed:
                try:
                    self.state_provider.kill_job(job, perform=True)
                    self.notify(f"Job {job.identifier} killed", severity="information")
                    self.refresh_jobs()
                except Exception as e:
                    self.notify(f"Failed to kill job: {e}", severity="error")

        self.app.push_screen(
            KillConfirmScreen("orphan job", job.identifier), handle_kill
        )

    def _action_kill_all(self) -> None:
        running_jobs = [j for j in self.jobs if j.state and j.state.running()]
        if not running_jobs:
            self.notify("No running orphan jobs to kill", severity="warning")
            return

        def handle_kill_all(confirmed: bool) -> None:
            if confirmed:
                killed = failed = 0
                for job in running_jobs:
                    try:
                        self.state_provider.kill_job(job, perform=True)
                        killed += 1
                    except Exception:
                        failed += 1
                msg = f"Killed {killed}/{len(running_jobs)} jobs"
                if failed:
                    msg += f" ({failed} failed)"
                self.notify(msg, severity="information" if not failed else "warning")
                self.refresh_jobs()

        self.app.push_screen(
            KillConfirmScreen("all running orphan jobs", f"{len(running_jobs)} jobs"),
            handle_kill_all,
        )

    def _action_delete_all(self) -> None:
        deletable = [j for j in self.jobs if not j.state or not j.state.running()]
        if not deletable:
            self.notify("No deletable orphan jobs", severity="warning")
            return

        def handle_delete_all(confirmed: bool) -> None:
            if confirmed:
                deleted = 0
                for job in deletable:
                    success, _ = self.state_provider.delete_job_safely(job)
                    if success:
                        deleted += 1
                self.notify(f"Deleted {deleted} orphan jobs", severity="information")
                self.refresh_jobs()

        self.app.push_screen(
            DeleteConfirmScreen(
                "all finished orphan jobs",
                f"{len(deletable)} jobs",
                "This action cannot be undone",
            ),
            handle_delete_all,
        )


class OrphanJobsTab(Vertical):
    """Container with sub-tabs for stray and orphan jobs"""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("T", "sort_by_task", "Sort Task", show=False),
        Binding("Z", "sort_by_size", "Sort Size", show=False),
    ]

    def __init__(self, state_provider: StateProvider) -> None:
        super().__init__()
        self.state_provider = state_provider

    def compose(self) -> ComposeResult:
        with TabbedContent(id="orphan-subtabs"):
            with TabPane("Stray (0)", id="stray-subtab"):
                yield StrayJobsPanel(self.state_provider)
            with TabPane("Orphans (0)", id="orphan-subtab"):
                yield OrphanJobsPanel(self.state_provider)

    @property
    def stray_panel(self) -> StrayJobsPanel:
        return self.query_one(StrayJobsPanel)

    @property
    def orphan_panel(self) -> OrphanJobsPanel:
        return self.query_one(OrphanJobsPanel)

    @property
    def running_count(self) -> int:
        """Number of stray (running) jobs"""
        return len(self.stray_panel.jobs)

    @property
    def finished_count(self) -> int:
        """Number of orphan jobs"""
        return len(self.orphan_panel.jobs)

    @property
    def orphan_count(self) -> int:
        """Total count for backwards compat"""
        return self.running_count + self.finished_count

    def refresh_orphan_jobs(self) -> None:
        """Refresh both panels"""
        self.stray_panel.refresh_jobs()
        self.orphan_panel.refresh_jobs()
        self._update_subtab_titles()

    def _update_subtab_titles(self) -> None:
        try:
            tabs = self.query_one("#orphan-subtabs", TabbedContent)
            stray_tab = tabs.get_tab("stray-subtab")
            if stray_tab:
                stray_tab.label = f"Stray ({self.running_count})"
            orphan_tab = tabs.get_tab("orphan-subtab")
            if orphan_tab:
                orphan_tab.label = f"Orphans ({self.finished_count})"
        except Exception:
            pass

    def action_refresh(self) -> None:
        self.refresh_orphan_jobs()
        self.notify("Refreshed", severity="information")

    def action_sort_by_task(self) -> None:
        for panel in [self.stray_panel, self.orphan_panel]:
            if panel._sort_column == "task":
                panel._sort_reverse = not panel._sort_reverse
            else:
                panel._sort_column = "task"
                panel._sort_reverse = False
            panel._rebuild_table()

    def action_sort_by_size(self) -> None:
        for panel in [self.stray_panel, self.orphan_panel]:
            if panel._sort_column == "size":
                panel._sort_reverse = not panel._sort_reverse
            else:
                panel._sort_column = "size"
                panel._sort_reverse = True
            panel._rebuild_table()


# Keep old name for backwards compatibility
StrayJobsTab = OrphanJobsTab
