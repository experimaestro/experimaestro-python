"""UI driver for the Textual interface

Runs the full :class:`ExperimaestroUI` app under ``App.run_test()`` and drives
it with key presses, reading back what the widgets display. Data is loaded by
thread workers, so every interaction is followed by :meth:`TuiDriver.settle`.
"""

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Literal

from rich.text import Text
from textual.pilot import Pilot
from textual.widgets import (
    DataTable,
    Input,
    RichLog,
    SelectionList,
    Static,
    TabbedContent,
    Tree,
)

from experimaestro.scheduler.state_provider import StateProvider
import experimaestro.tui.app
from experimaestro.tui.app import ExperimaestroUI
from experimaestro.tui.dialogs import DeleteConfirmScreen, KillConfirmScreen
from experimaestro.tui.log_viewer import LogViewerScreen, LogWidget
from experimaestro.tui.widgets.experiments import ExperimentsList
from experimaestro.tui.widgets.jobs import (
    DependencyList,
    JobDetailView,
    JobsTable,
    TreeGroupData,
    TreeJobData,
)
from experimaestro.tui.widgets.services import ServicesList
from experimaestro.tui.widgets.stray_jobs import (
    OrphanJobsPanel,
    OrphanJobsTab,
    StrayJobsPanel,
)

from .drivers import (
    LOG_EXTENSIONS,
    SEVERITIES,
    JobDetail,
    Notification,
    OrphanRow,
    append,
)


#: Experiment status category, from the first word of the TUI status column
_EXPERIMENT_STATUS = {
    "❌": "failed",
    "🏃": "running",
    "✓": "done",
    "⏹": "stopped",
    "Empty": "empty",
}


class RecordingUI(ExperimaestroUI):
    """The TUI, recording the notifications it shows"""

    # Relative paths are resolved from the module of the subclass
    CSS_PATH = Path(experimaestro.tui.app.__file__).parent / ExperimaestroUI.CSS_PATH

    def __init__(self, state_provider: StateProvider):
        super().__init__(state_provider=state_provider)
        self.notifications: list[Notification] = []

    def notify(self, message, *, severity="information", **kwargs):
        self.notifications.append(Notification(SEVERITIES[severity], str(message)))
        super().notify(message, severity=severity, **kwargs)


class TuiDriver:
    name = "tui"

    def __init__(self, provider: StateProvider):
        self.provider = provider
        self.app = RecordingUI(provider)
        self.pilot: Pilot
        self._stack = AsyncExitStack()
        self._experiment: str | None = None

    async def __aenter__(self) -> "TuiDriver":
        self.pilot = await self._stack.enter_async_context(
            self.app.run_test(size=(200, 60))
        )
        await self.settle()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    # --- Helpers

    async def settle(self) -> None:
        """Wait until background loads are done and their results displayed"""
        for _ in range(3):
            await self.pilot.pause()
            await self.app.workers.wait_for_complete()
        await self.pilot.pause()

    def _table(self, selector: str) -> DataTable:
        return self.app.query_one(selector, DataTable)

    async def _select_row(self, table: DataTable, key: str) -> None:
        """Move the cursor onto a row and give the table the focus"""
        table.focus()
        table.move_cursor(row=table.get_row_index(key))
        await self.pilot.pause()

    async def _expect_notification(self, start: int, confirm: str | None):
        """Confirms the dialog (if any), then returns the new notification"""
        await self.settle()
        if confirm is not None and isinstance(
            self.app.screen, (KillConfirmScreen, DeleteConfirmScreen)
        ):
            await self.pilot.click(confirm)
            await self.settle()
        new = self.app.notifications[start:]
        assert new, "no notification shown"
        return new[-1]

    async def _show_experiments(self) -> None:
        """Back to the (expanded) experiment list"""
        while self._experiment is not None or isinstance(
            self.app.screen, LogViewerScreen
        ):
            await self.pilot.press("escape")
            await self.settle()
            if not self.app.query_one("#experiment-tabs").has_class("hidden"):
                continue
            if self.app.query_one("#job-detail-container").has_class("hidden"):
                self._experiment = None
        self._table("#experiments-table").focus()

    # --- Experiments

    async def experiments(self) -> dict[str, str]:
        await self._show_experiments()
        table = self._table("#experiments-table")
        return {
            str(key.value): _EXPERIMENT_STATUS[
                str(table.get_cell(key, "status")).split()[0]
            ]
            for key in table.rows
        }

    async def experiment_order(self) -> list[str]:
        await self._show_experiments()
        await self.settle()
        return [str(key.value) for key in self._table("#experiments-table").rows]

    async def sort_experiments(self, column: Literal["status", "started", "id"]):
        await self._show_experiments()
        await self.pilot.press({"status": "S", "started": "D", "id": "O"}[column])
        await self.settle()

    async def search_experiments(self, text: str) -> None:
        await self._show_experiments()
        await self.pilot.press("/")
        search = self.app.query_one("#experiments-search", Input)
        assert self.app.focused is search
        if text:
            search.value = ""
            await self.pilot.press(*text)
            await self.pilot.press("enter")
        else:
            await self.pilot.press("escape")
        await self.settle()

    async def experiment_tags(self) -> dict[str, set[str]]:
        await self._show_experiments()
        table = self._table("#experiments-table")
        tags = {}
        for key in table.rows:
            # The ID followed by "#tag" for each tag
            exp_id, *words = Text.from_markup(
                str(table.get_cell(key, "id"))
            ).plain.split()
            tags[exp_id] = {word[1:] for word in words if word.startswith("#")}
        return tags

    async def group_experiments(
        self, grouping: Literal["none", "prefix", "tag"]
    ) -> dict[str, set[str]]:
        await self._show_experiments()
        widget = self.app.query_one(ExperimentsList)
        while widget.grouping != grouping:
            await self.pilot.press("g")
            await self.settle()
        groups: dict[str, set[str]] = {}
        name = ""
        for key in self._table("#experiments-table").rows:
            exp_id = widget.row_experiments[str(key.value)]
            if exp_id is None:
                # Group header ("\0" + name)
                name = str(key.value)[1:]
                groups[name] = set()
            else:
                groups.setdefault(name, set()).add(exp_id)
        return groups

    async def runs(self, experiment_id: str) -> list[str]:
        await self._show_experiments()
        await self._select_row(self._table("#experiments-table"), experiment_id)
        await self.pilot.press("d")
        await self.settle()
        runs = [str(key.value) for key in self._table("#runs-table").rows]
        await self.pilot.press("escape")
        await self.settle()
        return runs

    async def open_experiment(
        self, experiment_id: str, run_id: str | None = None
    ) -> None:
        await self._show_experiments()
        await self._select_row(self._table("#experiments-table"), experiment_id)
        if run_id is None:
            await self.pilot.press("enter")
        else:
            await self.pilot.press("d")
            await self.settle()
            await self._select_row(self._table("#runs-table"), run_id)
            await self.pilot.press("enter")
        await self.settle()
        self._experiment = experiment_id

    async def kill_experiment(self, experiment_id: str) -> Notification:
        await self._show_experiments()
        await self._select_row(self._table("#experiments-table"), experiment_id)
        start = len(self.app.notifications)
        await self.pilot.press("ctrl+k")
        return await self._expect_notification(start, "#kill-yes")

    async def delete_experiment(self, experiment_id: str) -> Notification:
        await self._show_experiments()
        await self._select_row(self._table("#experiments-table"), experiment_id)
        start = len(self.app.notifications)
        await self.pilot.press("ctrl+d")
        return await self._expect_notification(start, "#delete-yes")

    # --- Jobs

    async def _show_jobs(self) -> None:
        """Back from the job detail (or the logs) to the job list"""
        assert self._experiment is not None, "no experiment open"
        while isinstance(self.app.screen, LogViewerScreen) or not self.app.query_one(
            "#job-detail-container"
        ).has_class("hidden"):
            await self.pilot.press("escape")
            await self.settle()
        tabs = self.app.query_one("#experiment-tabs", TabbedContent)
        tabs.active = "jobs-tab"
        self._table("#jobs-table").focus()
        await self.pilot.pause()

    async def visible_jobs(self) -> set[str]:
        await self._show_jobs()
        await self.settle()
        return {str(key.value) for key in self._table("#jobs-table").rows}

    async def job_order(self) -> list[str]:
        await self._show_jobs()
        await self.settle()
        return [str(key.value) for key in self._table("#jobs-table").rows]

    async def sort_by(self, column: Literal["task", "status"]) -> None:
        await self._show_jobs()
        key = {"task": "T", "status": "S"}[column]
        await self.pilot.press(key)
        await self.settle()
        if self.app.query_one(JobsTable)._sort_reverse:
            await self.pilot.press(key)
            await self.settle()

    async def hide_tags(self, keys: set[str]) -> dict[str, set[str]]:
        await self._show_jobs()
        await self.pilot.press("number_sign")
        await self.settle()
        selection = self.app.screen.query_one("#tag-visibility-list", SelectionList)
        for index in range(selection.option_count):
            value = selection.get_option_at_index(index).value
            if value in keys:
                selection.deselect(value)
            else:
                selection.select(value)
        await self.pilot.press("escape")
        await self.settle()

        table = self._table("#jobs-table")
        shown = {}
        for key in table.rows:
            # "key=value, key=value" or "-"
            text = str(table.get_cell(key, "tags"))
            shown[str(key.value)] = (
                set()
                if text == "-"
                else {tag.split("=", 1)[0] for tag in text.split(", ")}
            )
        return shown

    async def group_by(self, key: Literal["status", "task"]) -> dict[str, set[str]]:
        await self._show_jobs()
        # Tree view, keeping only the requested grouping level
        await self.pilot.press("t")
        await self.settle()
        disabled = {"status": "23", "task": "13"}[key]
        for level in disabled:
            await self.pilot.press(level)
        await self.settle()

        def leaves(node) -> set[str]:
            if isinstance(node.data, TreeJobData):
                return {node.data.job_id}
            return set().union(*(leaves(child) for child in node.children))

        tree = self.app.query_one("#jobs-tree", Tree)
        groups = {
            str(child.data.value): leaves(child)
            for child in tree.root.children
            if isinstance(child.data, TreeGroupData)
        }

        # Back to the table view
        for level in disabled:
            await self.pilot.press(level)
        await self.pilot.press("t")
        await self.settle()
        return groups

    async def set_filter(self, query: str) -> str | None:
        await self._show_jobs()
        await self.pilot.press("/")
        await self.pilot.pause()
        self.app.query_one("#search-input", Input).value = query
        await self.pilot.press("enter")
        await self.settle()
        error = self.app.query_one("#search-error", Static)
        if error.display:
            message = str(error.render())
            # Leave the search bar, as a user would after reading the error
            await self.pilot.press("escape")
            await self.settle()
            return message
        return None

    async def job_detail(self, job_id: str) -> JobDetail:
        await self._show_jobs()
        await self._select_row(self._table("#jobs-table"), job_id)
        await self.pilot.press("enter")
        await self.settle()
        view = self.app.query_one(JobDetailView)
        job = view.job_data
        assert job is not None and job.identifier == job_id
        deps = self.app.query_one(DependencyList)
        dependencies = {}
        for dep_id in view.dependencies_map.get(job_id, []):
            _, state = view.jobs_info.get(dep_id, ("", None))
            dependencies[dep_id] = state.name.lower() if state else None
        assert set(deps.task_ids) <= set(dependencies)
        return JobDetail(
            job_id=job.identifier,
            task_id=job.task_id,
            state=job.state.name.lower(),
            tags=dict(view.tags_map.get(job_id, {})),
            dependencies=dependencies,
        )

    async def _open_log(self, job_id: str) -> LogViewerScreen:
        await self._show_jobs()
        await self._select_row(self._table("#jobs-table"), job_id)
        await self.pilot.press("l")
        await self.settle()
        assert isinstance(self.app.screen, LogViewerScreen)
        return self.app.screen

    async def _show_log(self, job_id: str, stream: str) -> LogWidget:
        screen = await self._open_log(job_id)
        for widget in screen.query(LogWidget):
            if widget.file_path.endswith(LOG_EXTENSIONS[stream]):
                # Several files are shown in tabs; a hidden RichLog does not
                # render what it is given
                for tabs in screen.query(TabbedContent):
                    tabs.active = widget.parent.id
                await self.settle()
                return widget
        raise AssertionError(f"no {stream} log displayed")

    @staticmethod
    def _log_lines(widget: LogWidget) -> list[str]:
        lines = [strip.text for strip in widget.query_one(RichLog).lines]
        partial = widget.query_one(f"#{widget.id}-partial", Static)
        if partial.display:
            lines.append(str(partial.render()))
        return [line.rstrip() for line in lines]

    async def job_log(self, job_id: str, stream: Literal["stdout", "stderr"]) -> str:
        widget = await self._show_log(job_id, stream)
        log = widget.query_one(RichLog)
        return "".join(f"{strip.text}\n" for strip in log.lines)

    async def follow_log(
        self, job_id: str, stream: Literal["stdout", "stderr"], chunks: list[str]
    ) -> list[str]:
        widget = await self._show_log(job_id, stream)
        path = Path(widget.file_path)
        for chunk in chunks:
            before = self._log_lines(widget)
            append(path, chunk)
            # The viewer polls its files (every 0.5s)
            for _ in range(50):
                await self.pilot.pause(0.1)
                if self._log_lines(widget) != before:
                    break
            else:
                raise AssertionError(f"log display unchanged after {chunk!r}")
        return self._log_lines(widget)

    async def log_streams(self, job_id: str) -> tuple[list[str], str]:
        screen = await self._open_log(job_id)
        streams = {ext: stream for stream, ext in LOG_EXTENSIONS.items()}
        widgets = list(screen.query(LogWidget))
        order = [streams[Path(w.file_path).suffix] for w in widgets]
        shown = order[0]
        for tabs in screen.query(TabbedContent):
            (active,) = [w for w in widgets if w.parent.id == tabs.active]
            shown = streams[Path(active.file_path).suffix]
        return order, shown

    async def job_path(self, job_id: str) -> str:
        await self.job_detail(job_id)
        label = str(self.app.query_one("#job-path-label", Static).render())
        return label.removeprefix("Path: ")

    async def copy_job_id(self, job_id: str, *, shift: bool = False) -> str | None:
        raise NotImplementedError("the job identifier is clicked in the web UI")

    async def kill_job(self, job_id: str) -> Notification:
        await self._show_jobs()
        await self._select_row(self._table("#jobs-table"), job_id)
        start = len(self.app.notifications)
        await self.pilot.press("ctrl+k")
        return await self._expect_notification(start, "#kill-yes")

    async def delete_job(self, job_id: str) -> Notification:
        await self._show_jobs()
        await self._select_row(self._table("#jobs-table"), job_id)
        start = len(self.app.notifications)
        await self.pilot.press("ctrl+d")
        return await self._expect_notification(start, "#delete-yes")

    # --- Orphans

    async def _show_orphans(self) -> OrphanJobsTab:
        self.app.query_one("#main-tabs", TabbedContent).active = "orphan-tab"
        await self.settle()
        tab = self.app.query_one(OrphanJobsTab)
        tab.refresh_orphan_jobs()
        await self.settle()
        return tab

    async def orphans(self) -> dict[str, OrphanRow]:
        tab = await self._show_orphans()
        rows: dict[str, OrphanRow] = {}
        for panel, stray in ((tab.stray_panel, True), (tab.orphan_panel, False)):
            for job in panel.jobs:
                if job.identifier in rows:
                    continue
                rows[job.identifier] = OrphanRow(
                    stray=stray,
                    size=panel._size_bytes_cache.get(job.identifier),
                )
        return rows

    async def _orphan_panel_action(
        self, panel_type: type, job_id: str, key: str, confirm: str
    ) -> Notification:
        tab = await self._show_orphans()
        panel = tab.query_one(panel_type)
        tabs = tab.query_one("#orphan-subtabs", TabbedContent)
        tabs.active = (
            "stray-subtab" if panel_type is StrayJobsPanel else "orphan-subtab"
        )
        await self.settle()
        await self._select_row(panel.query_one(DataTable), job_id)
        start = len(self.app.notifications)
        await self.pilot.press(key)
        return await self._expect_notification(start, confirm)

    async def kill_orphan(self, job_id: str) -> Notification:
        return await self._orphan_panel_action(
            StrayJobsPanel, job_id, "ctrl+k", "#kill-yes"
        )

    async def delete_orphan(self, job_id: str) -> Notification:
        return await self._orphan_panel_action(
            OrphanJobsPanel, job_id, "ctrl+d", "#delete-yes"
        )

    async def kill_all_strays(self) -> Notification:
        tab = await self._show_orphans()
        tab.query_one("#orphan-subtabs", TabbedContent).active = "stray-subtab"
        await self.settle()
        start = len(self.app.notifications)
        await self.pilot.click("#stray-kill-all-btn")
        return await self._expect_notification(start, "#kill-yes")

    async def delete_all_orphans(self) -> Notification:
        tab = await self._show_orphans()
        tab.query_one("#orphan-subtabs", TabbedContent).active = "orphan-subtab"
        await self.settle()
        start = len(self.app.notifications)
        await self.pilot.click("#orphan-delete-all-btn")
        return await self._expect_notification(start, "#delete-yes")

    # --- Services

    async def graph(
        self, experiment_id: str, *, tasks_only: bool = True, stack: bool = True
    ) -> set[frozenset[str]]:
        raise NotImplementedError("the TUI has no dependency graph view")

    async def filter_completions(self, query: str) -> list[str]:
        raise NotImplementedError("the TUI search bar has no completion")

    async def accept_filter_completion(self, query: str, label: str) -> str:
        raise NotImplementedError("the TUI search bar has no completion")

    async def services(self, experiment_id: str) -> dict[str, str]:
        await self.open_experiment(experiment_id)
        self.app.query_one("#experiment-tabs", TabbedContent).active = "services-tab"
        await self.settle()
        services = self.app.query_one(ServicesList)
        return {
            service_id: service.state.name.lower()
            for service_id, service in services._services.items()
        }

    async def _service_action(
        self, experiment_id: str, service_id: str, key: str
    ) -> Notification:
        await self.services(experiment_id)
        await self._select_row(self._table("#services-table"), service_id)
        start = len(self.app.notifications)
        await self.pilot.press(key)
        return await self._expect_notification(start, None)

    async def active_services(self) -> dict[str, str]:
        raise NotImplementedError("the TUI lists the services in its Services tab")

    async def start_service(self, experiment_id: str, service_id: str) -> Notification:
        return await self._service_action(experiment_id, service_id, "ctrl+s")

    async def stop_service(self, experiment_id: str, service_id: str) -> Notification:
        return await self._service_action(experiment_id, service_id, "ctrl+k")

    async def connection_state(self) -> str | None:
        await self.settle()
        status = self.app.query_one("#connection-status", Static)
        if status.has_class("hidden"):
            return None
        (state,) = {"reconnecting", "disconnected"} & set(status.classes)
        return state
