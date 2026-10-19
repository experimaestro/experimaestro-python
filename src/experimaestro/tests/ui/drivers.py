"""The interface UI feature tests are written against

A driver performs what a user does (open an experiment, type a filter, kill a
job) and reports what the interface shows. Each interface has its own driver:

- ``tui``: :class:`~.tui_driver.TuiDriver`, the Textual app under ``run_test()``
- ``web``: :class:`~.web_driver.WebDriver`, the WebSocket protocol through
  FastAPI's ``TestClient`` (server side: the React code is not exercised)

- ``browser``: :class:`~.browser_driver.BrowserDriver`, the web UI in Chromium
  (Playwright)

Results are normalized so that the same assertions hold for every interface;
notifications in particular keep their message but map the severities onto
:data:`Severity`. Operations return ``None`` when the interface does not offer
them in the current state (e.g. no kill button for a finished job): for the
user, that is another way of refusing them.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol

Severity = Literal["info", "warning", "error"]

#: Interface-specific severities mapped onto :data:`Severity`
SEVERITIES: dict[str, Severity] = {
    # Textual
    "information": "info",
    "warning": "warning",
    "error": "error",
    # Bootstrap (web UI toasts)
    "success": "info",
    "info": "info",
    "danger": "error",
}


@dataclass(frozen=True)
class Notification:
    severity: Severity
    message: str


def refused(result: Notification | None, message: str) -> bool:
    """Whether an operation was refused: not offered, or refused with a warning"""
    return result is None or result == Notification("warning", message)


@dataclass
class JobDetail:
    job_id: str
    task_id: str
    #: Lowercase state name
    state: str
    tags: dict[str, str] = field(default_factory=dict)
    #: Dependency job id -> lowercase state name (None when unknown)
    dependencies: dict[str, str | None] = field(default_factory=dict)


@dataclass
class OrphanRow:
    #: Whether the job is still running (stray job)
    stray: bool
    #: Folder size in bytes (None if not computed)
    size: int | None


class UIDriver(Protocol):
    #: Interface name, as used by the ``ui`` marker
    name: str

    async def experiments(self) -> dict[str, str]:
        """Experiments shown, with their (lowercase) status"""
        ...

    async def runs(self, experiment_id: str) -> list[str]:
        """Runs of an experiment, as listed in the run history"""
        ...

    async def open_experiment(
        self, experiment_id: str, run_id: str | None = None
    ) -> None:
        """Show the jobs of an experiment (its current run by default)"""
        ...

    async def visible_jobs(self) -> set[str]:
        """Identifiers of the jobs listed for the open experiment"""
        ...

    async def job_order(self) -> list[str]:
        """Identifiers of the listed jobs, in display order"""
        ...

    async def sort_by(self, column: Literal["task", "status"]) -> None:
        """Sort the job list by a column (tasks by name, statuses by rank)"""
        ...

    async def group_by(self, key: Literal["status", "task"]) -> dict[str, set[str]]:
        """Group the job list; returns the jobs of each group (by its label)"""
        ...

    async def set_filter(self, query: str) -> str | None:
        """Filter the job list; returns the error shown for an invalid query"""
        ...

    async def job_detail(self, job_id: str) -> JobDetail:
        """Open the detail view of a job of the open experiment"""
        ...

    async def job_log(self, job_id: str, stream: Literal["stdout", "stderr"]) -> str:
        """Text of a job log, as displayed"""
        ...

    async def kill_job(self, job_id: str) -> Notification | None:
        """Kill a job of the open experiment (confirming if asked)"""
        ...

    async def delete_job(self, job_id: str) -> Notification | None:
        """Delete a job of the open experiment (confirming if asked)"""
        ...

    async def kill_experiment(self, experiment_id: str) -> Notification | None:
        """Kill the running jobs of an experiment (confirming if asked)"""
        ...

    async def delete_experiment(self, experiment_id: str) -> Notification | None:
        """Delete an experiment (confirming if asked)"""
        ...

    async def orphans(self) -> dict[str, OrphanRow]:
        """Orphan jobs (stray ones included) shown in the orphans view"""
        ...

    async def kill_orphan(self, job_id: str) -> Notification | None: ...

    async def delete_orphan(self, job_id: str) -> Notification | None: ...

    async def kill_all_strays(self) -> Notification | None:
        """Kill every stray (running orphan) job (confirming if asked)"""
        ...

    async def delete_all_orphans(self) -> Notification | None:
        """Delete every finished orphan job (confirming if asked)"""
        ...

    async def services(self, experiment_id: str) -> dict[str, str]:
        """Services of an experiment, with their (lowercase) state"""
        ...
