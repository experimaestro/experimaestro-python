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
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import StateProvider

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


#: Log file extension of each stream
LOG_EXTENSIONS = {"stdout": ".out", "stderr": ".err"}


def log_file(provider: "StateProvider", task_id: str, job_id: str, stream: str) -> Path:
    """Log file of a job stream"""
    job = provider.get_job(task_id, job_id)
    assert job is not None and job.path is not None
    return job.path / f"{task_id.rsplit('.', 1)[-1]}{LOG_EXTENSIONS[stream]}"


def append(path: Path, text: str) -> None:
    with path.open("a", newline="") as out:
        out.write(text)


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
        """Experiments shown, with their status (``BaseExperiment.status_category``)"""
        ...

    async def experiment_order(self) -> list[str]:
        """Identifiers of the listed experiments, in display order"""
        ...

    async def sort_experiments(self, column: Literal["status", "started", "id"]):
        """Press the sort key of a column (the key of the current sort reverses
        it)"""
        ...

    async def search_experiments(self, text: str) -> None:
        """Show the experiments whose ID or one of whose tags contains a text
        (all of them when empty)"""
        ...

    async def experiment_tags(self) -> dict[str, set[str]]:
        """Tags shown for each listed experiment"""
        ...

    async def group_experiments(
        self, grouping: Literal["none", "prefix", "tag"]
    ) -> dict[str, set[str]]:
        """Group the experiment list; returns the experiments of each group (a
        single group named "" when not grouped)"""
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

    async def hide_tags(self, keys: set[str]) -> dict[str, set[str]]:
        """Hide the tags with these keys (and show the others) in the job list;
        returns the tag keys shown for each listed job"""
        ...

    async def set_filter(self, query: str) -> str | None:
        """Filter the job list; returns the error shown for an invalid query"""
        ...

    async def filter_completions(self, query: str) -> list[str]:
        """Suggestions offered (when asked for) for a filter being typed, with
        the cursor at its end"""
        ...

    async def accept_filter_completion(self, query: str, label: str) -> str:
        """Type a filter, accept one of the suggestions (with the keyboard);
        returns the filter shown"""
        ...

    async def job_detail(self, job_id: str) -> JobDetail:
        """Open the detail view of a job of the open experiment"""
        ...

    async def job_log(self, job_id: str, stream: Literal["stdout", "stderr"]) -> str:
        """Text of a job log, as displayed"""
        ...

    async def follow_log(
        self, job_id: str, stream: Literal["stdout", "stderr"], chunks: list[str]
    ) -> list[str]:
        """Open a job log, then append each chunk to its file, waiting for the
        display to change after each; returns the lines displayed at the end
        (the line being written included, trailing spaces removed)"""
        ...

    async def log_streams(self, job_id: str) -> tuple[list[str], str]:
        """Log streams of a job (``stdout``, ``stderr``) in the order they are
        shown, and the one displayed when opening the logs"""
        ...

    async def job_path(self, job_id: str) -> str:
        """Folder of a job, as shown in its detail view"""
        ...

    async def copy_job_id(self, job_id: str, *, shift: bool = False) -> str | None:
        """Click (with shift if asked) the identifier of a job in the job list;
        returns what was copied to the clipboard (None if nothing)"""
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

    async def graph(
        self, experiment_id: str, *, tasks_only: bool = True, stack: bool = True
    ) -> set[frozenset[str]]:
        """Nodes of the dependency graph of the current run (as the identifiers
        of the nodes stacked in each)"""
        ...

    async def active_services(self) -> dict[str, str]:
        """Services starting or running, shown whatever the page, as
        ``experiment/service`` with their (lowercase) state"""
        ...

    async def start_service(
        self, experiment_id: str, service_id: str
    ) -> Notification | None:
        """Start a service (None if not offered)"""
        ...

    async def stop_service(
        self, experiment_id: str, service_id: str
    ) -> Notification | None:
        """Stop a service (None if not offered)"""
        ...

    async def connection_state(self) -> str | None:
        """Connection state shown, whatever the page, when the state provider
        is not connected (``reconnecting`` or ``disconnected``; None when
        nothing is shown)"""
        ...
