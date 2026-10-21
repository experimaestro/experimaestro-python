"""UI driver for the web interface, in a real browser

Serves the built frontend (``webui/data``) with uvicorn and drives it with
Playwright (headless Chromium): it clicks what a user clicks and reads what the
page shows, so the React code — sorting, grouping, formatting, the client-side
store — is exercised too.

Requires the ``browser`` dependency group, a Chromium for Playwright and the
frontend build (committed; rebuild it after changing ``app/``)::

    uv sync --all-extras --group browser
    uv run playwright install chromium

Browser tests run when ``XPM_TEST_BROWSER`` is set (see ``conftest.py``).
Playwright's async API is used: its sync API relies on greenlet, which crashes
on free-threaded Python.
"""

import json
import threading
import time
from types import SimpleNamespace
from typing import Literal

import uvicorn
from playwright.async_api import Locator, Page, WebSocket, async_playwright

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.webui.app import create_app

from .drivers import SEVERITIES, JobDetail, Notification, OrphanRow

TOKEN = "test-token"

#: Resolves once the page has not changed for a while (or after a timeout)
_SETTLE_JS = """
([quiet, timeout]) => new Promise((resolve) => {
    let timer;
    const done = () => { observer.disconnect(); resolve(); };
    const observer = new MutationObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(done, quiet);
    });
    observer.observe(document.body, {
        subtree: true, childList: true, attributes: true, characterData: true,
    });
    timer = setTimeout(done, quiet);
    setTimeout(done, timeout);
})
"""

#: Toast severity, from its Bootstrap background class
_TOAST_CLASSES = {f"bg-{name}": severity for name, severity in SEVERITIES.items()}


class _Server:
    """The web UI server (uvicorn) in a thread, on a free port"""

    def __init__(self, provider: StateProvider):
        app = create_app(SimpleNamespace(state_provider=provider, token=TOKEN))
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> str:
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)
        (socket,) = self.server.servers[0].sockets
        return f"http://127.0.0.1:{socket.getsockname()[1]}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join()


class BrowserDriver:
    name = "browser"

    def __init__(self, provider: StateProvider):
        self.provider = provider
        self.server = _Server(provider)
        self.page: Page
        #: Address of the server
        self.url: str
        #: Messages the page sent over the WebSocket
        self.sent: list[dict] = []

    async def __aenter__(self) -> "BrowserDriver":
        url = self.url = self.server.start()
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        context = await self._browser.new_context(
            viewport={"width": 1600, "height": 1000}
        )
        await context.add_cookies(
            [{"name": "experimaestro_token", "value": TOKEN, "url": url}]
        )
        self.page = await context.new_page()
        self.page.on("websocket", self._record)
        await self.page.goto(f"{url}/index.html")
        await self.page.locator("tr[data-experiment-id]").first.wait_for()
        await self.settle()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._browser.close()
        await self._playwright.stop()
        self.server.stop()

    # --- Helpers

    def _record(self, websocket: WebSocket) -> None:
        websocket.on("framesent", lambda frame: self.sent.append(json.loads(frame)))

    def requests(self, msg_type: str) -> list[dict]:
        """Payloads of the requests of a type the page sent so far"""
        return [m.get("payload", {}) for m in self.sent if m["type"] == msg_type]

    async def settle(self, quiet: int = 150) -> None:
        """Wait until the page stops changing"""
        await self.page.evaluate(_SETTLE_JS, [quiet, 5000])

    async def _nav(self, label: str) -> None:
        await self.page.locator("a.nav-link", has_text=label).click()
        await self.settle()

    async def _close_modals(self) -> None:
        while await self.page.locator(".modal.show").count():
            await self.page.keyboard.press("Escape")
            await self.settle()

    async def _expect_toast(self, start: int, confirm: str | None) -> Notification:
        """Confirms the modal (if any), then returns the new toast"""
        if confirm is not None:
            modal = self.page.locator(".modal.show")
            await modal.wait_for(timeout=2000)
            await modal.get_by_role("button", name=confirm, exact=True).click()
        toasts = self.page.locator(".toast")
        await toasts.nth(start).wait_for(timeout=5000)
        toast = toasts.last
        classes = (await toast.get_attribute("class") or "").split()
        (severity,) = [_TOAST_CLASSES[c] for c in classes if c in _TOAST_CLASSES]
        message = await toast.locator(".toast-body").inner_text()
        await self.settle()
        return Notification(severity, message.strip())

    def _job_row(self, job_id: str) -> Locator:
        return self.page.locator(f"tr.job-row[data-job-id='{job_id}']")

    async def _job_action(
        self, job_id: str, title: str, confirm: str
    ) -> Notification | None:
        await self._close_modals()
        action = self._job_row(job_id).locator(f"i[title='{title}']")
        if not await action.count():
            return None
        start = await self.page.locator(".toast").count()
        await action.click()
        return await self._expect_toast(start, confirm)

    # --- Experiments

    async def _show_experiments(self) -> None:
        await self._close_modals()
        await self._nav("Experiments")

    def _experiment_row(self, experiment_id: str) -> Locator:
        return self.page.locator(f"tr[data-experiment-id='{experiment_id}']")

    async def experiments(self) -> dict[str, str]:
        await self._show_experiments()
        rows = self.page.locator("tr[data-experiment-id]")
        return {
            await row.get_attribute("data-experiment-id"): await row.get_attribute(
                "data-status"
            )
            for row in await rows.all()
        }

    async def experiment_order(self) -> list[str]:
        await self._show_experiments()
        rows = self.page.locator("tr[data-experiment-id]")
        return [
            await row.get_attribute("data-experiment-id") for row in await rows.all()
        ]

    async def sort_experiments(self, column: Literal["status", "started", "id"]):
        await self._show_experiments()
        # Shortcuts are ignored in text fields
        await self.page.evaluate("document.activeElement?.blur()")
        await self.page.keyboard.press(
            {"status": "S", "started": "D", "id": "O"}[column]
        )
        await self.settle()

    async def search_experiments(self, text: str) -> None:
        await self._show_experiments()
        await self.page.evaluate("document.activeElement?.blur()")
        await self.page.keyboard.press("/")
        search = self.page.locator("#searchexperiments:focus")
        if text:
            await search.fill(text)
        else:
            await search.press("Escape")
        await self.settle()

    async def experiment_tags(self) -> dict[str, set[str]]:
        await self._show_experiments()
        tags = {}
        for row in await self.page.locator("tr[data-experiment-id]").all():
            exp_id = await row.get_attribute("data-experiment-id")
            tags[exp_id] = set(await row.locator(".experiment-tag").all_inner_texts())
        return tags

    async def group_experiments(
        self, grouping: Literal["none", "prefix", "tag"]
    ) -> dict[str, set[str]]:
        await self._show_experiments()
        await self.page.locator("#groupexperiments").select_option(grouping)
        await self.settle()
        groups: dict[str, set[str]] = {}
        for body in await self.page.locator("table.experiments-table tbody").all():
            header = body.locator("tr.experiment-group-row")
            name = (
                await header.get_attribute("data-group") if await header.count() else ""
            )
            for row in await body.locator("tr[data-experiment-id]").all():
                groups.setdefault(name, set()).add(
                    await row.get_attribute("data-experiment-id")
                )
        return groups

    async def runs(self, experiment_id: str) -> list[str]:
        await self.open_experiment(experiment_id)
        selector = self.page.locator("select[title='Run']")
        if not await selector.count():
            # Not shown for a single run
            return [self.provider.get_current_run(experiment_id)]
        values = await selector.locator("option").evaluate_all(
            "options => options.map(o => o.value)"
        )
        return [value for value in values if value]

    async def open_experiment(
        self, experiment_id: str, run_id: str | None = None
    ) -> None:
        await self._show_experiments()
        # Second cell: the first ones open the experiment, the last has actions
        await self._experiment_row(experiment_id).locator("td").nth(1).click()
        await self.settle()
        if run_id is not None:
            selector = self.page.locator("select[title='Run']")
            await selector.wait_for()
            await selector.select_option(run_id)
            await self.settle()

    async def _experiment_action(
        self, experiment_id: str, title: str, confirm: str
    ) -> Notification | None:
        await self._show_experiments()
        action = self._experiment_row(experiment_id).locator(
            f"i[title='{title}']:not(.disabled)"
        )
        if not await action.count():
            return None
        start = await self.page.locator(".toast").count()
        await action.click()
        return await self._expect_toast(start, confirm)

    async def kill_experiment(self, experiment_id: str) -> Notification | None:
        return await self._experiment_action(
            experiment_id, "Kill all running jobs", "Kill"
        )

    async def delete_experiment(self, experiment_id: str) -> Notification | None:
        return await self._experiment_action(
            experiment_id, "Delete experiment", "Delete"
        )

    # --- Jobs

    async def visible_jobs(self) -> set[str]:
        await self._close_modals()
        await self.settle()
        rows = self.page.locator("tr.job-row[data-job-id]")
        return {await row.get_attribute("data-job-id") for row in await rows.all()}

    async def job_order(self) -> list[str]:
        await self._close_modals()
        await self.settle()
        rows = self.page.locator("tr.job-row[data-job-id]")
        return [await row.get_attribute("data-job-id") for row in await rows.all()]

    async def hide_tags(self, keys: set[str]) -> dict[str, set[str]]:
        await self._close_modals()
        await self.page.locator("#tagvisibility").click()
        for check in await self.page.locator("[data-tag-key]").all():
            await check.set_checked(
                await check.get_attribute("data-tag-key") not in keys
            )
        await self.page.keyboard.press("Escape")
        await self.settle()

        shown = {}
        for row in await self.page.locator("tr.job-tags-row[data-tags-of]").all():
            names = await row.locator(".tag .name").all_inner_texts()
            shown[await row.get_attribute("data-tags-of")] = set(names)
        return shown

    async def sort_by(self, column: Literal["task", "status"]) -> None:
        await self._close_modals()
        await self.page.locator("#sortkey").select_option(column)
        direction = self.page.locator("button[title='Toggle sort direction']")
        # Statuses are listed by rank in the default (descending) direction
        wanted = "down" if column == "status" else "up"
        if not await direction.locator(f".fa-arrow-{wanted}").count():
            await direction.click()
        await self.settle()

    async def group_by(self, key: Literal["status", "task"]) -> dict[str, set[str]]:
        await self._close_modals()
        await self.page.locator("#groupby").select_option(key)
        await self.settle()
        groups = {}
        for body in await self.page.locator("table.jobs-table tbody").all():
            label = await body.locator(".job-group-row .fw-semibold").inner_text()
            rows = await body.locator("tr.job-row[data-job-id]").all()
            groups[label.strip()] = {
                await row.get_attribute("data-job-id") for row in rows
            }
        await self.page.locator("#groupby").select_option("none")
        await self.settle()
        return groups

    async def set_filter(self, query: str) -> str | None:
        await self._close_modals()
        search = self.page.locator("#searchjobs")
        await search.fill(query)
        # The query is sent (debounced) to the server, the answer marks the
        # input as valid or invalid
        await self.page.locator(
            "#searchjobs.is-valid, #searchjobs.is-invalid"
        ).wait_for()
        await self.settle()
        if "is-invalid" in (await search.get_attribute("class") or ""):
            error = self.page.locator(".search .text-danger")
            return (await error.inner_text()).strip()
        return None

    async def _show_completions(self, query: str) -> list[str]:
        """Types a filter and asks for its completions (Ctrl+Space)"""
        await self._close_modals()
        search = self.page.locator("#searchjobs")
        await search.fill(query)
        await search.press("End")
        await search.press("Control+Space")
        await self.settle()
        options = self.page.locator("#searchjobs-completions [role=option]")
        return [
            (await option.locator(".completion-label").inner_text()).strip()
            for option in await options.all()
        ]

    async def filter_completions(self, query: str) -> list[str]:
        labels = await self._show_completions(query)
        # Leaving the input closes the suggestions
        await self.page.locator("#searchjobs").blur()
        return labels

    async def accept_filter_completion(self, query: str, label: str) -> str:
        labels = await self._show_completions(query)
        search = self.page.locator("#searchjobs")
        for _ in range(labels.index(label)):
            await search.press("ArrowDown")
        await search.press("Tab")
        await self.settle()
        value = await search.input_value()
        await search.blur()
        return value

    async def job_detail(self, job_id: str) -> JobDetail:
        await self._close_modals()
        await self._job_row(job_id).locator("i[title='Details']").click()
        modal = self.page.locator(".modal.show")
        await modal.wait_for()
        await self.settle()
        status = modal.locator(".job-detail-status")
        tags = {}
        for tag in await modal.locator(".job-detail-tags .tag").all():
            tags[await tag.locator(".name").inner_text()] = await tag.locator(
                ".value"
            ).inner_text()
        dependencies = {}
        for dep in await modal.locator(".dependency[data-job-id]").all():
            state = await dep.get_attribute("data-status")
            dependencies[await dep.get_attribute("data-job-id")] = (
                None if state == "unknown" else state
            )
        detail = JobDetail(
            job_id=job_id,
            task_id=(await modal.locator(".modal-title").inner_text()).strip(),
            state=await status.get_attribute("data-status"),
            tags=tags,
            dependencies=dependencies,
        )
        await self._close_modals()
        return detail

    async def job_log(self, job_id: str, stream: Literal["stdout", "stderr"]) -> str:
        await self._close_modals()
        await self._job_row(job_id).locator("i[title='Logs']").click()
        modal = self.page.locator(".modal.show")
        await modal.wait_for()
        await modal.get_by_role("tab", name=stream, exact=True).click()
        log = modal.locator("pre.log-viewer-content:visible")
        await log.wait_for()
        await self.page.wait_for_function(
            "el => el.textContent !== '(no output)'", arg=await log.element_handle()
        )
        text = await log.text_content()
        await self._close_modals()
        return text

    async def kill_job(self, job_id: str) -> Notification | None:
        return await self._job_action(job_id, "Kill", "Kill")

    async def delete_job(self, job_id: str) -> Notification | None:
        return await self._job_action(job_id, "Delete", "Delete")

    # --- Orphans

    async def _show_orphans(self) -> None:
        await self._close_modals()
        await self._nav("Orphans")

    async def orphans(self) -> dict[str, OrphanRow]:
        await self._show_orphans()
        rows: dict[str, OrphanRow] = {}
        for row in await self.page.locator("tr[data-job-id]").all():
            rows[await row.get_attribute("data-job-id")] = OrphanRow(
                stray=await row.get_attribute("data-stray") == "true",
                size=int(await row.get_attribute("data-size")),
            )
        return rows

    async def _orphan_action(self, job_id: str, title: str) -> Notification | None:
        await self._show_orphans()
        action = self.page.locator(
            f"tr[data-job-id='{job_id}'] button[title='{title}']"
        )
        if not await action.count():
            return None
        start = await self.page.locator(".toast").count()
        await action.click()
        return await self._expect_toast(start, None)

    async def kill_orphan(self, job_id: str) -> Notification | None:
        return await self._orphan_action(job_id, "Kill")

    async def delete_orphan(self, job_id: str) -> Notification | None:
        return await self._orphan_action(job_id, "Delete")

    async def _orphans_bulk(self, label: str, confirm: str) -> Notification | None:
        await self._show_orphans()
        button = self.page.get_by_role("button", name=label)
        if not await button.count():
            return None
        start = await self.page.locator(".toast").count()
        await button.click()
        return await self._expect_toast(start, confirm)

    async def kill_all_strays(self) -> Notification | None:
        return await self._orphans_bulk("Kill all running", "Kill")

    async def delete_all_orphans(self) -> Notification | None:
        return await self._orphans_bulk("Delete all finished", "Delete")

    # --- Services

    async def graph(
        self, experiment_id: str, *, tasks_only: bool = True, stack: bool = True
    ) -> set[frozenset[str]]:
        await self.open_experiment(experiment_id)
        await self._nav("Graph")
        await self.page.locator(".xpm-dag-node").first.wait_for()
        await self.page.get_by_label("Jobs only").set_checked(tasks_only)
        await self.page.get_by_label("Stack similar nodes").set_checked(stack)
        await self.settle()
        return {
            frozenset((await node.get_attribute("data-members") or "").split())
            for node in await self.page.locator(".xpm-dag-node").all()
        }

    async def services(self, experiment_id: str) -> dict[str, str]:
        await self.open_experiment(experiment_id)
        await self._nav("Services")
        services = {}
        rows = self.page.locator(f"tr[data-experiment-id='{experiment_id}']")
        for row in await rows.all():
            services[await row.get_attribute("data-service-id")] = (
                (await row.locator("td").nth(0).inner_text()).strip().lower()
            )
        return services

    async def _service_action(
        self, experiment_id: str, service_id: str, label: str
    ) -> Notification | None:
        await self.services(experiment_id)
        row = self.page.locator(
            f"tr[data-experiment-id='{experiment_id}'][data-service-id='{service_id}']"
        )
        button = row.get_by_role("button", name=label, exact=True)
        if not await button.count():
            return None
        start = await self.page.locator(".toast").count()
        await button.click()
        return await self._expect_toast(start, None)

    async def active_services(self) -> dict[str, str]:
        await self._close_modals()
        await self.settle()
        active = {}
        for item in await self.page.locator(".active-services .active-service").all():
            key = "/".join(
                [
                    await item.get_attribute("data-experiment-id"),
                    await item.get_attribute("data-service-id"),
                ]
            )
            active[key] = (await item.locator(".badge").inner_text()).lower()
        return active

    async def start_service(
        self, experiment_id: str, service_id: str
    ) -> Notification | None:
        return await self._service_action(experiment_id, service_id, "Start")

    async def stop_service(
        self, experiment_id: str, service_id: str
    ) -> Notification | None:
        return await self._service_action(experiment_id, service_id, "Stop")

    async def connection_state(self) -> str | None:
        await self.settle()
        banner = self.page.locator(".provider-connection")
        if await banner.count() == 0:
            return None
        return await banner.get_attribute("data-state")
