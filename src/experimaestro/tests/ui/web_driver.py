"""UI driver for the web interface, at the WebSocket protocol level

Talks to the real FastAPI app (``create_app``) through Starlette's
``TestClient``, as the React client does: it sends the same requests and keeps
the jobs, experiments and orphans the server pushed, the way the Redux store
does. Logic implemented in the browser (sorting, grouping, formatting) is out
of reach at this level.

The server sends no acknowledgement, so after each request the driver sends a
*barrier* (an empty ``jobs.filter`` request, which is always answered) and
reads messages until its answer arrives: the connection handles messages in
order, so everything the request produced has been received by then.
"""

import itertools
from types import SimpleNamespace
from typing import Any, Literal

from starlette.testclient import TestClient

from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.webui.app import create_app

from .drivers import SEVERITIES, JobDetail, Notification, OrphanRow

TOKEN = "test-token"


class WebDriver:
    name = "web"

    def __init__(self, provider: StateProvider):
        self.provider = provider
        server = SimpleNamespace(state_provider=provider, token=TOKEN)
        self.client = TestClient(create_app(server))
        self.client.cookies.set("experimaestro_token", TOKEN)
        self._request_ids = itertools.count(1)

        # Client state (what the Redux store holds)
        self.experiment_list: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.orphan_jobs: dict[str, dict[str, Any]] = {}
        #: Services by key ("<experiment>/<service>")
        self.services_by_key: dict[str, dict[str, Any]] = {}
        self.notifications: list[Notification] = []
        self._runs: dict[str, list[dict[str, Any]]] = {}
        self._filter: list[str] | None = None
        self._experiment: str | None = None
        self._run: str | None = None

    async def __aenter__(self) -> "WebDriver":
        self.client.__enter__()
        self.ws = self.client.websocket_connect(f"/ws?token={TOKEN}").__enter__()
        await self._request("experiments")
        return self

    async def __aexit__(self, *exc) -> None:
        self.ws.__exit__(None, None, None)
        self.client.__exit__(None, None, None)

    # --- Protocol

    def _receive(self, message: dict[str, Any]) -> None:
        """Apply a server message to the client state"""
        payload = message.get("payload", {})
        match message["type"]:
            case "experiment.add":
                self.experiment_list[payload["experiment_id"]] = payload
            case "experiment.removed":
                self.experiment_list.pop(payload["experimentId"], None)
            case "experiment.runs":
                self._runs[payload["experimentId"]] = payload["runs"]
            case "job.add":
                self.jobs[payload["jobId"]] = payload
            case "job.update":
                if payload["jobId"] in self.jobs:
                    self.jobs[payload["jobId"]].update(payload)
            case "job.removed":
                self.jobs.pop(payload["jobId"], None)
            case "orphans":
                self.orphan_jobs = {job["jobId"]: job for job in payload["jobs"]}
            case "orphan.removed":
                self.orphan_jobs.pop(payload["jobId"], None)
            case "service.add" | "service.update":
                self.services_by_key.setdefault(payload["key"], {}).update(payload)
            case "notify":
                self.notifications.append(
                    Notification(SEVERITIES[payload["severity"]], payload["message"])
                )
            case "error":
                self.notifications.append(Notification("error", payload["message"]))

    async def _request(self, msg_type: str, **payload) -> list[dict[str, Any]]:
        """Send a request; returns the messages received in response"""
        self.ws.send_json({"type": msg_type, "payload": payload})
        barrier = next(self._request_ids)
        self.ws.send_json(
            {"type": "jobs.filter", "payload": {"query": "", "requestId": barrier}}
        )
        received = []
        while True:
            message = self.ws.receive_json()
            if (
                message["type"] == "jobs.filter"
                and message["payload"]["requestId"] == barrier
            ):
                return received
            received.append(message)
            self._receive(message)

    async def _notification(self, msg_type: str, **payload) -> Notification:
        start = len(self.notifications)
        await self._request(msg_type, **payload)
        new = self.notifications[start:]
        assert new, f"no notification for {msg_type}"
        return new[-1]

    def _task_id(self, job_id: str) -> str:
        job = self.jobs.get(job_id) or self.orphan_jobs[job_id]
        return job["taskId"]

    # --- Experiments

    async def experiments(self) -> dict[str, str]:
        await self._request("experiments")
        return {
            xp_id: payload["statusCategory"]
            for xp_id, payload in self.experiment_list.items()
        }

    async def runs(self, experiment_id: str) -> list[str]:
        await self._request("experiment.runs", experimentId=experiment_id)
        return [run["run_id"] for run in self._runs[experiment_id]]

    async def open_experiment(
        self, experiment_id: str, run_id: str | None = None
    ) -> None:
        # As the client does when switching experiment/run
        self.jobs.clear()
        self._filter = None
        self._experiment, self._run = experiment_id, run_id
        payload = {"experimentId": experiment_id}
        if run_id is not None:
            payload["runId"] = run_id
        await self._request("refresh", **payload)

    async def kill_experiment(self, experiment_id: str) -> Notification:
        return await self._notification("experiment.kill", experimentId=experiment_id)

    async def delete_experiment(self, experiment_id: str) -> Notification:
        return await self._notification("experiment.delete", experimentId=experiment_id)

    # --- Jobs

    async def visible_jobs(self) -> set[str]:
        assert self._experiment is not None, "no experiment open"
        job_ids = {
            job_id
            for job_id, job in self.jobs.items()
            if self._experiment in job["experimentIds"]
        }
        if self._filter is not None:
            job_ids &= set(self._filter)
        return job_ids

    # Sorting and grouping are done by the client: see the browser driver

    async def job_order(self) -> list[str]:
        raise NotImplementedError("job order is set by the browser client")

    async def sort_by(self, column: Literal["task", "status"]) -> None:
        raise NotImplementedError("sorting is done by the browser client")

    async def group_by(self, key: Literal["status", "task"]) -> dict[str, set[str]]:
        raise NotImplementedError("grouping is done by the browser client")

    async def set_filter(self, query: str) -> str | None:
        request_id = next(self._request_ids)
        payload = {"query": query, "requestId": request_id}
        if self._experiment is not None:
            payload["experimentId"] = self._experiment
        if self._run is not None:
            payload["runId"] = self._run
        (response,) = [
            m["payload"]
            for m in await self._request("jobs.filter", **payload)
            if m["type"] == "jobs.filter" and m["payload"]["requestId"] == request_id
        ]
        if response["error"]:
            return response["error"]
        self._filter = response["jobIds"]
        return None

    async def job_detail(self, job_id: str) -> JobDetail:
        await self._request("job.details", jobId=job_id, taskId=self._task_id(job_id))
        job = self.jobs[job_id]
        return JobDetail(
            job_id=job["jobId"],
            task_id=job["taskId"],
            state=job["status"],
            tags={key: value for key, value in job["tags"]},
            dependencies={
                dep: self.jobs[dep]["status"] if dep in self.jobs else None
                for dep in job["dependsOn"]
            },
        )

    async def job_log(self, job_id: str, stream: Literal["stdout", "stderr"]) -> str:
        response = self.client.get(
            f"/api/jobs/{self._task_id(job_id)}/{job_id}/logs/{stream}"
        )
        response.raise_for_status()
        return response.json()["content"]

    async def kill_job(self, job_id: str) -> Notification:
        return await self._notification(
            "job.kill", jobId=job_id, taskId=self._task_id(job_id)
        )

    async def delete_job(self, job_id: str) -> Notification:
        return await self._notification(
            "job.delete", jobId=job_id, taskId=self._task_id(job_id)
        )

    # --- Orphans

    async def orphans(self) -> dict[str, OrphanRow]:
        await self._request("orphans")
        return {
            job_id: OrphanRow(stray=job["isStray"], size=job["sizeBytes"])
            for job_id, job in self.orphan_jobs.items()
        }

    async def kill_orphan(self, job_id: str) -> Notification:
        return await self._notification(
            "orphan.kill", jobId=job_id, taskId=self._task_id(job_id)
        )

    async def delete_orphan(self, job_id: str) -> Notification:
        return await self._notification(
            "orphan.delete", jobId=job_id, taskId=self._task_id(job_id)
        )

    async def kill_all_strays(self) -> Notification:
        return await self._notification("orphans.kill")

    async def delete_all_orphans(self) -> Notification:
        return await self._notification("orphans.delete")

    # --- Graph

    async def graph(
        self, experiment_id: str, *, tasks_only: bool = True, stack: bool = True
    ) -> set[frozenset[str]]:
        request_id = next(self._request_ids)
        (response,) = [
            m["payload"]
            for m in await self._request(
                "experiment.graph",
                requestId=request_id,
                experimentId=experiment_id,
                config=None,
                tasksOnly=tasks_only,
                stack=stack,
            )
            if m["type"] == "experiment.graph"
            and m["payload"]["requestId"] == request_id
        ]
        nodes = response["graph"]["nodes"]
        return {
            frozenset(nodes[ix]["id"] for ix in node["members"])
            for node in response["view"]["nodes"]
        }

    # --- Services

    async def services(self, experiment_id: str) -> dict[str, str]:
        self.services_by_key.clear()
        await self._request("services", experimentId=experiment_id)
        return {
            service["id"]: service["state"].lower()
            for service in self.services_by_key.values()
            if service["experimentId"] == experiment_id
        }

    async def start_service(self, experiment_id: str, service_id: str) -> Notification:
        return await self._notification(
            "service.start", experimentId=experiment_id, id=service_id
        )

    async def stop_service(self, experiment_id: str, service_id: str) -> Notification:
        return await self._notification(
            "service.stop", experimentId=experiment_id, id=service_id
        )
