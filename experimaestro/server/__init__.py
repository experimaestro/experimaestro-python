import logging
import asyncio
import pkg_resources
import websockets
import websockets.http
import http
import json
import threading
from typing import Optional
import time
import functools
from experimaestro.scheduler import Scheduler
import re


def formattime(v: Optional[float]):
    if not v:
        return ""

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v))


def job_details(job):
    return {
        "type": "JOB_UPDATE",
        "payload": {
            "jobId": job.identifier,
            "taskId": job.name,
            "locator": str(job.jobpath),
            "status": job.state.name.lower(),
            "start": formattime(job.starttime),
            "end": formattime(job.endtime),
            "submitted": formattime(job.submittime),
            "tags": list(job.tags.items()),
            "progress": job.progress,
        },
    }


def job_create(job):
    return {
        "type": "JOB_ADD",
        "payload": {
            "jobId": job.identifier,
            "taskId": job.name,
            "locator": str(job.jobpath),
            "status": job.state.name.lower(),
            "tags": list(job.tags.items()),
            "progress": job.progress,
        },
    }


class Listener:
    def __init__(self, loop, scheduler):
        self.loop = loop
        self.scheduler = scheduler
        self.scheduler.addlistener(self)
        self.websockets = set()

    def send_message(self, message):
        if self.websockets:
            message = json.dumps(message)
            self.loop.call_soon_threadsafe(
                self.loop.create_task,
                asyncio.wait([user.send(message) for user in self.websockets]),
            )

    def job_submitted(self, job):
        self.send_message(job_create(job))

    def job_state(self, job):
        self.send_message(
            {
                "type": "JOB_UPDATE",
                "payload": {
                    "jobId": job.identifier,
                    "status": job.state.name.lower(),
                    "progress": job.progress,
                },
            }
        )
        # self.loop.call_soon_threadsafe(print, "!!!! Job state changed")

    async def register(self, websocket):
        self.websockets.add(websocket)

    async def unregister(self, websocket):
        self.websockets.remove(websocket)


async def register(websocket):
    Scheduler.listeners.add(Listener(websocket))


async def handler(websocket, path, listener):
    await listener.register(websocket)
    try:
        while True:
            try:
                action = json.loads(await websocket.recv())
                actiontype = action["type"]
                assert isinstance(actiontype, str)
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                await websocket.send(
                    json.dumps({"error": True, "message": "message parsing error"})
                )
                continue

            if actiontype == "refresh":
                for job in listener.scheduler.jobs.values():
                    await websocket.send(json.dumps(job_create(job)))
            elif actiontype == "quit":
                break
            elif actiontype == "details":
                jobid = action["payload"]
                await websocket.send(
                    json.dumps(job_details(listener.scheduler.jobs[jobid]))
                )
            elif actiontype == "kill":
                jobid = action["payload"]
                process = listener.scheduler.jobs[jobid].process
                if process is not None:
                    process.kill()
            else:
                await websocket.send(
                    json.dumps(
                        {
                            "error": True,
                            "message": "Unknown message action %s" % actiontype,
                        }
                    )
                )
    finally:
        await listener.unregister(websocket)


MIMETYPES = {
    "html": "text/html",
    "map": "text/plain",
    "txt": "text/plain",
    "ico": "image/x-icon",
    "png": "image/png",
    "css": "text/css",
    "js": "application/javascript",
    "json": "application/json",
    "eot": "font/vnd.ms-fontobject",
    "woff": "font/woff",
    "woff2": "font/woff2",
    "ttf": "font/ttf",
}


class RequestProcessor:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    async def __call__(self, path, request_headers):
        headers = websockets.http.Headers()

        if path == "/ws":
            # Continue HTTP upgrade
            return None

        if path.startswith("/notifications/"):
            m = re.match(r"^/notifications/([a-z0-9]+)/progress/([0-9.]+)$", path)
            if m:
                jobid = m.group(1)
                progress = float(m.group(2))
                try:
                    if progress >= 0 and progress <= 1.0:
                        self.scheduler.jobs[jobid].progress = progress
                except KeyError:
                    # Just ignore
                    pass
                return (http.HTTPStatus.OK, headers, "")

        if path == "/":
            path = "/index.html"

        datapath = "data%s" % path
        if pkg_resources.resource_exists("experimaestro.server", datapath):
            code = http.HTTPStatus.OK
            headers["Cache-Control"] = "max-age=0"
            mimetype = MIMETYPES[datapath.rsplit(".", 1)[1]]
            headers["Content-Type"] = mimetype
            logging.info("Reading %s [%s]", datapath, mimetype)
            body = pkg_resources.resource_string("experimaestro.server", datapath)
            return (code, headers, body)

        headers["Content-Type"] = MIMETYPES["txt"]
        return (http.HTTPStatus.NOT_FOUND, headers, "No such path %s" % path)


class Server:
    def __init__(self, scheduler: Scheduler, port: int, *, host=None):
        self.bindinghost = (
            "127.0.0.1" if host is None or host == "127.0.0.1" else "0.0.0.0"
        )
        self.host = host or "127.0.0.1"
        self.port = port
        self.scheduler = scheduler
        self._loop = None
        self._stop = None

    def getNotificationURL(self):
        return f"""http://{self.host}:{self.port}/notifications"""

    def stop(self):
        if self._stop:
            logging.info("Stopping server")
            self._loop.call_soon_threadsafe(self._stop.set_result, True)
            self.listener.scheduler.removelistener(self.listener)

    def start(self):
        """Start the websocket server in a new process process"""

        def run(loop, stop):
            self.listener = Listener(loop, self.scheduler)
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve(stop))

        self._loop = asyncio.new_event_loop()
        self._stop = self._loop.create_future()
        threading.Thread(target=run, args=(self._loop, self._stop)).start()

    def serve(self):
        # asyncio.ensure_future(self._serve())
        pass

    async def _serve(self, stop):
        logging.info("Webserver started on http://%s:%d", self.host, self.port)
        bound_handler = functools.partial(handler, listener=self.listener)
        process_request = RequestProcessor(self.scheduler)
        async with websockets.serve(
            bound_handler, self.bindinghost, self.port, process_request=process_request
        ):
            await stop
        logging.info("Server stopped")
