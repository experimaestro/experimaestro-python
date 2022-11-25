import logging
import asyncio
import urllib
from urllib.parse import parse_qs, urlparse
import uuid
from experimaestro.server.auth import get_cookie
import pkg_resources
import websockets
import websockets.exceptions
import websockets.http
import http
import json
import threading
from typing import Optional, Tuple
import time
import functools
from json import JSONEncoder
import re
from experimaestro.scheduler import Scheduler, Listener as BaseListener
from experimaestro.notifications import LevelInformation


class XPMJsonEncoder(JSONEncoder):
    """Handles simple serialization cases"""

    def default(self, o):
        if isinstance(o, LevelInformation):
            return {"level": o.level, "progress": o.progress, "desc": o.desc}
        super().default(o)


def json_dumps(o):
    return XPMJsonEncoder().encode(o)


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


class Listener(BaseListener):
    def __init__(self, loop, scheduler: Scheduler):
        self.loop = loop
        self.scheduler = scheduler
        self.scheduler.addlistener(self)
        self.websockets = set()

    def send_message(self, message):
        if self.websockets:
            message = json_dumps(message)
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


async def handler(websocket, path, listener: Listener):
    if not websocket.authorized:
        await websocket.send("unauthorized")
        return

    await listener.register(websocket)
    try:
        while True:
            try:
                action = json.loads(await websocket.recv())
                actiontype = action["type"]
                assert isinstance(actiontype, str)
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception:
                await websocket.send(
                    json_dumps({"error": True, "message": "message parsing error"})
                )
                continue

            if actiontype == "refresh":
                for job in listener.scheduler.jobs.values():
                    await websocket.send(json_dumps(job_create(job)))
            elif actiontype == "quit":
                break
            elif actiontype == "details":
                jobid = action["payload"]
                await websocket.send(
                    json_dumps(job_details(listener.scheduler.jobs[jobid]))
                )
            elif actiontype == "kill":
                jobid = action["payload"]
                job = listener.scheduler.jobs[jobid]
                process = await job.aio_process()
                if process is not None:
                    process.kill()
            else:
                await websocket.send(
                    json_dumps(
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


class ServerProtocol(websockets.WebSocketServerProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = None

    async def process_request(self, path, headers):
        # if "Upgrade" in headers:
        if path == "/api" or path == "/ws":
            self.authorized = False
            token = get_cookie(headers.get("Cookie", ""), "token")
            if token == self.token:
                self.authorized = True

            return None

        resp_headers = websockets.http.Headers()

        # Detect a token and redirect (to login or page)
        if path.startswith("/notifications/"):
            m = re.match(r"^/notifications/([a-z0-9]+)/progress\?(.*)$", path)
            if m:
                jobid = m.group(1)
                params = urllib.parse.parse_qs(m.group(2))
                try:
                    self.scheduler.jobs[jobid].set_progress(
                        int(params["level"][0]),
                        float(params["progress"][0]),
                        params.get("desc", [None])[0],
                    )
                except KeyError:
                    # Just ignore
                    pass
                return (http.HTTPStatus.OK, headers, "")

        if path != "/login.html":
            url = urlparse(path)
            if token := parse_qs(url.query).get("token", ""):
                resp_headers["Set-Cookie"] = f"token={token[0]}"
                resp_headers["Location"] = f"{url.path}"
                return (
                    http.HTTPStatus.TEMPORARY_REDIRECT,
                    resp_headers,
                    b"Redirection\n",
                )

            token = get_cookie(headers.get("Cookie", ""), "token")
            if token != self.token:
                resp_headers[
                    "Set-Cookie"
                ] = "token=;expires=Thu, 01 Jan 1970 00:00:00 GMT"
                resp_headers["Location"] = f"/login.html"
                return http.HTTPStatus.TEMPORARY_REDIRECT, resp_headers, b"Redirect\n"

        if path == "/":
            path = "/index.html"

        datapath = "data%s" % path
        if pkg_resources.resource_exists("experimaestro.server", datapath):
            code = http.HTTPStatus.OK
            resp_headers["Cache-Control"] = "max-age=0"
            mimetype = MIMETYPES[datapath.rsplit(".", 1)[1]]
            resp_headers["Content-Type"] = mimetype
            logging.debug("Reading %s [%s]", datapath, mimetype)
            body = pkg_resources.resource_string("experimaestro.server", datapath)
            return (code, resp_headers, body)

        resp_headers["Content-Type"] = MIMETYPES["txt"]
        return (http.HTTPStatus.NOT_FOUND, resp_headers, "No such path %s" % path)


class ServerProtocolFactory:
    def __init__(self, scheduler: Scheduler, token: str):
        self.scheduler = scheduler
        self.token = token

    def __call__(self, *args, **kwargs):
        protocol = ServerProtocol(*args, **kwargs)
        protocol.scheduler = self.scheduler
        protocol.token = self.token
        return protocol


class Server:
    def __init__(self, scheduler: Scheduler, port: int, *, host=None):
        self.bindinghost = (
            "127.0.0.1" if (host is None or host == "127.0.0.1") else "0.0.0.0"
        )
        self.host = host or "127.0.0.1"
        self._port = port
        self.scheduler = scheduler
        self._loop = None
        self._stop = None
        self.token = uuid.uuid4().hex

    def getNotificationSpec(self) -> Tuple[str, str]:
        """Returns a tuple (server ID, server URL)"""
        # TODO: return fqdn if the job is from outside
        return (
            f"""{self.host}_{self.port}.url""",
            f"""http://{self.host}:{self.port}/notifications""",
        )

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
        self._port_future: asyncio.futures.Future[int] = self._loop.create_future()
        threading.Thread(target=run, args=(self._loop, self._stop)).start()

    def serve(self):
        # asyncio.ensure_future(self._serve())
        pass

    async def aio_port(self):
        return await self._port_future

    @property
    def port(self):
        """Returns the running server port"""
        return asyncio.run_coroutine_threadsafe(self.aio_port(), self._loop).result()

    async def _serve(self, stop):
        bound_handler = functools.partial(handler, listener=self.listener)
        logger = logging.getLogger("websockets.server")
        logger.setLevel(logging.WARNING)
        async with websockets.serve(
            bound_handler,
            self.bindinghost,
            self._port,  # process_request=process_request,
            create_protocol=ServerProtocolFactory(self.scheduler, self.token),
        ) as server:
            self._port_future.set_result(server.sockets[0].getsockname()[1])
            logging.info(
                "Webserver started on http://%s:%d?token=%s",
                self.host,
                self._port,
                self.token,
            )
            await stop
        logging.info("Server stopped")
