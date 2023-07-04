from datetime import datetime
import logging
import asyncio
import platform
import socket
import uuid
from experimaestro.scheduler.base import Job
import pkg_resources
import http
import threading
from typing import Optional, Tuple
from experimaestro.scheduler import Scheduler, Listener as BaseListener
from experimaestro.scheduler.services import Service, ServiceListener
from experimaestro.settings import ServerSettings
from flask import Flask, Request, Response
from flask import request, redirect
from flask_socketio import SocketIO, emit, ConnectionRefusedError
import requests


def formattime(v: Optional[float]):
    if not v:
        return ""

    return datetime.fromtimestamp(v).isoformat()


def progress_state(job: Job):
    return [
        {"level": o.level, "progress": o.progress, "desc": o.desc} for o in job.progress
    ]


def job_details(job):
    return {
        "jobId": job.identifier,
        "taskId": job.name,
        "locator": str(job.jobpath),
        "status": job.state.name.lower(),
        "start": formattime(job.starttime),
        "end": formattime(job.endtime),
        "submitted": formattime(job.submittime),
        "tags": list(job.tags.items()),
        "progress": progress_state(job),
    }


def job_create(job: Job):
    return {
        "jobId": job.identifier,
        "taskId": job.name,
        "locator": str(job.jobpath),
        "status": job.state.name.lower(),
        "tags": list(job.tags.items()),
        "progress": progress_state(job),
    }


class Listener(BaseListener, ServiceListener):
    def __init__(self, scheduler: Scheduler, socketio):
        self.scheduler = scheduler
        self.socketio = socketio
        self.scheduler.addlistener(self)
        self.services = {}
        for service in self.scheduler.xp.services.values():
            self.service_add(service)

    def job_submitted(self, job):
        self.socketio.emit("job.add", job_create(job))

    def job_state(self, job):
        self.socketio.emit(
            "job.update",
            {
                "jobId": job.identifier,
                "status": job.state.name.lower(),
                "progress": progress_state(job),
            },
        )

    def service_add(self, service: Service):
        service.add_listener(self)
        self.services[service.id] = service
        self.socketio.emit(
            "service.add",
            {
                "id": service.id,
                "description": service.description(),
                "state": service.state.name,
            },
        )

    def service_state_changed(self, service: Service):
        self.socketio.emit("service.update", {"state": service.state.name})


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


def proxy_response(base_url: str, request: Request, path: str):
    # Whitelist a few headers to pass on
    request_headers = {}
    for key, value in request.headers.items():
        request_headers[key] = value

    if request.query_string:
        path = f"""{path}?{request.query_string.decode("utf-8")}"""

    data = None
    if request.method == "POST":
        data = request.get_data()

    response = requests.request(
        request.method,
        f"{base_url}{path}",
        data=data,
        stream=True,
        headers=request_headers,
    )
    headers = {}
    for key, value in response.headers.items():
        headers[key] = value

    flask_response = Response(
        response=response.raw.read(),
        status=response.status_code,
        headers=headers,
        content_type=response.headers["content-type"],
    )
    return flask_response


def start_app(server: "Server"):
    logging.debug("Starting Flask server...")
    app = Flask("experimaestro")

    logging.debug("Starting Flask server (SocketIO)...")
    socketio = SocketIO(app, path="/api", async_mode="gevent")
    listener = Listener(server.scheduler, socketio)

    logging.debug("Starting Flask server (setting up socketio)...")

    @socketio.on("connect")
    def handle_connect():
        if server.token != request.cookies.get("experimaestro_token", None):
            raise ConnectionRefusedError("invalid token")

    @socketio.on("refresh")
    def handle_refresh():
        for job in listener.scheduler.jobs.values():
            emit("job.add", job_create(job))

    @socketio.on("job.details")
    def handle_details(jobid):
        emit("job.update", job_details(listener.scheduler.jobs[jobid]))

    @socketio.on("services")
    def handle_services_list():
        for service in listener.services.values():
            emit(
                "service.add",
                {
                    "id": service.id,
                    "description": service.description(),
                    "state": service.state.name,
                },
            )

    @socketio.on("job.kill")
    def handle_job_kill(jobid: str):
        job = server.scheduler.jobs[jobid]
        future = asyncio.run_coroutine_threadsafe(
            job.aio_process(), server.scheduler.loop
        )
        process = future.result()
        if process is not None:
            process.kill()

    logging.debug("Starting Flask server (setting up routes)...")

    @app.route("/services/<path:path>", methods=["GET", "POST"])
    def route_service(path):
        service, *path = path.split("/", 1)
        if not path:
            return redirect(f"/services/{service}/", http.HTTPStatus.PERMANENT_REDIRECT)

        service = server.scheduler.xp.services.get(service, None)
        if service is None:
            return Response(f"Service {service} not found", http.HTTPStatus.NOT_FOUND)

        base_url = service.get_url()
        return proxy_response(base_url, request, path[0] if path else "/")

    @app.route("/notifications/<jobid>/progress")
    def notifications_progress(jobid):
        level = int(request.args.get("level", 0))
        progress = float(request.args.get("progress", 0.0))

        try:
            server.scheduler.jobs[jobid].set_progress(
                level,
                progress,
                request.args.get("desc", None),
            )
        except KeyError:
            # Just ignore
            pass
        return Response("", http.HTTPStatus.OK)

    @app.route("/")
    def route_root():
        if server.token == request.cookies.get("experimaestro_token", None):
            return redirect("/index.html", 302)
        return redirect("/login.html", 302)

    @app.route("/auth")
    def route_auth():
        if token := request.args.get("xpm-token", None):
            if server.token == token:
                resp = redirect("/index.html", 302)
                resp.set_cookie("experimaestro_token", token)
                return resp
        return redirect("/login.html", 302)

    @app.route("/stop")
    def route_stop():
        if (server.token == request.args.get("xpm-token", None)) or (
            server.token == request.cookies.get("experimaestro_token", None)
        ):
            socketio.stop()
            return Response(status=http.HTTPStatus.ACCEPTED)
        return Response(status=http.HTTPStatus.UNAUTHORIZED)

    @app.route("/<path:path>")
    def static_route(path):
        if token := request.form.get("experimaestro_token", None):
            if server.token == token:
                request.cookies["experimaestro_token"] = token

        if path == "index.html":
            if server.token != request.cookies.get("experimaestro_token", None):
                return redirect("/login.html", code=302)

        datapath = "data/%s" % path
        logging.debug("Looking for %s", datapath)
        if pkg_resources.resource_exists("experimaestro.server", datapath):
            mimetype = MIMETYPES[datapath.rsplit(".", 1)[1]]
            content = pkg_resources.resource_string("experimaestro.server", datapath)
            return Response(content, mimetype=mimetype)
        return Response("Page not found", status=404)

    # Start the app
    if server.port is None or server.port == 0:
        logging.info("Searching for an available port")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        server.port = sock.getsockname()[1]
        sock.close()

    logging.info(
        "Web server started on http://%s:%d/auth?xpm-token=%s",
        server.host,
        server.port,
        server.token,
    )

    server.instance = socketio
    with server.cv_running:
        server.running = True
        server.cv_running.notify()
    socketio.run(
        app,
        host=server.host,
        port=server.port,
        debug=False,
        use_reloader=False,
    )
    logging.info("Web server stopped")


class Server:
    def __init__(self, scheduler: Scheduler, settings: ServerSettings):
        if settings.autohost == "fqdn":
            settings.host = socket.getfqdn()
            logging.info("Auto host name (fqdn): %s", settings.host)
        elif settings.autohost == "name":
            settings.host = platform.node()
            logging.info("Auto host name (name): %s", settings.host)

        if settings.host is None or settings.host == "127.0.0.1":
            self.bindinghost = "127.0.0.1"
        else:
            self.bindinghost = "0.0.0.0"

        self.host = settings.host or "127.0.0.1"
        self.port = settings.port
        self.scheduler = scheduler
        self.token = settings.token or uuid.uuid4().hex
        self.instance = None
        self.running = False
        self.cv_running = threading.Condition()

    def getNotificationSpec(self) -> Tuple[str, str]:
        """Returns a tuple (server ID, server URL)"""
        return (
            f"""{self.host}_{self.port}.url""",
            f"""http://{self.host}:{self.port}/notifications""",
        )

    def stop(self):
        if self.instance:
            try:
                requests.get(
                    f"http://{self.host}:{self.port}/stop?xpm-token={self.token}"
                )
            except requests.exceptions.ConnectionError:
                # This is expected
                pass

    def start(self):
        """Start the websocket server in a new process process"""
        logging.info("Starting the web server")

        # Avoids clutering
        logging.getLogger("geventwebsocket.handler").setLevel(logging.WARNING)

        self.thread = threading.Thread(target=start_app, args=(self,)).start()

        # Wait until we really started
        while True:
            with self.cv_running:
                if self.running:
                    break
