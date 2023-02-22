from datetime import datetime
import logging
import asyncio
import socket
import uuid
from experimaestro.scheduler.base import Job
import pkg_resources
import http
import threading
from typing import Optional, Tuple
from experimaestro.scheduler import Scheduler, Listener as BaseListener
from experimaestro.settings import ServerSettings
from flask import Flask, Response
from flask import request, redirect
from flask_socketio import SocketIO, emit, ConnectionRefusedError


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


class Listener(BaseListener):
    def __init__(self, scheduler: Scheduler, socketio):
        self.scheduler = scheduler
        self.socketio = socketio
        self.scheduler.addlistener(self)

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


def start_app(server: "Server"):
    app = Flask("experimaestro")
    socketio = SocketIO(app, path="/api")
    listener = Listener(server.scheduler, socketio)

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
        emit(
            "services",
            {
                id: service.description()
                for id, service in listener.scheduler.xp.services.items()
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

    @app.route("/notifications/<path:jobid>/progress")
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
    if server.port is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        server.port = sock.getsockname()[1]
        sock.close()

    logging.info(
        "Web server started on http://%s:%d?token=%s",
        server.host,
        server.port,
        server.token,
    )

    server.instance = socketio
    socketio.run(
        app, host=server.host, port=server.port, debug=True, use_reloader=False
    )
    logging.info("Web server stopped")


class Server:
    def __init__(self, scheduler: Scheduler, settings: ServerSettings):
        if settings.host is None or settings.host == "127.0.0.1":
            self.bindinghost = "127.0.0.1"
        else:
            self.bindinghost = "0.0.0.0"

        self.host = settings.host or "127.0.0.1"
        self.port = settings.port
        self.scheduler = scheduler
        self.token = settings.token or uuid.uuid4().hex
        self.instance = None

    def getNotificationSpec(self) -> Tuple[str, str]:
        """Returns a tuple (server ID, server URL)"""
        return (
            f"""{self.host}_{self.port}.url""",
            f"""http://{self.host}:{self.port}/notifications""",
        )

    def stop(self):
        if self.instance and self.instance.wsgi_server:
            self.instance.wsgi_server.shutdown()

    def start(self):
        """Start the websocket server in a new process process"""
        logging.info("Starting the web server")
        self.thread = threading.Thread(target=start_app, args=(self,)).start()
