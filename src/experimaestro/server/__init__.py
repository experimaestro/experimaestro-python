from datetime import datetime
import logging
import asyncio
import platform
import socket
import uuid
from experimaestro.scheduler.base import Job
import sys
import http
import threading
from typing import Optional, Tuple, ClassVar

from importlib.resources import files
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
    # Get experiment IDs from job.experiments list
    experiment_ids = [xp.workdir.name for xp in job.experiments]

    return {
        "jobId": job.identifier,
        "taskId": job.name,
        "locator": str(job.jobpath),
        "status": job.state.name.lower(),
        "tags": list(job.tags.items()),
        "progress": progress_state(job),
        "experimentIds": experiment_ids,  # Add experiment IDs
    }


class Listener(BaseListener, ServiceListener):
    def __init__(self, socketio, state_provider):
        self.socketio = socketio
        self.state_provider = state_provider

        # Try to get the scheduler (if one is running for active experiments)
        # Otherwise we're in monitoring mode and don't need scheduler events
        try:
            from experimaestro.scheduler import Scheduler

            # Check if a scheduler instance exists (would be created if experiments are running)
            if Scheduler._instance is not None:
                self.scheduler = Scheduler._instance
                self.scheduler.addlistener(self)
                self.services = {}
                # Initialize services from all registered experiments
                for xp in self.scheduler.experiments.values():
                    for service in xp.services.values():
                        self.service_add(service)
            else:
                # No scheduler running - monitoring mode
                self.scheduler = None
                self.services = {}
        except Exception:
            # Scheduler not available - monitoring mode
            self.scheduler = None
            self.services = {}

    def job_submitted(self, job):
        self.socketio.emit("job.add", job_create(job))

    def job_state(self, job):
        experiment_ids = [xp.workdir.name for xp in job.experiments]
        self.socketio.emit(
            "job.update",
            {
                "jobId": job.identifier,
                "status": job.state.name.lower(),
                "progress": progress_state(job),
                "experimentIds": experiment_ids,
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


# flake8: noqa: C901
def start_app(server: "Server"):
    logging.debug("Starting Flask server...")
    app = Flask("experimaestro")

    logging.debug("Starting Flask server (SocketIO)...")
    socketio = SocketIO(app, path="/api", async_mode="gevent")
    listener = Listener(socketio, server.state_provider)

    logging.debug("Starting Flask server (setting up socketio)...")

    @socketio.on("connect")
    def handle_connect():
        if server.token != request.cookies.get("experimaestro_token", None):
            raise ConnectionRefusedError("invalid token")

    @socketio.on("refresh")
    def handle_refresh(experiment_id=None):
        """Refresh jobs for an experiment (or all experiments if None)"""
        if experiment_id:
            # Refresh specific experiment
            jobs = listener.state_provider.get_jobs(experiment_id)
            for job_data in jobs:
                emit("job.add", job_data)
        else:
            # Refresh all experiments
            if listener.scheduler:
                # Active experiments: get jobs from scheduler
                for job in listener.scheduler.jobs.values():
                    emit("job.add", job_create(job))
            else:
                # Monitoring mode: get jobs from WorkspaceStateProvider
                for exp in listener.state_provider.get_experiments():
                    exp_id = exp["experiment_id"]
                    jobs = listener.state_provider.get_jobs(exp_id)
                    for job_data in jobs:
                        emit("job.add", job_data)

    @socketio.on("experiments")
    def handle_experiments():
        """List all experiments"""
        experiments = listener.state_provider.get_experiments()
        for exp in experiments:
            emit("experiment.add", exp)

    @socketio.on("job.details")
    def handle_details(data):
        """Get job details - expects {experimentId, jobId} or just jobId (backward compat)"""
        # Backward compatibility: if data is a string, treat it as jobId
        if isinstance(data, str):
            jobid = data
            if listener.scheduler:
                emit("job.update", job_details(listener.scheduler.jobs[jobid]))
        else:
            experiment_id = data.get("experimentId")
            job_id = data.get("jobId")
            job_data = listener.state_provider.get_job(experiment_id, job_id)
            if job_data:
                emit("job.update", job_data)

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
    def handle_job_kill(data):
        """Kill a job - expects {experimentId, jobId} or just jobId (backward compat)"""
        # Backward compatibility: if data is a string, treat it as jobId
        if isinstance(data, str):
            jobid = data
            if listener.scheduler:
                job = listener.scheduler.jobs[jobid]
                future = asyncio.run_coroutine_threadsafe(
                    job.aio_process(), listener.scheduler.loop
                )
                process = future.result()
                if process is not None:
                    process.kill()
        else:
            experiment_id = data.get("experimentId")
            job_id = data.get("jobId")
            try:
                listener.state_provider.kill_job(experiment_id, job_id)
            except NotImplementedError:
                logging.warning("kill_job not supported for this state provider")

    logging.debug("Starting Flask server (setting up routes)...")

    @app.route("/services/<path:path>", methods=["GET", "POST"])
    def route_service(path):
        service, *path = path.split("/", 1)
        if not path:
            return redirect(f"/services/{service}/", http.HTTPStatus.PERMANENT_REDIRECT)

        # Get service from all registered experiments
        scheduler = Scheduler.instance()
        service_obj = None
        for xp in scheduler.experiments.values():
            service_obj = xp.services.get(service, None)
            if service_obj:
                break

        service = service_obj
        if service is None:
            return Response(f"Service {service} not found", http.HTTPStatus.NOT_FOUND)

        base_url = service.get_url()
        return proxy_response(base_url, request, path[0] if path else "/")

    @app.route("/notifications/<jobid>/progress")
    def notifications_progress(jobid):
        level = int(request.args.get("level", 0))
        progress = float(request.args.get("progress", 0.0))

        try:
            scheduler = Scheduler.instance()
            scheduler.jobs[jobid].set_progress(
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

        try:
            package_files = files("experimaestro.server")
            resource_file = package_files / datapath
            if resource_file.is_file():
                mimetype = MIMETYPES[datapath.rsplit(".", 1)[1]]
                content = resource_file.read_bytes()
                return Response(content, mimetype=mimetype)
        except (FileNotFoundError, KeyError):
            pass

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
    _instance: ClassVar[Optional["Server"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    def instance(settings: ServerSettings = None, state_provider=None) -> "Server":
        """Get or create the global server instance

        Args:
            settings: Server settings (optional)
            state_provider: WorkspaceStateProvider instance (required)
        """
        if Server._instance is None:
            with Server._lock:
                if Server._instance is None:
                    if settings is None:
                        from experimaestro.settings import get_settings

                        settings = get_settings().server

                    # State provider is required - it should be passed explicitly
                    if state_provider is None:
                        raise ValueError(
                            "state_provider parameter is required. "
                            "Get it via WorkspaceStateProvider.get_instance(workspace.path)"
                        )

                    Server._instance = Server(settings, state_provider)
        return Server._instance

    def __init__(self, settings: ServerSettings, state_provider):
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
        self.token = settings.token or uuid.uuid4().hex
        self.state_provider = state_provider
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
        """Start the websocket server in a daemon thread"""
        logging.info("Starting the web server")

        # Avoids clutering
        logging.getLogger("geventwebsocket.handler").setLevel(logging.WARNING)

        self.thread = threading.Thread(target=start_app, args=(self,), daemon=True)
        self.thread.start()

        # Wait until we really started
        while True:
            with self.cv_running:
                if self.running:
                    break
