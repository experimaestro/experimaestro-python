"""Log streaming routes for the WebUI.

Exposes job and service stdout/stderr for the frontend log viewer. The frontend
polls with a byte ``offset`` and receives the new content plus the next offset,
or asks for the lines ``before`` what it shows, mirroring the TUI log viewer
(shared logic in ``logs/reader.py``).

The logs are read from the state provider's log source (for remote
workspaces, by blocks from the remote host): the routes are synchronous
functions, run by FastAPI in a thread pool, so that waiting for the network
does not block the server.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from experimaestro.scheduler.service_operations import find_service
from experimaestro.logs.reader import empty_log_slice, read_log_slice

logger = logging.getLogger("xpm.webui.logs")

router = APIRouter()

# Map the frontend stream name to the job log file extension.
_STREAM_EXT = {"stdout": "out", "stderr": "err", "out": "out", "err": "err"}


def _check_auth(request: Request) -> bool:
    server = request.app.state.server
    return request.cookies.get("experimaestro_token") == server.token


def _int_param(request: Request, name: str) -> int | None:
    raw = request.query_params.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_log(request: Request, path: Path, remote: bool = True) -> JSONResponse:
    """Read a slice of a log file (from the state provider's log source,
    unless the file is local)"""
    server = request.app.state.server
    source = server.state_provider.log_source(path) if remote else path
    try:
        data = read_log_slice(
            source,
            _int_param(request, "offset"),
            before=_int_param(request, "before"),
            generation=_int_param(request, "generation"),
        )
    except Exception as e:
        logger.warning("Cannot read the log %s: %s", path, e)
        return JSONResponse({"error": str(e)}, status_code=502)
    logger.debug(
        "Log %s: %d bytes, read from %d to %d",
        path,
        data["size"],
        data["start"],
        data["offset"],
    )
    return JSONResponse(data)


@router.get("/api/jobs/{task_id}/{job_id}/logs/{stream}")
def job_logs(request: Request, task_id: str, job_id: str, stream: str):
    """Return a slice of a job's stdout/stderr log."""
    if not _check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    ext = _STREAM_EXT.get(stream)
    if ext is None:
        return JSONResponse({"error": f"unknown stream {stream}"}, status_code=400)

    server = request.app.state.server
    state_provider = server.state_provider

    job = state_provider.get_job(task_id, job_id)
    if job is None or job.path is None:
        logger.warning("Log of %s/%s: job not found", task_id, job_id)
        return JSONResponse({"error": "job not found"}, status_code=404)

    scriptname = task_id.rsplit(".", 1)[-1]
    return _read_log(request, Path(job.path) / f"{scriptname}.{ext}")


@router.get("/api/logs/experimaestro")
def experimaestro_logs(request: Request):
    """Return a slice of the experimaestro scheduler log."""
    if not _check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    server = request.app.state.server
    state_provider = server.state_provider

    log_path = state_provider.get_scheduler_log_path()
    if log_path is None:
        return JSONResponse(empty_log_slice())

    return _read_log(request, Path(log_path))


@router.get("/api/services/{experiment_id}/{service_id}/logs/{stream}")
def service_logs(request: Request, experiment_id: str, service_id: str, stream: str):
    """Return a slice of a service's stdout/stderr log."""
    if not _check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    if stream not in ("stdout", "stderr", "out", "err"):
        return JSONResponse({"error": f"unknown stream {stream}"}, status_code=400)

    server = request.app.state.server
    state_provider = server.state_provider

    service = find_service(state_provider, experiment_id, service_id)
    if service is None:
        return JSONResponse({"error": "service not found"}, status_code=404)

    attr = "stdout" if stream in ("stdout", "out") else "stderr"
    log_path = getattr(service, attr, None)
    if log_path is None:
        return JSONResponse(empty_log_slice())

    # Read locally: the logs are those of the live service, which runs on
    # this machine (also when monitoring a remote workspace)
    return _read_log(request, Path(log_path), remote=False)
