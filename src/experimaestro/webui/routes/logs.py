"""Log streaming routes for the WebUI.

Exposes job and service stdout/stderr for the frontend log viewer. The frontend
polls with a byte ``offset`` and receives the new content plus the next offset,
mirroring the TUI log viewer (shared logic in ``scheduler/logs.py``).
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from experimaestro.scheduler.logs import read_log_slice

logger = logging.getLogger("xpm.webui.logs")

router = APIRouter()

# Map the frontend stream name to the job log file extension.
_STREAM_EXT = {"stdout": "out", "stderr": "err", "out": "out", "err": "err"}


def _check_auth(request: Request) -> bool:
    server = request.app.state.server
    return request.cookies.get("experimaestro_token") == server.token


def _parse_offset(request: Request) -> Optional[int]:
    raw = request.query_params.get("offset")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _maybe_sync(state_provider, path: Path, include: list[str]) -> Path:
    """For remote providers, sync the file locally before reading."""
    if getattr(state_provider, "is_remote", False):
        try:
            local = state_provider.sync_path(str(path), include=include)
            if local is not None:
                return Path(local)
        except TypeError:
            # Base signature without ``include``
            local = state_provider.sync_path(str(path))
            if local is not None:
                return Path(local)
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Failed to sync %s: %s", path, e)
    return path


@router.get("/api/jobs/{task_id}/{job_id}/logs/{stream}")
async def job_logs(request: Request, task_id: str, job_id: str, stream: str):
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
        return JSONResponse({"error": "job not found"}, status_code=404)

    scriptname = task_id.rsplit(".", 1)[-1]
    job_dir = _maybe_sync(
        state_provider,
        Path(job.path),
        include=[f"{scriptname}.out", f"{scriptname}.err"],
    )
    log_path = job_dir / f"{scriptname}.{ext}"

    return JSONResponse(read_log_slice(log_path, _parse_offset(request)))


@router.get("/api/services/{service_id}/logs/{stream}")
async def service_logs(request: Request, service_id: str, stream: str):
    """Return a slice of a service's stdout/stderr log."""
    if not _check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    if stream not in ("stdout", "stderr", "out", "err"):
        return JSONResponse({"error": f"unknown stream {stream}"}, status_code=400)

    server = request.app.state.server
    state_provider = server.state_provider

    # Locate the service across experiments/runs.
    service = None
    for svc in state_provider.get_services():
        if svc.id == service_id:
            service = svc
            break

    if service is None:
        return JSONResponse({"error": "service not found"}, status_code=404)

    attr = "stdout" if stream in ("stdout", "out") else "stderr"
    log_path = getattr(service, attr, None)
    if log_path is None:
        return JSONResponse({"content": "", "offset": 0, "size": 0})

    log_path = _maybe_sync(
        state_provider, Path(log_path), include=[Path(log_path).name]
    )
    return JSONResponse(read_log_slice(log_path, _parse_offset(request)))
