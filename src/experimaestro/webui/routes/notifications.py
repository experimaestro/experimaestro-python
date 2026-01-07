"""Progress notification routes for WebUI"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, Response

logger = logging.getLogger("xpm.webui.notifications")

router = APIRouter()


@router.get("/notifications/{job_id}/progress")
async def job_progress(
    request: Request,
    job_id: str,
    level: int = 0,
    progress: float = 0.0,
    desc: Optional[str] = None,
):
    """Receive progress notification from a running job

    Jobs can call this endpoint to report their progress.
    This updates the job's progress state in the scheduler.

    Args:
        job_id: The job identifier
        level: Progress nesting level (0 = top level)
        progress: Progress value between 0.0 and 1.0
        desc: Optional progress description
    """
    server = request.app.state.server

    # Get scheduler (only works in active mode)
    from experimaestro.scheduler.base import Scheduler

    if not isinstance(server.state_provider, Scheduler):
        # In monitoring mode, we can't update progress
        # Just acknowledge the request
        return Response(status_code=200)

    scheduler: Scheduler = server.state_provider

    try:
        job = scheduler.jobs.get(job_id)
        if job:
            job.set_progress(level, progress, desc)
    except Exception as e:
        logger.debug("Progress update failed for job %s: %s", job_id, e)

    return Response(status_code=200)
