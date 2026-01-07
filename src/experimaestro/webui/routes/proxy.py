"""Service proxy routes for WebUI"""

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse

logger = logging.getLogger("xpm.webui.proxy")

router = APIRouter()


@router.api_route(
    "/services/{service_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_service(request: Request, service_id: str, path: str = ""):
    """Proxy requests to a service

    Routes /services/{service_id}/{path} to the service's URL.
    """
    server = request.app.state.server

    # Get service from scheduler (only works in active mode)
    from experimaestro.scheduler.base import Scheduler

    if not isinstance(server.state_provider, Scheduler):
        return Response(
            content="Service proxy only available in active mode",
            status_code=503,
        )

    scheduler: Scheduler = server.state_provider

    # Find service in experiments
    service = None
    for xp in scheduler.experiments.values():
        service = xp.services.get(service_id)
        if service:
            break

    if service is None:
        return Response(
            content=f"Service {service_id} not found",
            status_code=404,
        )

    # Get service URL
    base_url = service.get_url()
    if not base_url:
        return Response(
            content=f"Service {service_id} has no URL",
            status_code=503,
        )

    # Proxy the request using httpx
    import httpx

    # Build target URL
    target_url = f"{base_url}/{path}"
    if request.query_params:
        target_url = f"{target_url}?{request.query_params}"

    # Forward headers (filter sensitive ones)
    headers = {}
    for key, value in request.headers.items():
        key_lower = key.lower()
        if key_lower not in ("host", "content-length", "transfer-encoding"):
            headers[key] = value

    # Get request body for POST/PUT/PATCH
    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                follow_redirects=False,
            )

            # Build response
            response_headers = {}
            for key, value in response.headers.items():
                key_lower = key.lower()
                if key_lower not in (
                    "content-encoding",
                    "transfer-encoding",
                    "content-length",
                ):
                    response_headers[key] = value

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type"),
            )
    except httpx.RequestError as e:
        logger.error("Proxy error for service %s: %s", service_id, e)
        return Response(
            content=f"Proxy error: {e}",
            status_code=502,
        )


@router.get("/services/{service_id}")
async def redirect_service(service_id: str):
    """Redirect to service with trailing slash"""
    return RedirectResponse(
        url=f"/services/{service_id}/",
        status_code=308,  # Permanent redirect, preserves method
    )
