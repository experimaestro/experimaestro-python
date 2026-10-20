"""Service proxy routes for WebUI"""

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse

from experimaestro.scheduler.service_operations import find_service

logger = logging.getLogger("xpm.webui.proxy")

router = APIRouter()


@router.api_route(
    "/services/{experiment_id}/{service_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_service(
    request: Request, experiment_id: str, service_id: str, path: str = ""
):
    """Proxy requests to a service

    Routes /services/{experiment_id}/{service_id}/{path} to the service's URL.
    """
    server = request.app.state.server
    service = find_service(server.state_provider, experiment_id, service_id)
    if service is None:
        return Response(
            content=f"Service {service_id} not found",
            status_code=404,
        )

    # Only proxy started services (get_url() would start a stopped one)
    live = service.to_service()
    if live.state.name != "RUNNING":
        return Response(
            content=f"Service {service_id} is not running",
            status_code=503,
        )
    base_url = live.get_url()
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


@router.get("/services/{experiment_id}/{service_id}")
async def redirect_service(experiment_id: str, service_id: str):
    """Redirect to service with trailing slash"""
    return RedirectResponse(
        url=f"/services/{experiment_id}/{service_id}/",
        status_code=308,  # Permanent redirect, preserves method
    )
