"""Service proxy routes for WebUI"""

import asyncio
import contextlib
import logging

from fastapi import APIRouter, Request, Response, WebSocket
from fastapi.responses import RedirectResponse

from experimaestro.scheduler.service_operations import find_service

logger = logging.getLogger("xpm.webui.proxy")

router = APIRouter()


def _service_url(server, experiment_id: str, service_id: str, run_id):
    """Returns ``(url, None)``, or ``(None, (status, message))``"""
    service = find_service(
        server.state_provider, experiment_id, service_id, run_id=run_id
    )
    if service is None:
        return None, (404, f"Service {service_id} not found")

    # Only proxy started services (get_url() would start a stopped one)
    live = service.to_service()
    if live.state.name != "RUNNING":
        return None, (503, f"Service {service_id} is not running")
    base_url = live.get_url()
    if not base_url:
        return None, (503, f"Service {service_id} has no URL")
    # The path is appended as "/{path}"
    return base_url.rstrip("/"), None


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
    base_url, error = _service_url(
        request.app.state.server,
        experiment_id,
        service_id,
        request.query_params.get("runId"),
    )
    if base_url is None:
        status, message = error
        return Response(content=message, status_code=status)

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


@router.websocket("/services/{experiment_id}/{service_id}/{path:path}")
async def proxy_service_websocket(
    websocket: WebSocket, experiment_id: str, service_id: str, path: str = ""
):
    """Proxy a websocket to a service (e.g. a viewer streaming its data)"""
    import websockets
    from websockets.exceptions import ConnectionClosed

    server = websocket.app.state.server
    if websocket.cookies.get("experimaestro_token") != server.token:
        await websocket.close(code=1008, reason="Invalid token")
        return

    base_url, error = _service_url(
        server, experiment_id, service_id, websocket.query_params.get("runId")
    )
    if base_url is None:
        await websocket.close(code=1011, reason=error[1])
        return

    target_url = "ws" + base_url.removeprefix("http") + f"/{path}"
    if websocket.query_params:
        target_url = f"{target_url}?{websocket.query_params}"

    try:
        upstream = await websockets.connect(target_url, max_size=None, open_timeout=10)
    except Exception as e:
        logger.error("Proxy error for service %s: %s", service_id, e)
        await websocket.close(code=1011, reason="Service unreachable")
        return

    await websocket.accept()

    async def client_to_service():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await upstream.send(message["bytes"])
            elif message.get("text") is not None:
                await upstream.send(message["text"])

    async def service_to_client():
        with contextlib.suppress(ConnectionClosed):
            async for message in upstream:
                if isinstance(message, bytes):
                    await websocket.send_bytes(message)
                else:
                    await websocket.send_text(message)

    async with upstream:
        tasks = [
            asyncio.create_task(client_to_service()),
            asyncio.create_task(service_to_client()),
        ]
        try:
            # Stop as soon as one side closes
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    with contextlib.suppress(Exception):
        await websocket.close()
