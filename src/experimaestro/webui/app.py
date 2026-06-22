"""FastAPI application for WebUI

Creates the FastAPI app with WebSocket endpoint and routes.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, Response

from experimaestro.webui.websocket import WebSocketHandler
from experimaestro.webui.state_bridge import StateBridge
from experimaestro.webui.routes import auth, proxy, logs

if TYPE_CHECKING:
    from experimaestro.webui.server import WebUIServer

logger = logging.getLogger("xpm.webui")

# MIME types for static files
MIMETYPES = {
    "html": "text/html",
    "map": "text/plain",
    "txt": "text/plain",
    "ico": "image/x-icon",
    "png": "image/png",
    "svg": "image/svg+xml",
    "css": "text/css",
    "js": "application/javascript",
    "json": "application/json",
    "eot": "font/vnd.ms-fontobject",
    "woff": "font/woff",
    "woff2": "font/woff2",
    "ttf": "font/ttf",
}


def create_app(server: "WebUIServer") -> FastAPI:
    """Create FastAPI application

    Args:
        server: WebUIServer instance with state_provider and settings

    Returns:
        Configured FastAPI app
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The state provider notifies listeners from the scheduler thread, but
        # the WebSocket connections (and their asyncio locks) live on the loop
        # uvicorn serves on. Capture that loop here so the bridge can hand
        # broadcasts to it via run_coroutine_threadsafe.
        app.state.state_bridge.set_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(
        title="Experimaestro WebUI",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Create WebSocket handler
    ws_handler = WebSocketHandler(server.state_provider, server.token)

    # Create state bridge to forward state events to WebSocket
    state_bridge = StateBridge(server.state_provider, ws_handler)

    # Store references on app for routes to access
    app.state.server = server
    app.state.ws_handler = ws_handler
    app.state.state_bridge = state_bridge

    # Include route modules
    app.include_router(auth.router)
    app.include_router(proxy.router)
    app.include_router(logs.router)

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for real-time updates"""
        await ws_handler.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                await ws_handler.handle_message(websocket, data)
        except WebSocketDisconnect:
            await ws_handler.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            await ws_handler.disconnect(websocket)

    # Root route
    @app.get("/")
    async def root(request: Request):
        """Root redirect based on auth status"""
        # Check cookie for authentication
        token = request.cookies.get("experimaestro_token")
        if token == server.token:
            return RedirectResponse(url="/index.html", status_code=302)
        return RedirectResponse(url="/login.html", status_code=302)

    # Static file serving (catch-all, must be last)
    @app.get("/{path:path}")
    async def static_files(request: Request, path: str):
        """Serve static files from data/ directory"""
        # Check authentication for index.html
        if path == "index.html":
            token = request.cookies.get("experimaestro_token")
            if token != server.token:
                return RedirectResponse(url="/login.html", status_code=302)

        # Get static file
        datapath = f"data/{path}"
        logger.debug("Looking for %s", datapath)

        try:
            package_files = files("experimaestro.webui")
            resource_file = package_files / datapath
            if resource_file.is_file():
                ext = datapath.rsplit(".", 1)[-1]
                mimetype = MIMETYPES.get(ext, "application/octet-stream")
                content = resource_file.read_bytes()
                # The entrypoint assets keep a fixed name (index.html/js/css) and
                # are overwritten on every rebuild, so they must never be served
                # from a stale browser cache. Hashed assets (fonts/images) are
                # safe to cache for a long time.
                base = path.rsplit("/", 1)[-1]
                if base in (
                    "index.html",
                    "index.js",
                    "index.css",
                    "login.html",
                    "icon.svg",
                    "favicon.svg",
                ):
                    cache_control = "no-cache, no-store, must-revalidate"
                else:
                    cache_control = "public, max-age=31536000, immutable"
                return Response(
                    content=content,
                    media_type=mimetype,
                    headers={"Cache-Control": cache_control},
                )
        except (FileNotFoundError, KeyError):
            pass

        return Response(content="Page not found", status_code=404)

    return app
