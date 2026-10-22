"""Authentication routes for WebUI"""

import logging
from urllib.parse import urlencode
from fastapi import APIRouter, Request, Response, Query
from fastapi.responses import RedirectResponse

logger = logging.getLogger("xpm.webui.auth")

router = APIRouter()


@router.get("/auth")
async def auth(
    request: Request,
    xpm_token: str | None = Query(None, alias="xpm-token"),
    next_page: str | None = Query(None, alias="next"),
):
    """Authenticate with token and set cookie

    If valid token provided, redirects to the page ``next`` (by default, the
    home page) with cookie set. The token is passed as ?xpm-token=... in the
    URL.
    """
    server = request.app.state.server

    # Only local pages (not //host/... or /\host/...)
    if not next_page or not next_page.startswith("/") or next_page[1:2] in ("/", "\\"):
        next_page = "/"

    if xpm_token and server.token == xpm_token:
        response = RedirectResponse(url=next_page, status_code=302)
        response.set_cookie(
            key="experimaestro_token",
            value=xpm_token,
            httponly=True,
            samesite="lax",
        )
        return response

    return RedirectResponse(
        url=f"/login.html?{urlencode({'next': next_page})}", status_code=302
    )


@router.get("/stop")
async def stop(
    request: Request,
    xpm_token: str | None = Query(None, alias="xpm-token"),
):
    """Stop the server (requires authentication)"""
    server = request.app.state.server

    # Check token from query param or cookie
    token = xpm_token or request.cookies.get("experimaestro_token")

    if server.token == token:
        server.request_quit()
        return Response(status_code=202)

    return Response(status_code=401)
