"""Authentication routes for WebUI"""

import logging
from typing import Optional
from fastapi import APIRouter, Request, Response, Query
from fastapi.responses import RedirectResponse

logger = logging.getLogger("xpm.webui.auth")

router = APIRouter()


@router.get("/auth")
async def auth(
    request: Request,
    xpm_token: Optional[str] = Query(None, alias="xpm-token"),
):
    """Authenticate with token and set cookie

    If valid token provided, redirects to index.html with cookie set.
    The token is passed as ?xpm-token=... in the URL.
    """
    server = request.app.state.server

    if xpm_token and server.token == xpm_token:
        response = RedirectResponse(url="/index.html", status_code=302)
        response.set_cookie(
            key="experimaestro_token",
            value=xpm_token,
            httponly=True,
            samesite="lax",
        )
        return response

    return RedirectResponse(url="/login.html", status_code=302)


@router.get("/stop")
async def stop(
    request: Request,
    xpm_token: Optional[str] = Query(None, alias="xpm-token"),
):
    """Stop the server (requires authentication)"""
    server = request.app.state.server

    # Check token from query param or cookie
    token = xpm_token or request.cookies.get("experimaestro_token")

    if server.token == token:
        server.request_quit()
        return Response(status_code=202)

    return Response(status_code=401)
