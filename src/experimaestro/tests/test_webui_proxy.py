"""Tests for the service proxy of the web UI"""

import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.sync.server import serve

from experimaestro.webui.routes import proxy


@pytest.fixture
def echo_service():
    """A websocket service answering each message with its path and content"""

    def handler(connection):
        for message in connection:
            if isinstance(message, bytes):
                connection.send(message[::-1])
            else:
                connection.send(f"{connection.request.path}:{message}")

    with serve(handler, "localhost", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://localhost:{server.socket.getsockname()[1]}"
        server.shutdown()


@pytest.fixture
def client(echo_service, monkeypatch):
    monkeypatch.setattr(proxy, "_service_url", lambda *_: (echo_service, None))
    app = FastAPI()
    app.state.server = SimpleNamespace(token="secret")
    app.include_router(proxy.router)
    return TestClient(app)


def test_websocket_proxy(client):
    client.cookies.set("experimaestro_token", "secret")
    with client.websocket_connect("/services/xp/viewer/websocket?runId=1") as ws:
        ws.send_text("hello")
        assert ws.receive_text() == "/websocket?runId=1:hello"
        ws.send_bytes(b"abc")
        assert ws.receive_bytes() == b"cba"


def test_websocket_proxy_requires_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/services/xp/viewer/websocket"):
            pass
