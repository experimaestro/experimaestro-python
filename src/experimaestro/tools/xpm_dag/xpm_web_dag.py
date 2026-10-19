"""Interactive web DAG visualization generator and server for Experimaestro.

This module generates a standalone, responsive, interactive HTML5 web page
displaying the experiment DAG with client-side Dagre layout, D3 zoom/pan,
custom card nodes, status coloring, Left-to-Right / Top-to-Bottom flow toggles,
real-time single-config trial filtering, lineage path highlighting, and an
integrated side inspector drawer.
"""

from __future__ import annotations

import http.server
from importlib.resources import files
import json
from pathlib import Path
import socketserver
import threading
from typing import Any
import webbrowser


def _load_html_template() -> str:
    """Load the standalone interactive HTML template."""
    try:
        return (
            files("experimaestro.tools.xpm_dag")
            .joinpath("dag_template.html")
            .read_text(encoding="utf-8")
        )
    except (TypeError, ModuleNotFoundError, FileNotFoundError):
        return (Path(__file__).parent / "dag_template.html").read_text(encoding="utf-8")


HTML_TEMPLATE = _load_html_template()


def generate_dag_html(
    dag_json: dict[str, Any],
    title: str = "",
    default_layout: str = "dagre",
) -> str:
    """Generate a self-contained interactive HTML web page for the experiment DAG."""
    exp_id = dag_json.get("experiment_id") or "Experiment"
    run_id = dag_json.get("run_id") or ""
    page_title = title or f"Experimaestro DAG - {exp_id}"

    html = HTML_TEMPLATE.replace("__TITLE__", page_title)
    html = html.replace("__RUN_ID__", run_id if run_id else "latest")
    html = html.replace("__DEFAULT_LAYOUT__", default_layout)
    html = html.replace("__GRAPH_DATA__", json.dumps(dag_json))
    return html


def serve_dag_web(
    dag_json: dict[str, Any],
    port: int = 0,
    open_browser: bool = True,
    default_layout: str = "dagre",
) -> None:
    """Start an ephemeral HTTP server to view the interactive DAG and open the browser."""
    html_content = generate_dag_html(dag_json, default_layout=default_layout).encode(
        "utf-8"
    )

    class DAGRequestHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content)

        def log_message(self, format, *args):  # noqa: A002
            # Suppress normal HTTP request logging
            pass

    # Use ThreadingTCPServer with port 0 for automatic port selection
    class ThreadedServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = ThreadedServer(("127.0.0.1", port), DAGRequestHandler)
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}"

    print(f"Starting interactive DAG web server on {url}")  # noqa: T201
    print("Press Ctrl+C to stop the server.")  # noqa: T201

    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")  # noqa: T201
    finally:
        server.server_close()
