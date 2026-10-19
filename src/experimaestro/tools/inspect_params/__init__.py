"""Experimaestro Parameters Inspector tool.

Inspects params.json and experimaestro.json files with a rich, collapsible
configuration tree, cycle detection, path resolution, and Python code synthesis.
"""

from __future__ import annotations

from typing import Optional

from experimaestro.tools.inspect_params.app import (
    NodeData,
    NodeDetailPane,
    NodeKind,
    ParamsInspectorApp,
    ParamsTree,
)
from experimaestro.tools.inspect_params.discovery import (
    find_local_workspace_file,
    find_params_file,
    find_workspace_file,
    resolve_workspace,
)
from experimaestro.tools.inspect_params.models import (
    ObjectNode,
    ParamsGraph,
    ResolvedPath,
)
from experimaestro.utils.remote import (
    clean_job_target,
    fetch_remote_file_rsync,
    fetch_remote_workspace_file,
    find_remote_workspace_file,
    get_session_cache_dir,
    resolve_job_file_pattern,
    run_ssh_command,
)

__all__ = [
    "NodeData",
    "NodeDetailPane",
    "NodeKind",
    "ObjectNode",
    "ParamsGraph",
    "ParamsInspectorApp",
    "ParamsTree",
    "ResolvedPath",
    "clean_job_target",
    "fetch_remote_file_rsync",
    "fetch_remote_workspace_file",
    "find_local_workspace_file",
    "find_params_file",
    "find_remote_workspace_file",
    "find_workspace_file",
    "get_session_cache_dir",
    "resolve_job_file_pattern",
    "resolve_workspace",
    "run_inspector",
    "run_ssh_command",
]


def run_inspector(
    target: Optional[str] = None, workspace: Optional[str] = None
) -> None:
    """Find parameters file and launch the interactive Textual inspector app."""
    params_path = find_params_file(target, workspace)
    app = ParamsInspectorApp(params_path)
    app.run()
