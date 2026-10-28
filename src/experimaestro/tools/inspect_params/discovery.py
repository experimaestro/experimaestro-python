# ruff: noqa: T201
"""File and workspace discovery utilities for Experimaestro parameters inspector.

Resolves local workspace paths, remote SSH workspaces (via `experimaestro.utils.remote`),
and locates params.json / experimaestro.json files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from experimaestro.utils.remote import (
    clean_job_target,
    find_remote_workspace_file,
    resolve_job_file_pattern,
)

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("experimaestro")


def resolve_workspace(
    workspace: str | None = None,
) -> Path | Any:
    """Resolve workspace directory from an explicit name/path or Experimaestro default.

    Resolution order:
    1. If workspace argument is provided:
       - If it points to an existing local directory, return it.
       - Otherwise, look up configured workspace settings via `find_workspace(workspace=...)`.
    2. If no workspace is provided:
       - Look up default workspace via `find_workspace()`.
       - Check `EXPERIMAESTRO_WORKSPACE` environment variable.
       - Search parent directories for `.experimaestro`.

    Args:
        workspace: Optional workspace name or filesystem path.

    Returns:
        Resolved Path to the local workspace directory, or remote WorkspaceSettings.

    Raises:
        FileNotFoundError: If the workspace cannot be determined or does not exist.
    """
    if workspace:
        p = Path(workspace).expanduser().resolve()
        if p.is_dir():
            logger.info("Using workspace %s", p)
            return p
        try:
            from experimaestro.settings import find_workspace

            ws_env = find_workspace(workspace=workspace)
            if ws_env:
                if ws_env.is_remote:
                    return ws_env
                if ws_env.path:
                    return Path(ws_env.path).resolve()
        except Exception:
            pass
        raise FileNotFoundError(
            f"Specified workspace '{workspace}' could not be found."
        )

    # Try default workspace from experimaestro settings
    try:
        from experimaestro.settings import find_workspace

        ws_env = find_workspace()
        if ws_env:
            if ws_env.is_remote:
                return ws_env
            if ws_env.path:
                return Path(ws_env.path).resolve()
    except Exception:
        pass

    # Fallback: check environment variable or walk up from current working directory
    env_ws = os.environ.get("EXPERIMAESTRO_WORKSPACE")
    if env_ws and Path(env_ws).is_dir():
        ws_path = Path(env_ws).resolve()
        logger.info("Using default workspace %s", ws_path)
        return ws_path

    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / ".experimaestro").is_dir():
            ws_path = p.resolve()
            logger.info("Using default workspace %s", ws_path)
            return ws_path

    raise FileNotFoundError(
        "No workspace specified and default workspace could not be determined."
    )


def find_local_workspace_file(
    ws_dir: Path, target: str, filename: str = "params.json"
) -> Path:
    """Find a specific file within a local workspace job directory.

    Args:
        ws_dir: Local workspace root directory.
        target: Target job identifier, prefix, or job path.
        filename: Relative filename or pattern to locate (default: 'params.json').

    Returns:
        Resolved Path to the matching file.

    Raises:
        FileNotFoundError: If target is empty or no matching file exists in workspace.
    """
    try:
        clean = clean_job_target(target, str(ws_dir), filename=filename)
    except ValueError as e:
        raise FileNotFoundError(
            f"Please specify a job ID or path. {filename} not found in current directory."
        ) from e

    pattern = resolve_job_file_pattern(clean, filename)
    matches = sorted(list(ws_dir.glob(pattern)))

    if not matches:
        raise FileNotFoundError(
            f"No job matching '{clean}' found in workspace {ws_dir} (pattern: {pattern})"
        )

    if len(matches) > 1:
        print(
            f"Warning: Multiple jobs ({len(matches)}) matched '{clean}'. "
            f"Using the first match: {matches[0].parent.name} ({matches[0].parent.parent.name})",
            file=sys.stderr,
        )

    return matches[0].resolve()


def find_workspace_file(
    ws_or_path: Any,
    target: str,
    filename: str = "params.json",
    cache_dir: Path | None = None,
) -> Path:
    """Find or fetch a file from either a local or remote Experimaestro workspace.

    Args:
        ws_or_path: WorkspaceSettings instance (remote/local) or Path to local workspace.
        target: Job ID prefix, locator, or job directory path.
        filename: Relative filename within the job directory (default: 'params.json').
        cache_dir: Optional local cache directory when fetching from a remote workspace.

    Returns:
        Resolved local Path to the found or synchronized file.
    """
    from experimaestro.settings import WorkspaceSettings

    if isinstance(ws_or_path, WorkspaceSettings) and ws_or_path.is_remote:
        return find_remote_workspace_file(
            ws_or_path, target, filename=filename, cache_dir=cache_dir
        )

    ws_dir = (
        ws_or_path.path
        if isinstance(ws_or_path, WorkspaceSettings)
        else Path(ws_or_path)
    )
    return find_local_workspace_file(ws_dir, target, filename=filename)


def find_job_params_file(
    provider: "StateProvider", target: str, ws_base: str = ""
) -> Path:
    """Find the parameters file of a job through a state provider.

    Used for remote workspaces: the job is looked up in the provider state,
    and its parameters file is fetched into the provider's local cache.

    Args:
        provider: State provider of the workspace.
        target: Job ID / hash prefix (``[task/]hash``) or job directory path.
        ws_base: Workspace path, stripped from the target if it starts with it.

    Returns:
        Local Path to the parameters file.

    Raises:
        FileNotFoundError: If no job matches, or if it has no parameters file.
    """
    try:
        clean = clean_job_target(target, ws_base, filename="params.json")
    except ValueError as e:
        raise FileNotFoundError("Please specify a job ID or path.") from e

    task_prefix, _, job_prefix = clean.replace(":", "/").rpartition("/")

    def matching(jobs):
        return sorted(
            (
                job
                for job in jobs
                if job.identifier.startswith(job_prefix)
                and job.task_id.startswith(task_prefix)
            ),
            key=lambda job: job.locator,
        )

    # Jobs of experiments first, then jobs that do not belong to any
    matches = matching(provider.get_all_jobs()) or matching(provider.get_orphan_jobs())
    if not matches:
        raise FileNotFoundError(f"No job matching '{clean}' found in workspace")

    if len(matches) > 1:
        print(
            f"Warning: Multiple jobs ({len(matches)}) matched '{clean}'. "
            f"Using the first match: {matches[0].locator}",
            file=sys.stderr,
        )

    if params_file := provider.get_job_params_file(matches[0]):
        return params_file
    raise FileNotFoundError(f"No parameters file found for job {matches[0].locator}")


def _find_params_in_path(p: Path) -> Path | None:
    """Check if a path points directly to, or contains, a parameters JSON file.

    Checks if `p` is a file directly, or if it is a directory containing
    `params.json` or `experimaestro.json` (or any descendant match).

    Args:
        p: Path to inspect.

    Returns:
        Resolved Path to the found file, or None if no match is found.
    """
    if p.is_file():
        return p.resolve()
    if p.is_dir():
        for name in ("params.json", "experimaestro.json", "objects.jsonl"):
            cand = p / name
            if cand.is_file():
                return cand.resolve()
        m = (
            list(p.glob("**/params.json"))
            + list(p.glob("**/experimaestro.json"))
            + list(p.glob("**/objects.jsonl"))
        )
        if m:
            return m[0].resolve()
    return None


def find_params_file(target: str | None = None, workspace: str | None = None) -> Path:
    """Find and resolve the path to a params.json or experimaestro.json file.

    Search resolution order:
    1. If an explicit workspace is specified:
       - If remote, query and synchronize via SSH.
       - If target is an existing local absolute path, return it directly.
       - Otherwise, search inside the specified local workspace.
    2. If no workspace is specified:
       - If target is an absolute or user-tilde path, resolve locally; do NOT
         attempt workspace discovery.
       - If target is an existing local relative path/directory, resolve it.
       - Otherwise, discover the default workspace (remote or local) and search
         for matching jobs.

    Args:
        target: Job ID / hash prefix, path to params.json, or job directory.
        workspace: Optional explicit workspace name or path.

    Returns:
        Resolved local Path to the parameters JSON file.

    Raises:
        FileNotFoundError: If the target cannot be found or matched.
    """
    raw = target or "."
    p = Path(raw).expanduser()

    # 1. If explicit workspace is specified, check it first (remote overrides local path)
    if workspace is not None:
        ws_or_path = resolve_workspace(workspace)
        from experimaestro.settings import WorkspaceSettings

        if isinstance(ws_or_path, WorkspaceSettings) and ws_or_path.is_remote:
            return find_workspace_file(ws_or_path, raw, filename="params.json")

        # Explicit local workspace: if target is an existing local absolute path, return it directly
        if p.is_absolute():
            resolved = _find_params_in_path(p)
            if resolved:
                return resolved

        return find_workspace_file(ws_or_path, raw, filename="params.json")

    # 2. No explicit workspace: check if absolute or tilde path exists locally (never load from workspace)
    if p.is_absolute():
        resolved = _find_params_in_path(p)
        if resolved:
            return resolved
        raise FileNotFoundError(f"params.json not found at absolute path: {raw}")

    # 3. Existing relative path locally
    resolved = _find_params_in_path(p)
    if resolved:
        return resolved

    # 4. Otherwise fall back to default workspace discovery
    ws_or_path = resolve_workspace()
    return find_workspace_file(ws_or_path, raw, filename="params.json")
