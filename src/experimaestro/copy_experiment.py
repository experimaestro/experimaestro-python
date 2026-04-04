"""Copy experiments between workspaces (local or remote via SSH).

This module provides the core logic for copying experiment run directories
and their associated job directories between workspaces. It supports
local-to-local and local-to-remote (via SSH+rsync) transfers.

Usage::

    from experimaestro.copy_experiment import copy_experiment, CopyResult
    from experimaestro.settings import get_workspace

    src = get_workspace("cluster", include_remote=True)
    dst = get_workspace("local")
    result = copy_experiment(src, dst, "my_experiment", "20260121_105710")
"""

import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from experimaestro.scheduler.interfaces import ExperimentJobInformation
from experimaestro.settings import WorkspaceSettings

logger = logging.getLogger("xpm.copy")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CopyResult:
    """Result of a copy_experiment operation."""

    jobs_copied: int = 0
    jobs_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level helpers: SSH, rsync, workspace introspection
# ---------------------------------------------------------------------------


def run_ssh_command(
    ws: WorkspaceSettings, remote_cmd: str
) -> subprocess.CompletedProcess:
    """Run a command on a remote workspace via SSH.

    The *remote_cmd* is passed as a single string so the remote shell
    handles tilde expansion and other shell features.
    """
    cmd = ["ssh"]
    if ws.ssh.options:
        cmd.extend(ws.ssh.options)
    cmd.extend([ws.ssh.host, remote_cmd])
    return subprocess.run(cmd, capture_output=True, text=True)


def rsync_path(ws: WorkspaceSettings, relpath: str) -> str:
    """Build an rsync-compatible path for a workspace.

    Local: ``/abs/path/relpath``
    Remote: ``user@host:/remote/path/relpath``
    """
    if ws.is_remote:
        base = ws._raw_path.rstrip("/")
        return f"{ws.ssh.host}:{base}/{relpath}"
    else:
        return str(ws.path / relpath)


def ssh_args(ws: WorkspaceSettings) -> list[str]:
    """Return ``-e "ssh <options>"`` for rsync when the workspace is remote."""
    if ws.is_remote and ws.ssh.options:
        return ["-e", "ssh " + " ".join(ws.ssh.options)]
    return []


def run_rsync(
    src_ws: WorkspaceSettings,
    dst_ws: WorkspaceSettings,
    src_rel: str,
    dst_rel: str,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Execute an rsync command between workspaces.

    Adds compression (``-z``) when either side is remote.
    Always uses ``-a`` (archive mode) and ``--progress``.
    """
    if src_ws.is_remote and dst_ws.is_remote:
        raise RuntimeError(
            "Remote-to-remote copy is not supported. "
            "Run this command from one of the two hosts instead."
        )

    src_path = rsync_path(src_ws, src_rel)
    dst_path = rsync_path(dst_ws, dst_rel)

    # Ensure trailing slash on source for directory sync
    if not src_path.endswith("/"):
        src_path += "/"

    cmd = ["rsync", "-aH", "--mkpath", "--progress"]

    # Add compression for remote transfers
    if src_ws.is_remote or dst_ws.is_remote:
        cmd.append("-z")

    # Add SSH options from the remote side
    remote_ws = src_ws if src_ws.is_remote else dst_ws
    ssh_extra = ssh_args(remote_ws)
    if ssh_extra:
        cmd.extend(ssh_extra)

    if extra_args:
        cmd.extend(extra_args)

    if dry_run:
        cmd.append("--dry-run")

    cmd.extend([src_path, dst_path])

    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Workspace introspection (local or remote)
# ---------------------------------------------------------------------------


def read_jobs_jsonl(
    ws: WorkspaceSettings,
    experiment_id: str,
    run_id: str,
) -> list[ExperimentJobInformation]:
    """Read ``jobs.jsonl`` from a workspace (local or remote)."""
    relpath = f"experiments/{experiment_id}/{run_id}/jobs.jsonl"

    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{relpath}"
        result = run_ssh_command(ws, f"cat {remote_path}")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Failed to read {remote_path}")
        content = result.stdout
    else:
        local_path = ws.path / relpath
        content = local_path.read_text()

    jobs = []
    for line in content.strip().splitlines():
        line = line.strip()
        if line:
            jobs.append(ExperimentJobInformation.from_dict(json.loads(line)))
    return jobs


def path_exists(ws: WorkspaceSettings, relpath: str) -> bool:
    """Check if a directory exists in a workspace (local or remote)."""
    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{relpath}"
        result = run_ssh_command(ws, f"test -d {remote_path}")
        return result.returncode == 0
    else:
        return (ws.path / relpath).is_dir()


def resolve_current_run(ws: WorkspaceSettings, experiment_id: str) -> str | None:
    """Resolve the ``current`` symlink for an experiment.

    Returns the run_id string, or ``None`` if no current symlink exists.
    """
    relpath = f"experiments/{experiment_id}/current"

    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{relpath}"
        result = run_ssh_command(ws, f"readlink -f {remote_path}")
        if result.returncode != 0:
            logger.debug(
                "readlink -f failed for %s (rc=%d): %s",
                remote_path,
                result.returncode,
                result.stderr.strip(),
            )
            return None
        target = result.stdout.strip()
        if not target:
            return None
        return Path(target).name
    else:
        local_path = ws.path / relpath
        if not local_path.is_symlink():
            return None
        return local_path.resolve().name


def format_run_id(run_id: str) -> str:
    """Format a run ID like ``20260118_170744`` into ``2026-01-18 17:07:44``.

    Returns *run_id* unchanged if it doesn't match the expected format.
    """
    if len(run_id) == 15 and run_id[8] == "_":
        try:
            return (
                f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]} "
                f"{run_id[9:11]}:{run_id[11:13]}:{run_id[13:15]}"
            )
        except (IndexError, ValueError):
            pass
    return run_id


def list_experiments(ws: WorkspaceSettings) -> list[str]:
    """List experiment IDs in a workspace."""
    relpath = "experiments"

    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{relpath}"
        result = run_ssh_command(ws, f"ls {remote_path}")
        if result.returncode != 0:
            return []
        return [
            name.strip() for name in result.stdout.strip().splitlines() if name.strip()
        ]
    else:
        exp_dir = ws.path / relpath
        if not exp_dir.is_dir():
            return []
        return sorted(d.name for d in exp_dir.iterdir() if d.is_dir())


def list_runs(ws: WorkspaceSettings, experiment_id: str) -> list[str]:
    """List run IDs for an experiment in a workspace."""
    relpath = f"experiments/{experiment_id}"

    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{relpath}"
        result = run_ssh_command(ws, f"ls {remote_path}")
        if result.returncode != 0:
            return []
        return [
            name.strip()
            for name in result.stdout.strip().splitlines()
            if name.strip() and name.strip() != "current"
        ]
    else:
        run_dir = ws.path / relpath
        if not run_dir.is_dir():
            return []
        return sorted(
            d.name
            for d in run_dir.iterdir()
            if d.is_dir() and d.name != "current" and not d.is_symlink()
        )


# ---------------------------------------------------------------------------
# Path rewriting for params.json
# ---------------------------------------------------------------------------


def _rewrite_path(value: str, src_prefix: str, dst_prefix: str) -> str:
    """Replace *src_prefix* with *dst_prefix* at the start of a path string."""
    if value == src_prefix:
        return dst_prefix
    if value.startswith(src_prefix + "/"):
        return dst_prefix + value[len(src_prefix) :]
    return value


def _rewrite_json_paths(obj, src_prefix: str, dst_prefix: str):
    """Recursively walk a JSON structure and rewrite path values.

    Rewrites ``{"type": "path", "value": "..."}`` and
    ``{"type": "path.serialized", "value": "..."}`` entries.
    """
    if isinstance(obj, dict):
        if obj.get("type") in ("path", "path.serialized") and "value" in obj:
            obj["value"] = _rewrite_path(obj["value"], src_prefix, dst_prefix)
        else:
            for v in obj.values():
                _rewrite_json_paths(v, src_prefix, dst_prefix)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_json_paths(item, src_prefix, dst_prefix)


def _rewrite_params_json(
    ws: WorkspaceSettings,
    relpath: str,
    src_workspace_path: str,
    dst_workspace_path: str,
) -> None:
    """Rewrite workspace-dependent paths in a job's ``params.json``.

    Updates the top-level ``"workspace"`` field and all
    ``path`` / ``path.serialized`` values within objects.
    """
    params_rel = f"{relpath}/params.json"

    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{params_rel}"
        result = run_ssh_command(ws, f"cat {remote_path}")
        if result.returncode != 0:
            raise RuntimeError(
                f"Cannot read remote {remote_path}: {result.stderr.strip()}"
            )
        params = json.loads(result.stdout)

        params["workspace"] = _rewrite_path(
            params["workspace"], src_workspace_path, dst_workspace_path
        )
        _rewrite_json_paths(
            params.get("objects", []), src_workspace_path, dst_workspace_path
        )

        new_content = json.dumps(params)
        write_result = run_ssh_command(
            ws, f"printf '%s' '{new_content}' > {remote_path}"
        )
        if write_result.returncode != 0:
            raise RuntimeError(
                f"Failed to write remote {remote_path}: {write_result.stderr.strip()}"
            )
    else:
        params_path = ws.path / params_rel
        if not params_path.is_file():
            return
        params = json.loads(params_path.read_text())

        params["workspace"] = _rewrite_path(
            params["workspace"], src_workspace_path, dst_workspace_path
        )
        _rewrite_json_paths(
            params.get("objects", []), src_workspace_path, dst_workspace_path
        )

        params_path.write_text(json.dumps(params))


# ---------------------------------------------------------------------------
# Workspace path resolution
# ---------------------------------------------------------------------------


def resolve_workspace_path(ws: WorkspaceSettings) -> str:
    """Get the absolute workspace path (resolving ``~`` for remote)."""
    if ws.is_remote:
        result = run_ssh_command(ws, f"realpath {ws._raw_path}")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        raw = ws._raw_path
        if raw.startswith("~/"):
            home_result = run_ssh_command(ws, "echo $HOME")
            if home_result.returncode == 0:
                return home_result.stdout.strip() + raw[1:]
        return raw
    else:
        return str(ws.path.resolve())


# ---------------------------------------------------------------------------
# Staging helpers (atomic copy via temp directory)
# ---------------------------------------------------------------------------


def _make_staging_dir(ws: WorkspaceSettings) -> str:
    """Create a temporary staging directory inside the destination workspace.

    Returns the relative path (from workspace root) of the staging directory.
    A ``.lock`` file marks the copy as in-progress.
    """
    staging_rel = f".xpm-staging/{uuid.uuid4().hex[:12]}"

    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{staging_rel}"
        result = run_ssh_command(
            ws, f"mkdir -p {remote_path} && touch {remote_path}/.lock"
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create staging dir: {result.stderr.strip()}")
    else:
        staging_path = ws.path / staging_rel
        staging_path.mkdir(parents=True)
        (staging_path / ".lock").touch()

    return staging_rel


def _move_staged(ws: WorkspaceSettings, staging_rel: str, src_rel: str, dst_rel: str):
    """Move a directory from staging to its final location."""
    staged_path_rel = f"{staging_rel}/{src_rel}"

    if ws.is_remote:
        remote_staged = f"{ws._raw_path.rstrip('/')}/{staged_path_rel}"
        remote_final = f"{ws._raw_path.rstrip('/')}/{dst_rel}"
        result = run_ssh_command(
            ws,
            f"mkdir -p $(dirname {remote_final}) && mv {remote_staged} {remote_final}",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to move staged dir: {result.stderr.strip()}")
    else:
        staged = ws.path / staged_path_rel
        final = ws.path / dst_rel
        final.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(final)


def _cleanup_staging(ws: WorkspaceSettings, staging_rel: str):
    """Remove the staging directory after a successful copy."""
    if ws.is_remote:
        remote_path = f"{ws._raw_path.rstrip('/')}/{staging_rel}"
        run_ssh_command(ws, f"rm -rf {remote_path}")
    else:
        staging_path = ws.path / staging_rel
        if staging_path.exists():
            shutil.rmtree(staging_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def copy_experiment(
    src_workspace: WorkspaceSettings,
    dst_workspace: WorkspaceSettings,
    experiment_id: str,
    run_id: str,
    dry_run: bool = False,
) -> CopyResult:
    """Copy an experiment run and its jobs from source to destination workspace.

    Each job directory is first rsynced into a staging area inside the
    destination workspace, then atomically moved to its final location.
    Workspace-dependent paths in ``params.json`` are rewritten after copy.

    Args:
        src_workspace: Source workspace settings.
        dst_workspace: Destination workspace settings.
        experiment_id: The experiment identifier.
        run_id: The resolved run ID (not ``"current"``).
        dry_run: If ``True``, show what would be copied without actually copying.

    Returns:
        A :class:`CopyResult` with counts of copied/skipped jobs and any errors.
    """
    result = CopyResult()

    # 0. Resolve absolute workspace paths for rewriting params.json
    src_ws_path = resolve_workspace_path(src_workspace)
    dst_ws_path = resolve_workspace_path(dst_workspace)
    logger.info("Source workspace path: %s", src_ws_path)
    logger.info("Destination workspace path: %s", dst_ws_path)

    # 1. Read jobs.jsonl to find all job directories
    try:
        jobs = read_jobs_jsonl(src_workspace, experiment_id, run_id)
    except Exception as e:
        result.errors.append(f"Failed to read jobs.jsonl: {e}")
        return result

    # 2. Compute job relative paths
    job_relpaths = [f"jobs/{job.task_id}/{job.job_id}" for job in jobs]

    # 3. Check which jobs already exist at destination
    existing_jobs: set[str] = set()
    for relpath in job_relpaths:
        try:
            if path_exists(dst_workspace, relpath):
                existing_jobs.add(relpath)
        except Exception as e:
            logger.warning("Failed to check existence of %s: %s", relpath, e)

    # 4. Create staging directory for atomic copies
    staging_rel = None
    if not dry_run:
        try:
            staging_rel = _make_staging_dir(dst_workspace)
            logger.info("Created staging directory: %s", staging_rel)
        except Exception as e:
            result.errors.append(f"Failed to create staging directory: {e}")
            return result

    try:
        # 5. Rsync experiment run directory into staging (exclude jobs symlinks)
        exp_run_rel = f"experiments/{experiment_id}/{run_id}"
        staging_exp_rel = f"{staging_rel}/{exp_run_rel}" if staging_rel else exp_run_rel
        try:
            run_rsync(
                src_workspace,
                dst_workspace,
                exp_run_rel,
                staging_exp_rel if not dry_run else exp_run_rel,
                extra_args=["--exclude=jobs"],
                dry_run=dry_run,
            )
            logger.info("Copied experiment run directory: %s", exp_run_rel)
        except subprocess.CalledProcessError as e:
            result.errors.append(f"Failed to copy experiment run dir: {e.stderr}")
            return result

        # 6. Rsync each missing job directory into staging, rewrite paths, move
        for relpath in job_relpaths:
            if relpath in existing_jobs:
                result.jobs_skipped += 1
                logger.info("Skipping existing job: %s", relpath)
                continue

            staging_job_rel = f"{staging_rel}/{relpath}" if staging_rel else relpath
            try:
                run_rsync(
                    src_workspace,
                    dst_workspace,
                    relpath,
                    staging_job_rel if not dry_run else relpath,
                    dry_run=dry_run,
                )
                result.jobs_copied += 1
                logger.info("Copied job: %s", relpath)

                # Rewrite workspace paths in params.json (in staging)
                if not dry_run and src_ws_path != dst_ws_path:
                    try:
                        _rewrite_params_json(
                            dst_workspace,
                            staging_job_rel,
                            src_ws_path,
                            dst_ws_path,
                        )
                    except Exception as e:
                        result.errors.append(
                            f"Failed to rewrite params.json for {relpath}: {e}"
                        )

                # Move from staging to final location
                if not dry_run:
                    _move_staged(dst_workspace, staging_rel, relpath, relpath)

            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to copy job {relpath}: {e.stderr}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        # Move experiment run dir from staging to final location
        if not dry_run:
            _move_staged(dst_workspace, staging_rel, exp_run_rel, exp_run_rel)

    finally:
        # Clean up staging directory
        if staging_rel and not dry_run:
            try:
                _cleanup_staging(dst_workspace, staging_rel)
            except Exception as e:
                logger.warning("Failed to clean up staging dir: %s", e)

    return result
