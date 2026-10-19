"""Remote workspace utilities via SSH and Rsync.

Provides lightweight remote querying, pattern matching, and file synchronization
to support tools like `experimaestro inspect` and `experimaestro dag` on remote SSH workspaces.
"""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("experimaestro")

_SESSION_REMOTE_CACHE_DIR: Optional[Path] = None


def get_session_cache_dir() -> Path:
    """Return a managed temporary directory for cached remote workspace files.

    Creates the directory lazily on first access and registers automatic cleanup
    upon process termination via atexit.

    Returns:
        Path to the session temporary cache directory.
    """
    global _SESSION_REMOTE_CACHE_DIR
    if _SESSION_REMOTE_CACHE_DIR is None or not _SESSION_REMOTE_CACHE_DIR.exists():
        tmp = tempfile.mkdtemp(prefix="xpm_cache_")
        _SESSION_REMOTE_CACHE_DIR = Path(tmp)
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    return _SESSION_REMOTE_CACHE_DIR


def clean_job_target(
    raw: str, ws_base: str = "", filename: Optional[str] = None
) -> str:
    """Normalize a target job locator, prefix, or path.

    Strips surrounding slashes, explicit file names, workspace root prefixes,
    and leading 'jobs/' prefixes.

    Args:
        raw: Raw input target string.
        ws_base: Optional workspace base path to strip.
        filename: Optional filename to strip from the end (e.g. 'params.json').

    Returns:
        Normalized target string (e.g. 'ee76fb8f' or 'my_task/ee76fb8f').

    Raises:
        ValueError: If the normalized target is empty or root.
    """
    clean = raw.strip("/")
    if filename:
        clean = clean.removesuffix(f"/{filename}").removesuffix(filename).strip("/")
    else:
        for ext in (".json", ".log", ".txt", ".out", ".err"):
            if clean.endswith(ext):
                clean = str(Path(clean).parent)
                break

    ws_clean = ws_base.strip("/")
    if ws_clean and clean.startswith(ws_clean):
        clean = clean[len(ws_clean) :].strip("/")

    if "/jobs/" in clean:
        clean = clean.split("/jobs/", 1)[1].strip("/")
    elif clean.startswith("jobs/"):
        clean = clean[len("jobs/") :].strip("/")

    if not clean or clean == ".":
        raise ValueError("Job target cannot be empty or root.")

    return clean


def resolve_job_file_pattern(clean_target: str, filename: str) -> str:
    """Construct workspace-relative glob pattern for a target and filename.

    Args:
        clean_target: Normalized target (e.g. 'my_task/ee76fb8f' or 'ee76fb8f').
        filename: Relative filename to find (e.g. 'params.json', 'experimaestro.json').

    Returns:
        Workspace-relative glob pattern string under the jobs directory.
    """
    clean_target = clean_target.replace(":", "/")
    if "/" in clean_target:
        return f"jobs/{clean_target}*/{filename}"
    return f"jobs/*/{clean_target}*/{filename}"


def build_ssh_cmd(ssh_settings: Any, remote_command: str) -> List[str]:
    """Construct an SSH command list respecting options and shell initialization.

    Args:
        ssh_settings: SSHSettings object with host, options, and shell_init.
        remote_command: Command string to execute on the remote host.

    Returns:
        Command list suitable for subprocess.run.
    """
    cmd = ["ssh"]
    if ssh_settings.options:
        cmd.extend(ssh_settings.options)
    cmd.append(ssh_settings.host)

    full_remote_cmd = remote_command
    if ssh_settings.shell_init:
        full_remote_cmd = f"{ssh_settings.shell_init}; {remote_command}"

    cmd.append(full_remote_cmd)
    return cmd


def run_ssh_command(
    ssh_settings: Any, remote_command: str
) -> subprocess.CompletedProcess:
    """Execute a command string on the remote host via SSH with captured output.

    Args:
        ssh_settings: SSHSettings instance.
        remote_command: Command string to run remotely.

    Returns:
        CompletedProcess instance with stdout and stderr.
    """
    cmd = build_ssh_cmd(ssh_settings, remote_command)
    return subprocess.run(cmd, capture_output=True, text=True)


def fetch_remote_file_rsync(
    ssh_settings: Any, remote_path: str, local_dest: Path
) -> None:
    """Transfer a single remote file to local_dest using rsync or scp.

    Args:
        ssh_settings: SSHSettings object with host and options.
        remote_path: Absolute or workspace path on the remote host.
        local_dest: Local Path where the file should be written.

    Raises:
        RuntimeError: If both rsync and scp transfer attempts fail.
    """
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    source = f"{ssh_settings.host}:{remote_path}"
    dest = str(local_dest)

    cmd = ["rsync", "-L", "-a"]
    if ssh_settings.options:
        ssh_opt_str = " ".join(shlex.quote(opt) for opt in ssh_settings.options)
        cmd.extend(["-e", f"ssh {ssh_opt_str}"])
    cmd.extend([source, dest])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to scp if rsync failed or is unavailable
        scp_cmd = ["scp"]
        if ssh_settings.options:
            scp_cmd.extend(ssh_settings.options)
        scp_cmd.extend([source, dest])
        scp_res = subprocess.run(scp_cmd, capture_output=True, text=True)
        if scp_res.returncode != 0:
            err = res.stderr.strip() or scp_res.stderr.strip()
            raise RuntimeError(
                f"Failed to transfer {remote_path} from {ssh_settings.host}: {err}"
            )


def fetch_remote_files_rsync(
    ssh_settings: Any,
    remote_src_dir: str,
    local_dest_dir: Path,
    includes: List[str],
) -> None:
    """Transfer specific files matching includes from remote directory using rsync.

    Args:
        ssh_settings: SSHSettings object.
        remote_src_dir: Remote source directory path.
        local_dest_dir: Local destination directory.
        includes: List of filename patterns to include (e.g. ['objects.jsonl', 'jobs.jsonl']).
    """
    local_dest_dir.mkdir(parents=True, exist_ok=True)
    source = f"{ssh_settings.host}:{remote_src_dir.rstrip('/')}/"
    dest = f"{str(local_dest_dir).rstrip('/')}/"

    cmd = ["rsync", "-L", "-a"]
    if ssh_settings.options:
        ssh_opt_str = " ".join(shlex.quote(opt) for opt in ssh_settings.options)
        cmd.extend(["-e", f"ssh {ssh_opt_str}"])

    for inc in includes:
        cmd.extend(["--include", inc])
    cmd.extend(["--exclude", "*", source, dest])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to pulling files individually
        for inc in includes:
            remote_file = f"{remote_src_dir.rstrip('/')}/{inc}"
            local_file = local_dest_dir / inc
            try:
                fetch_remote_file_rsync(ssh_settings, remote_file, local_file)
            except Exception:
                pass


def find_remote_workspace_file(
    ws: Any,
    target: str,
    filename: str = "params.json",
    cache_dir: Optional[Path] = None,
) -> Path:
    """Fetch a file from a remote workspace job directory using SSH and Rsync.

    Resolves remote workspace job directories matching `target` and `filename`
    via SSH pattern matching, and synchronizes the file into a local cache directory.

    Args:
        ws: Remote WorkspaceSettings instance with SSH configuration.
        target: Job ID / hash prefix or remote job directory subpath.
        filename: Relative filename to fetch (default: 'params.json').
        cache_dir: Optional local directory to store fetched files. If None,
            uses the managed session temporary cache directory.

    Returns:
        Local Path to the synchronized file.

    Raises:
        FileNotFoundError: If target is empty, or no matching file exists remotely.
        RuntimeError: If remote SSH command execution or file transfer fails.
    """
    ws_raw = str(getattr(ws, "_raw_path", "")).strip("/")
    try:
        clean_target = clean_job_target(target, ws_raw, filename=filename)
    except ValueError as e:
        raise FileNotFoundError(
            f"Please specify a job ID or path for remote workspace '{ws.id}'."
        ) from e

    pattern = resolve_job_file_pattern(clean_target, filename)
    remote_pattern = f"{ws._raw_path}/{pattern}"

    logger.info(
        "Querying remote workspace %s via SSH for %s matching '%s'...",
        ws.id,
        filename,
        clean_target,
    )

    py_bin = ws.ssh.python or "python3"
    py_script = "import glob, sys; [print(p) for p in glob.glob(sys.argv[1])]"
    remote_cmd = f"{py_bin} -c {py_script!r} {shlex.quote(remote_pattern)}"

    res = run_ssh_command(ws.ssh, remote_cmd)
    if res.returncode != 0:
        err = res.stderr.strip() or res.stdout.strip()
        raise RuntimeError(f"Failed to query remote workspace '{ws.id}' via SSH: {err}")

    matches = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not matches:
        raise FileNotFoundError(
            f"No job matching '{clean_target}' found in remote workspace '{ws.id}' ({ws._raw_path})"
        )

    if len(matches) > 1:
        print(  # noqa: T201
            f"Warning: Multiple jobs ({len(matches)}) matched '{clean_target}'. "
            f"Using the first match: {matches[0]}",
            file=sys.stderr,
        )

    remote_file_path = matches[0]
    rel_path = remote_file_path[len(ws._raw_path) :].lstrip("/")
    dest_cache = cache_dir or get_session_cache_dir()
    local_dest = dest_cache / rel_path

    logger.info("Synchronizing %s from %s into %s...", rel_path, ws.id, local_dest)
    fetch_remote_file_rsync(ws.ssh, remote_file_path, local_dest)

    if not local_dest.exists():
        raise FileNotFoundError(
            f"{filename} not found in local cache {local_dest} after transfer."
        )

    return local_dest


fetch_remote_workspace_file = find_remote_workspace_file


def find_remote_workspace_experiment_run(
    ws: Any,
    experiment_id: str,
    run_id: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[str, Path]:
    """Find and synchronize an experiment run's metadata from a remote workspace.

    Queries the remote workspace for available runs in `experiments/<experiment_id>/`,
    selects the requested run (or the most recent one), and pulls `objects.jsonl`,
    `jobs.jsonl`, and `progress.jsonl` into the local session cache directory.

    Args:
        ws: Remote WorkspaceSettings instance.
        experiment_id: Experiment identifier.
        run_id: Optional specific run timestamp ID. If None, picks the latest.
        cache_dir: Optional cache directory.

    Returns:
        Tuple of (selected_run_id, local_run_dir_path).

    Raises:
        FileNotFoundError: If experiment or run cannot be found remotely.
        RuntimeError: If remote SSH command execution fails.
    """
    ws_raw = str(getattr(ws, "_raw_path", "")).rstrip("/")
    exp_remote_path = f"{ws_raw}/experiments/{experiment_id}"

    py_bin = ws.ssh.python or "python3"
    # List run directories under the experiment directory
    py_script = (
        "import os, sys, glob; "
        "d = sys.argv[1]; "
        "runs = [os.path.basename(p) for p in glob.glob(os.path.join(d, '*')) if os.path.isdir(p)]; "
        "print('\\n'.join(sorted(runs)))"
    )
    remote_cmd = f"{py_bin} -c {py_script!r} {shlex.quote(exp_remote_path)}"

    res = run_ssh_command(ws.ssh, remote_cmd)
    if res.returncode != 0:
        err = res.stderr.strip() or res.stdout.strip()
        raise RuntimeError(
            f"Failed to query experiment '{experiment_id}' in remote workspace '{ws.id}': {err}"
        )

    available_runs = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not available_runs:
        raise FileNotFoundError(
            f"No runs found for experiment '{experiment_id}' in remote workspace '{ws.id}'."
        )

    if run_id:
        if run_id not in available_runs:
            raise FileNotFoundError(
                f"Run '{run_id}' not found for experiment '{experiment_id}' in remote workspace '{ws.id}'. "
                f"Available runs: {available_runs}"
            )
        selected_run_id = run_id
    else:
        selected_run_id = available_runs[-1]

    remote_run_dir = f"{exp_remote_path}/{selected_run_id}"
    dest_cache = cache_dir or get_session_cache_dir()
    local_run_dir = dest_cache / "experiments" / experiment_id / selected_run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Synchronizing DAG files for run %s (%s) from %s into %s...",
        selected_run_id,
        experiment_id,
        ws.id,
        local_run_dir,
    )

    # Fetch lightweight metadata files needed for DAG visualization
    includes = ["objects.jsonl", "jobs.jsonl", "progress.jsonl", "run.yaml"]
    fetch_remote_files_rsync(ws.ssh, remote_run_dir, local_run_dir, includes)

    if not (local_run_dir / "objects.jsonl").exists():
        # Fallback: pull objects.jsonl individually to verify
        remote_obj = f"{remote_run_dir}/objects.jsonl"
        fetch_remote_file_rsync(ws.ssh, remote_obj, local_run_dir / "objects.jsonl")

    return selected_run_id, local_run_dir
