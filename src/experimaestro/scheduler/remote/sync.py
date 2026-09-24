"""Remote File Synchronizer

Handles rsync-based file synchronization between remote and local workspaces
for SSH-based experiment monitoring. Only syncs specific paths on-demand
when services need them (e.g., TensorboardService).
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("xpm.remote.sync")


class RemoteFileSynchronizer:
    """Handles rsync-based file synchronization for remote monitoring

    Syncs specific paths on-demand from a remote host to a local cache
    directory. Used when services need access to remote files.
    """

    def __init__(
        self,
        host: str,
        remote_workspace: Path,
        local_cache: Path,
        ssh_options: Optional[List[str]] = None,
        compress: bool = False,
        bwlimit: Optional[int] = None,
    ):
        """Initialize the synchronizer

        Args:
            host: SSH host (user@host or just host)
            remote_workspace: Path to workspace on the remote host
            local_cache: Local directory to sync files to
            ssh_options: Additional SSH options (e.g., ["-p", "2222"])
            compress: Enable gzip compression (-z). Default is False to avoid
                      CPU bottlenecks on binary event files.
            bwlimit: Bandwidth limit in KB/s to prevent starving SSH multiplexed sockets.
        """
        self.host = host
        self.remote_workspace = remote_workspace
        self.local_cache = local_cache
        self.ssh_options = ssh_options or []
        self.compress = compress
        self.bwlimit = bwlimit

    def sync_path(
        self,
        remote_path: str,
        include: list[str] | None = None,
        compress: Optional[bool] = None,
    ) -> Path:
        """Sync a specific path from remote

        Args:
            remote_path: Absolute path on remote or path relative to workspace
            include: Optional list of filename patterns to include (e.g., ["*.out", "*.err"]).
                    If provided, only files matching these patterns will be synced.
            compress: Override compression setting for this sync.

        Returns:
            Local path where the files were synced to
        """
        # Normalize the path - get relative path within workspace
        if remote_path.startswith(str(self.remote_workspace)):
            relative_path = remote_path[len(str(self.remote_workspace)) :].lstrip("/")
        else:
            relative_path = remote_path.lstrip("/")

        if not relative_path:
            raise ValueError("Cannot sync empty path")

        logger.info("Syncing path: %s", relative_path)

        # Build source and destination
        source = f"{self.host}:{self.remote_workspace}/{relative_path}/"
        local_path = self.local_cache / relative_path
        local_path.mkdir(parents=True, exist_ok=True)
        dest = f"{local_path}/"

        use_compress = self.compress if compress is None else compress
        self._rsync(source, dest, include=include, compress=use_compress)

        return local_path

    def _rsync(
        self,
        source: str,
        dest: str,
        include: list[str] | None = None,
        compress: bool = False,
    ):
        """Execute rsync command

        Args:
            source: Remote source path (host:path/)
            dest: Local destination path
            include: Optional list of filename patterns to include
            compress: Whether to enable gzip compression
        """
        cmd = [
            "rsync",
            "--inplace",  # Update destination files in-place
            "--delete",  # Delete extraneous files from destination
            "-L",  # Transform symlinks into referent file/dir
            "-a",  # Archive mode (preserves permissions, times, etc.)
            "-v",  # Verbose
        ]

        if compress:
            cmd.append("-z")

        if self.bwlimit:
            cmd.append(f"--bwlimit={self.bwlimit}")

        # Include/exclude patterns (order matters: includes first, then exclude all)
        # We include "*/" first so directory hierarchies are traversed to find matching files
        if include:
            cmd.extend(["--include", "*/"])
            for pattern in include:
                if pattern != "*/":
                    cmd.extend(["--include", pattern])
            cmd.extend(["--exclude", "*", "--prune-empty-dirs"])

        # SSH options (use shlex.join for proper quoting of paths)
        if self.ssh_options:
            import shlex

            ssh_cmd = "ssh " + " ".join(shlex.quote(opt) for opt in self.ssh_options)
            cmd.extend(["-e", ssh_cmd])

        cmd.extend([source, dest])

        logger.debug("Running rsync: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                # rsync returns non-zero for some warnings
                if result.returncode == 23:
                    # Partial transfer due to error - some files may be missing
                    logger.warning("Rsync partial transfer: %s", result.stderr)
                elif result.returncode == 24:
                    # Partial transfer due to vanished source files
                    logger.debug("Rsync: some source files vanished")
                else:
                    logger.error(
                        "Rsync failed (code %d): %s",
                        result.returncode,
                        result.stderr,
                    )
                    raise RuntimeError(f"Rsync failed: {result.stderr}")
            else:
                logger.debug("Rsync completed successfully")

        except subprocess.TimeoutExpired:
            logger.error("Rsync timed out")
            raise
        except FileNotFoundError:
            logger.error("rsync command not found - please install rsync")
            raise RuntimeError("rsync command not found")

    def get_local_path(self, remote_path: str) -> Path:
        """Get the local cache path for a remote path

        Args:
            remote_path: Absolute path on the remote system

        Returns:
            Corresponding path in the local cache
        """
        if remote_path.startswith(str(self.remote_workspace)):
            relative = remote_path[len(str(self.remote_workspace)) :].lstrip("/")
            return self.local_cache / relative
        return Path(remote_path)
