"""Git utilities for tracking repository state during experiments

This module provides functions to capture git repository information
when experiments are run, enabling reproducibility by recording
which code version was used.
"""

import subprocess
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("xpm.git")


def get_git_info(cwd: Optional[Path] = None) -> Optional[dict]:
    """Get git repository information for a directory

    Args:
        cwd: Working directory to check (defaults to current working directory)

    Returns:
        Dictionary with git info if in a git repository, None otherwise.
        The dictionary contains:
        - commit: Full commit hash (40 characters)
        - commit_short: Short commit hash (7 characters)
        - branch: Current branch name (None if detached HEAD)
        - dirty: True if working directory has uncommitted changes
        - message: First line of commit message
        - author: Author of the commit
        - date: Commit date in ISO format
    """
    if cwd is None:
        cwd = Path.cwd()

    try:
        # Check if this is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("Not a git repository: %s", cwd)
            return None

        # Get full commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()

        # Get short commit hash
        commit_short = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()

        # Get current branch (may be None for detached HEAD)
        branch_result = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

        # Check if working directory is dirty
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        )
        dirty = bool(dirty_result.stdout.strip())

        # Get commit message (first line)
        message = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()

        # Get commit author
        author = subprocess.run(
            ["git", "log", "-1", "--format=%an <%ae>"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()

        # Get commit date in ISO format
        date = subprocess.run(
            ["git", "log", "-1", "--format=%aI"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()

        return {
            "commit": commit,
            "commit_short": commit_short,
            "branch": branch,
            "dirty": dirty,
            "message": message,
            "author": author,
            "date": date,
        }

    except FileNotFoundError:
        logger.debug("Git command not found")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning("Error getting git info: %s", e)
        return None
