"""Environment capture utilities for experiment reproducibility

This module provides functions to capture the full Python environment state
when experiments are run, including:
- Git information for editable (development) packages
- Version information for all installed Python packages
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from importlib.metadata import distributions

from experimaestro.utils.git import get_git_info

logger = logging.getLogger("xpm.environment")


def get_environment_info() -> dict:
    """Capture complete environment information for reproducibility

    Returns:
        Dictionary containing:
        - python_version: Python interpreter version
        - packages: Dict of package_name -> version for all installed packages
        - editable_packages: Dict of package_name -> {version, path, git_info}
          for packages installed in editable mode
    """
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    packages = {}
    editable_packages = {}

    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        packages[name] = version

        # Check if this is an editable install by looking for direct_url.json
        # PEP 610 specifies that editable installs have a direct_url.json
        # with "dir_info": {"editable": true}
        direct_url_file = dist._path / "direct_url.json"  # type: ignore
        if direct_url_file.exists():
            try:
                import json

                direct_url = json.loads(direct_url_file.read_text())
                dir_info = direct_url.get("dir_info", {})
                if dir_info.get("editable", False):
                    # Get the source path from the URL
                    url = direct_url.get("url", "")
                    if url.startswith("file://"):
                        source_path = Path(url[7:])  # Remove "file://"
                        git_info = get_git_info(source_path)
                        editable_packages[name] = {
                            "version": version,
                            "path": str(source_path),
                            "git": git_info,
                        }
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Error reading direct_url.json for %s: %s", name, e)

    return {
        "python_version": python_version,
        "packages": packages,
        "editable_packages": editable_packages,
    }


def get_editable_packages_git_info() -> dict:
    """Get git information for all editable packages

    This is a lighter-weight function that only captures git info for
    editable packages, without listing all installed packages.

    Returns:
        Dictionary mapping package_name -> {version, path, git}
        for packages installed in editable mode
    """
    editable_packages = {}

    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]

        # Check for editable install via direct_url.json
        direct_url_file = dist._path / "direct_url.json"  # type: ignore
        if direct_url_file.exists():
            try:
                import json

                direct_url = json.loads(direct_url_file.read_text())
                dir_info = direct_url.get("dir_info", {})
                if dir_info.get("editable", False):
                    url = direct_url.get("url", "")
                    if url.startswith("file://"):
                        source_path = Path(url[7:])
                        git_info = get_git_info(source_path)
                        editable_packages[name] = {
                            "version": version,
                            "path": str(source_path),
                            "git": git_info,
                        }
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Error reading direct_url.json for %s: %s", name, e)

    return editable_packages


def save_environment_info(path: Path) -> dict:
    """Save environment information to a JSON file

    Args:
        path: Path to save the environment info JSON file

    Returns:
        The environment info dictionary that was saved
    """
    import json

    env_info = get_environment_info()
    path.write_text(json.dumps(env_info, indent=2))
    logger.info("Saved environment info to %s", path)
    return env_info


def load_environment_info(path: Path) -> Optional[dict]:
    """Load environment information from a JSON file

    Args:
        path: Path to the environment info JSON file

    Returns:
        Environment info dictionary if file exists and is valid, None otherwise
    """
    import json

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error loading environment info from %s: %s", path, e)
        return None
