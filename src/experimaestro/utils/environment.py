"""Environment capture utilities for experiment reproducibility

This module provides functions to capture the full Python environment state
when experiments are run, including:
- Git information for editable (development) packages
- Version information for all installed Python packages
- Run information (hostname, start time) for experiment runs
"""

import json
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from importlib.metadata import distributions

from experimaestro.utils.git import get_git_info

logger = logging.getLogger("xpm.environment")


@dataclass
class ExperimentRunInfo:
    """Information about a single experiment run"""

    hostname: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    status: Optional[str] = None  # 'completed' or 'failed'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "hostname": self.hostname,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRunInfo":
        """Create from dictionary"""
        return cls(
            hostname=data.get("hostname"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            status=data.get("status"),
        )


@dataclass
class ExperimentEnvironment:
    """Experiment environment stored in environment.json

    This combines Python environment info with experiment run metadata.
    """

    python_version: Optional[str] = None
    packages: Dict[str, str] = field(default_factory=dict)
    editable_packages: Dict[str, Any] = field(default_factory=dict)
    run: Optional[ExperimentRunInfo] = None
    projects: list[Dict[str, Any]] = field(
        default_factory=list
    )  # Git info for projects

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result: Dict[str, Any] = {
            "python_version": self.python_version,
            "packages": self.packages,
            "editable_packages": self.editable_packages,
        }
        if self.run is not None:
            result["run"] = self.run.to_dict()
        if self.projects:
            result["projects"] = self.projects
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentEnvironment":
        """Create from dictionary"""
        run_data = data.get("run")
        run = ExperimentRunInfo.from_dict(run_data) if run_data else None
        return cls(
            python_version=data.get("python_version"),
            packages=data.get("packages", {}),
            editable_packages=data.get("editable_packages", {}),
            run=run,
            projects=data.get("projects", []),
        )

    @classmethod
    def load(cls, path: Path) -> "ExperimentEnvironment":
        """Load from a JSON file"""
        if not path.exists():
            return cls()
        try:
            with path.open("r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to read environment.json: %s", e)
            return cls()

    def save(self, path: Path) -> None:
        """Save to a JSON file"""
        path.write_text(json.dumps(self.to_dict(), indent=2))


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


def get_current_environment() -> ExperimentEnvironment:
    """Get current environment as an ExperimentEnvironment object

    Returns:
        ExperimentEnvironment with current Python version, packages, etc.
    """
    current_info = get_environment_info()
    return ExperimentEnvironment(
        python_version=current_info["python_version"],
        packages=current_info["packages"],
        editable_packages=current_info["editable_packages"],
    )


def load_environment_info(path: Path) -> Optional[ExperimentEnvironment]:
    """Load environment information from a JSON file

    Args:
        path: Path to the environment info JSON file

    Returns:
        ExperimentEnvironment if file exists and is valid, None otherwise
    """
    if not path.exists():
        return None

    return ExperimentEnvironment.load(path)
