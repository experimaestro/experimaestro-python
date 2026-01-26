"""Custom hatch build hook to build the web frontend."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class WebUIBuildHook(BuildHookInterface):
    """Build hook that compiles the React frontend before packaging."""

    PLUGIN_NAME = "webui"

    def initialize(self, version: str, build_data: dict) -> None:
        """Run npm build to compile the frontend."""
        root = Path(self.root)
        app_dir = root / "app"
        output_dir = root / "src" / "experimaestro" / "webui" / "data"

        def _handle_missing_webui():
            """Create an empty directory to satisfy the force-include directive"""
            # if the web UI is not built.
            output_dir.mkdir(parents=True, exist_ok=True)

        # Skip on ReadTheDocs - npm is not available there
        if os.environ.get("READTHEDOCS"):
            _handle_missing_webui()
            return

        # Skip if output directory already exists and has content
        if output_dir.exists() and any(output_dir.iterdir()):
            return

        if not app_dir.exists():
            # No app directory - skip silently (might be an sdist without app)
            _handle_missing_webui()
            return

        if not shutil.which("npm"):
            print(
                "warning: npm not found. Skipping web UI build.",
                file=sys.stderr,
            )
            _handle_missing_webui()
            return

        print("Building web frontend...", file=sys.stderr)  # noqa: T201

        # Run npm install
        try:
            subprocess.run(
                ["npm", "install", "--no-save"],
                cwd=app_dir,
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"npm install failed: {e!s}", file=sys.stderr)  # noqa: T201
            raise

        # Run npm build
        env = os.environ.copy()
        env["NODE_OPTIONS"] = "--loader ts-node/esm"
        try:
            subprocess.run(
                ["npm", "run", "build"],
                cwd=app_dir,
                check=True,
                capture_output=True,
                env=env,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"npm build failed: {e!s}", file=sys.stderr)  # noqa: T201
            raise

        print("Web frontend built successfully.", file=sys.stderr)  # noqa: T201
