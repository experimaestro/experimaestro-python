"""Custom hatch build hook to build the web frontend."""

import os
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class WebUIBuildHook(BuildHookInterface):
    """Build hook that compiles the React frontend before packaging."""

    PLUGIN_NAME = "webui"

    def initialize(self, version: str, build_data: dict) -> None:
        """Run npm build to compile the frontend."""
        # Skip on ReadTheDocs - npm is not available there
        if os.environ.get("READTHEDOCS"):
            return

        root = Path(self.root)
        app_dir = root / "app"
        output_dir = root / "src" / "experimaestro" / "webui" / "data"

        # Skip if output directory already exists and has content
        if output_dir.exists() and any(output_dir.iterdir()):
            return

        if not app_dir.exists():
            # No app directory - skip silently (might be an sdist without app)
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
        except subprocess.CalledProcessError as e:
            print(f"npm install failed: {e.stderr.decode()}", file=sys.stderr)  # noqa: T201
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
        except subprocess.CalledProcessError as e:
            print(f"npm build failed: {e.stderr.decode()}", file=sys.stderr)  # noqa: T201
            raise

        print("Web frontend built successfully.", file=sys.stderr)  # noqa: T201
