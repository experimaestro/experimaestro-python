"""Click CLI command for Experimaestro Parameters Inspector (`experimaestro jobs inspect`)."""

from __future__ import annotations

import sys

import click

from .remote import RemoteCommand


@click.command(name="inspect", cls=RemoteCommand)
@click.argument("target", required=False, default=".")
@click.option(
    "-w",
    "--workspace",
    default=None,
    help="Workspace ID or directory path (defaults to current Experimaestro workspace; supports remote SSH workspaces).",
)
def inspect_command(target: str, workspace: str | None) -> None:
    """Interactive TUI Parameters Inspector for Experimaestro jobs.

    TARGET can be a Job ID or hash prefix (e.g., `mymodule.MyTask/1822b2b`),
    a path to a `params.json` or `experimaestro.json` file, or a job directory.
    Defaults to `.` (current directory or default workspace).

    Examples:

    \b
        # Inspect current directory / job
        experimaestro jobs inspect

    \b
        # Inspect a specific job by task name or hash prefix
        experimaestro jobs inspect [mymodule.MyTask/]1822b2b

    \b
        # Inspect a specific params.json directly
        experimaestro jobs inspect /path/to/params.json

    \b
        # Inspect a job in a specific workspace (workspace can be local or remote)
        experimaestro jobs inspect -w my_(remote)_workspace [mymodule.MyTask/]1822b2b
    """
    from experimaestro.tools.inspect_params import (
        ParamsInspectorApp,
        find_params_file,
    )

    from experimaestro.cli import root_config

    workspace = workspace or root_config().workspace_spec
    try:
        params_path = find_params_file(target, workspace)
        app = ParamsInspectorApp(params_path)
        app.run()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running parameters inspector: {e}", err=True)
        sys.exit(1)
