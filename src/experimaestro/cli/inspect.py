"""Click CLI command for Experimaestro Parameters Inspector (`experimaestro inspect`)."""

from __future__ import annotations

import sys

import click


@click.command(name="inspect")
@click.argument("target", required=False, default=".")
def inspect_command(target: str = ".") -> None:
    """Interactive TUI Parameters Inspector for Experimaestro jobs and serialized objects.

    TARGET can be a Job ID or hash prefix (e.g., `mymodule.MyTask/1822b2b`),
    a path to a `params.json`, `experimaestro.json` (e.g. in exported models),
    or `objects.jsonl` file, or a job directory.
    Defaults to `.` (current directory or default workspace).

    The workspace is selected with the root option `experimaestro -w WORKSPACE`.

    Examples:

    \b
        # Inspect current directory / job
        experimaestro inspect

    \b
        # Inspect a specific job by task name or hash prefix
        experimaestro inspect [mymodule.MyTask/]1822b2b

    \b
        # Inspect a specific params.json or exported model config directly
        experimaestro inspect /path/to/experimaestro.json

    \b
        # Inspect a job in a specific workspace (local or remote)
        experimaestro -w my_workspace inspect [mymodule.MyTask/]1822b2b
    """
    from experimaestro.tools.inspect_params import (
        ParamsInspectorApp,
        find_params_file,
    )

    from experimaestro.cli import root_config

    workspace = root_config().workspace_spec
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
