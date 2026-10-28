"""Click CLI command for Experimaestro Parameters Inspector (`experimaestro jobs inspect`)."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from termcolor import cprint


@click.command(name="inspect")
@click.argument("target", required=False, default=".")
@click.option("-w", "--workspace", default=None, hidden=True)
def inspect_command(target: str, workspace: str | None) -> None:
    """Interactive TUI Parameters Inspector for jobs and serialized objects.

    TARGET can be a Job ID or hash prefix (e.g., `mymodule.MyTask/1822b2b`),
    a path to a `params.json`, `experimaestro.json` (e.g. in exported models)
    or `objects.jsonl` file, or a job directory.
    Defaults to `.` (current directory or default workspace).

    The workspace (local or remote) is selected with the root options
    `experimaestro [-w WORKSPACE | --workdir PATH | --ssh HOST:PATH]`.
    The shortcut `experimaestro inspect` can be used instead of
    `experimaestro jobs inspect`.

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
        find_job_params_file,
        find_params_file,
    )

    from experimaestro.cli import root_config

    cfg = root_config()
    if workspace is not None:
        cprint(
            "Deprecated: use `experimaestro -w WORKSPACE jobs inspect`",
            "yellow",
            file=sys.stderr,
        )
        if Path(workspace).expanduser().is_dir():
            cfg.set_workspace(Path(workspace).expanduser(), None)
        else:
            cfg.set_workspace(None, workspace)

    try:
        if (remote := cfg.remote) is not None:
            # The TUI runs locally: the files are fetched through the remote
            # state provider (kept open, with its cache, until the command ends)
            params_path = find_job_params_file(
                cfg.state_provider(), target, remote.path
            )
        else:
            params_path = find_params_file(target, cfg.workspace_spec)
        app = ParamsInspectorApp(params_path)
        app.run()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running parameters inspector: {e}", err=True)
        sys.exit(1)
