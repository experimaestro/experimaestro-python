"""Experimaestro CLI command for experiment dependency graphs"""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path

import click
from termcolor import cprint

from experimaestro.scheduler.state_provider import StateProvider


def _open_provider(workspace: str | None, experiment_id: str | None) -> StateProvider:
    """State provider for a workspace given by ID or path (or found from settings)"""
    from experimaestro.cli import root_config
    from experimaestro.settings import find_workspace, get_workspace

    cfg = root_config()
    if workspace:
        if get_workspace(workspace, include_remote=True) is not None:
            cfg.set_workspace(None, workspace)
        else:
            cfg.set_workspace(Path(workspace).expanduser(), None)
    if cfg.workspace_spec or cfg.remote is not None:
        return cfg.state_provider()

    # No workspace given: the experiment may select one (workspace triggers)
    from experimaestro.scheduler.workspace_state_provider import (
        WorkspaceStateProvider,
    )

    ws = find_workspace(experiment_id=experiment_id)
    return WorkspaceStateProvider(ws.path, no_cleanup=True)


def _parse_filter(spec: str) -> dict[str, object]:
    """Parse ``key=value,...`` (values are read as JSON when possible)"""
    tags: dict[str, object] = {}
    for item in spec.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise click.BadParameter(f"Expected key=value, got {item!r}")
        try:
            tags[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            tags[key.strip()] = value.strip()
    return tags


@click.command(name="dag")
@click.argument("experiment_id", required=False, default=None)
@click.option("-r", "--run", "run_id", default=None, help="Run ID (default: current)")
@click.option(
    "-w",
    "--workspace",
    default=None,
    help="Workspace ID or directory (default: from the settings)",
)
@click.option(
    "--html",
    "html_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the graph as a standalone HTML page",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path, allow_dash=True),
    default=None,
    help="Save the graph and its views as JSON ('-' for the standard output)",
)
@click.option("-t", "--tree", is_flag=True, help="Print the graph, stage by stage")
@click.option(
    "--list-configs", is_flag=True, help="List the grid search configurations"
)
@click.option(
    "-c",
    "--config",
    "config_index",
    type=int,
    default=None,
    help="Restrict --tree to a configuration (see --list-configs)",
)
@click.option(
    "--filter",
    "filter_spec",
    default=None,
    help="Restrict --tree to the nodes with these tags (e.g. 'lr=0.001,seed=1')",
)
@click.option(
    "--all-nodes",
    is_flag=True,
    help="With --tree, also show the configurations (not only jobs)",
)
@click.option(
    "--stack/--no-stack",
    default=True,
    help="With --tree, stack similar nodes (default: stacked)",
)
def dag(
    experiment_id: str | None,
    run_id: str | None,
    workspace: str | None,
    html_path: Path | None,
    json_path: Path | None,
    tree: bool,
    list_configs: bool,
    config_index: int | None,
    filter_spec: str | None,
    all_nodes: bool,
    stack: bool,
) -> None:
    """Show the dependency graph of an experiment

    By default, the graph is opened in the browser as a standalone page (the
    same view as the web UI graph).

    \b
    Examples:
      experimaestro experiments dag my_experiment                  # Open in the browser
      experimaestro experiments dag my_experiment --html dag.html  # Save a standalone page
      experimaestro experiments dag my_experiment --tree -c 0      # Print configuration #0
      experimaestro experiments dag my_experiment --json -         # Dump as JSON
    """
    from experimaestro.cli.experiments import _select_experiment
    from experimaestro.ui.graph import (
        export_state,
        load_experiment_graph,
        run_statuses,
        standalone_html,
    )

    provider = _open_provider(workspace, experiment_id)
    try:
        exp = _select_experiment(provider, experiment_id)
        if exp is None:
            raise click.exceptions.Exit(1)
        try:
            graph = load_experiment_graph(provider, exp.experiment_id, run_id)
        except FileNotFoundError as e:
            cprint(f"No dependency graph for this run: {e}", "red")
            raise click.exceptions.Exit(1)
        statuses = run_statuses(provider, exp.experiment_id, run_id)
    finally:
        provider.close()

    if list_configs:
        for ix, config in enumerate(graph.configs):
            click.echo(f"{ix:>3}  " + ", ".join(f"{k}={v}" for k, v in config.items()))
        if not graph.configs:
            cprint("No grid search configuration", "yellow")
        return

    if json_path is not None:
        with click.open_file(str(json_path), "w") as out:
            json.dump(export_state(graph, statuses), out)

    if html_path is not None:
        html_path.write_text(standalone_html(graph, statuses), encoding="utf-8")
        cprint(f"Graph saved to {html_path}", "green")

    if tree:
        config: int | dict | None = config_index
        if filter_spec:
            config = _parse_filter(filter_spec)
        elif config_index is not None and not 0 <= config_index < len(graph.configs):
            raise click.BadParameter(
                f"Configuration {config_index} does not exist (see --list-configs)"
            )
        view = graph.view(config, tasks_only=not all_nodes, stack=stack)
        for depth, stage in enumerate(graph.stages(view)):
            cprint(f"Stage {depth}", "cyan")
            for vn in stage:
                node = graph.nodes[vn.id]
                count = f" ×{len(vn.members)}" if len(vn.members) > 1 else ""
                tags = ", ".join(f"{k}={v}" for k, v in (vn.tags or node.tags).items())
                status = sorted({statuses.get(m, "") for m in vn.members} - {""})
                click.echo(
                    f"  {node.name}{count}"
                    + (f"  [{', '.join(status)}]" if status else "")
                    + (f"  {tags}" if tags else "")
                )

    if html_path is None and json_path is None and not tree:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", prefix="xpm-dag-", delete=False, encoding="utf-8"
        ) as out:
            out.write(standalone_html(graph, statuses))
        cprint(f"Opening {out.name}", "green")
        webbrowser.open(Path(out.name).as_uri())
