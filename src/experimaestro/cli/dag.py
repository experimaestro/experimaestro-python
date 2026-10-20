"""Experimaestro CLI command for experiment dependency DAG visualization and export."""

from __future__ import annotations

from pathlib import Path
import click

from experimaestro.tools.xpm_dag.draw_dag import run_dag


@click.command(
    name="dag",
    help="Visualize and export experiment dependency DAGs (web, HTML, JSON, plot, tree).",
)
@click.argument("experiment_id", required=False, default=None)
@click.option(
    "-e",
    "--experiment",
    "--experiment-id",
    "experiment_opt",
    default=None,
    help="Experiment identifier (alias for EXPERIMENT_ID argument).",
)
@click.option(
    "-r",
    "--run",
    "--run-id",
    "run_id",
    default=None,
    help="Run timestamp ID (default: most recent run).",
)
@click.option(
    "-w",
    "--workspace",
    default=None,
    help="Workspace ID or directory path (default: auto-detected).",
)
@click.option(
    "-W/-nw",
    "--web/--no-web",
    "web",
    default=None,
    help="Interactive web viewer in browser (default: enabled unless --tree, --plot, or file export is specified).",
)
@click.option(
    "-t",
    "--tree",
    is_flag=True,
    help="Print text tree representation of the DAG to terminal.",
)
@click.option(
    "--plot",
    is_flag=True,
    default=False,
    help="Display matplotlib interactive plot window (requires matplotlib).",
)
@click.option(
    "--port",
    type=int,
    default=0,
    help="Port for web server (default: random available port).",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Do not automatically open browser when launching web viewer.",
)
@click.option(
    "--save-as",
    type=click.Path(path_type=Path),
    default=None,
    help="Path where output should be saved (auto-detects .html, .json, .png, .svg, .pdf).",
)
@click.option(
    "--html",
    type=click.Path(path_type=Path),
    default=None,
    help="Path where standalone interactive HTML page should be saved.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Export DAG as JSON to standard output.",
)
@click.option(
    "--json-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Export DAG as JSON to specified file path (see also --save-as).",
)
@click.option(
    "--tasks-only",
    is_flag=True,
    help="Only display Task and Action nodes, skipping static config nodes.",
)
@click.option(
    "--list-configs",
    is_flag=True,
    help="List all unique configuration parameter sets found in the experiment and exit.",
)
@click.option(
    "--single-config",
    is_flag=True,
    help="Render DAG for a single grid configuration (defaults to config index 0).",
)
@click.option(
    "-c",
    "--config-index",
    type=int,
    default=None,
    help="Index of the configuration to display (see --list-configs). Implies --single-config.",
)
@click.option(
    "--filter",
    "filter_tags",
    default=None,
    help="Filter the DAG by tag values (e.g. 'lr=0.001,kernel_size=3').",
)
@click.option(
    "--include-actions",
    is_flag=True,
    help="Include downstream Action and EvaluatedModel nodes when filtering by configuration.",
)
@click.option(
    "--layout",
    type=click.Choice(["dagre", "aligned"], case_sensitive=False),
    default="dagre",
    help="Layout algorithm for interactive web view: 'dagre' (default) or 'aligned'.",
)
@click.option(
    "--stack/--no-stack",
    "stack",
    default=True,
    help="Stack structurally equivalent nodes sharing class, parents, and children (default: enabled).",
)
def dag(
    experiment_id: str | None,
    experiment_opt: str | None,
    run_id: str | None,
    workspace: str | None,
    web: bool | None,
    tree: bool,
    plot: bool,
    port: int,
    no_open: bool,
    save_as: Path | None,
    html: Path | None,
    as_json: bool,
    json_file: Path | None,
    tasks_only: bool,
    list_configs: bool,
    single_config: bool,
    config_index: int | None,
    filter_tags: str | None,
    include_actions: bool,
    layout: str,
    stack: bool,
) -> None:
    """Visualize and export experiment dependency DAGs.

    \b
    Examples:
      experimaestro dag                              # View most recent experiment in browser (default)
      experimaestro dag my_experiment                # Open web viewer for my_experiment
      experimaestro dag my_experiment --tree         # Print ASCII tree to console
      experimaestro dag my_experiment --save-as dag.html # Export standalone HTML
      experimaestro dag my_experiment --save-as dag.json # Export standardized JSON
      experimaestro dag my_experiment --save-as dag.png  # Export plot image
      experimaestro dag my_experiment --list-configs # List hyperparameter trials
      experimaestro dag my_experiment -c 0           # View trial #0 in browser
    """
    exp = experiment_id or experiment_opt
    json_out = str(json_file) if json_file else ("" if as_json else None)

    exit_code = run_dag(
        experiment_id=exp,
        run_id=run_id,
        workspace=workspace,
        web=web,
        port=port,
        no_open=no_open,
        tree=tree,
        plot=plot,
        save_as=save_as,
        html=html,
        json_out=json_out,
        tasks_only=tasks_only,
        list_configs=list_configs,
        single_config=single_config,
        config_index=config_index,
        filter_tags=filter_tags,
        include_actions=include_actions,
        layout=layout,
        stack=stack,
    )
    if exit_code != 0:
        raise click.exceptions.Exit(exit_code)
