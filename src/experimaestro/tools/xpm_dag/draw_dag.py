"""Draw the dependency DAG of an Experimaestro experiment run as a tree, plot, or web page.

Run with::

    # Interactive DAG web visualization in browser:
    uv run python -m experimaestro.tools.xpm_dag --web

    # Export standalone interactive HTML report (auto-detected, zero-server):
    uv run python -m experimaestro.tools.xpm_dag --save-as dag.html

    # Export standardized JSON graph:
    uv run python -m experimaestro.tools.xpm_dag --save-as dag.json

    # Text tree terminal output:
    uv run python -m experimaestro.tools.xpm_dag --no-plot

    # Interactive matplotlib plot:
    uv run python -m experimaestro.tools.xpm_dag

    # Save plot directly to image without GUI window:
    uv run python -m experimaestro.tools.xpm_dag --save-as dag.png

    # List all hyperparameter configurations:
    uv run python -m experimaestro.tools.xpm_dag --list-configs

    # Inspect a single configuration:
    uv run python -m experimaestro.tools.xpm_dag --single-config -c 0
    uv run python -m experimaestro.tools.xpm_dag --filter "lr=0.001,kernel_size=3"

This module provides the main CLI entrypoint and coordinates terminal ASCII,
matplotlib, and interactive web representations of the experiment DAG.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import networkx as nx

from experimaestro import tags

# Re-export core DAG utilities and models for display styles & test compatibility
from .xpm_dag import (
    BOX_H,
    BOX_W,
    MARGIN_X,
    MARGIN_Y,
    build_dag_json,
    compute_dag_depths,
    compute_dag_layout,
    extract_config_tags,
    filter_subgraph_by_config,
    filter_tasks_only,
    get_all_dependencies,
    get_experiment_configs,
    load_dag_from_json,
    load_run_graph,
    load_run_status,
    parse_filter_arg,
    resolve_run_dir,
    resolve_target_config,
)
from .xpm_web_dag import (
    generate_dag_html,
    serve_dag_web,
)

__all__ = [
    "BOX_H",
    "BOX_W",
    "MARGIN_X",
    "MARGIN_Y",
    "build_dag_json",
    "compute_dag_depths",
    "compute_dag_layout",
    "extract_config_tags",
    "filter_subgraph_by_config",
    "filter_tasks_only",
    "generate_dag_html",
    "get_all_dependencies",
    "get_experiment_configs",
    "load_dag_from_json",
    "load_run_graph",
    "load_run_status",
    "parse_filter_arg",
    "plot_dag",
    "print_robust_tree",
    "resolve_run_dir",
    "resolve_target_config",
    "serve_dag_web",
]


def print_robust_tree(
    job_ids: list[str],
    dependents: dict[str, list[str]],
    dependencies: dict[str, list[str]],
    info_jobs: dict,
    job_tags_dict: dict,
    visited: set[str],
    depths: dict[str, int],
    pos: dict[str, tuple[float, float]],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    """Recursively print the dependency tree with individual nodes, depth and Y info."""
    if not job_ids:
        return

    def job_sort_key(jid: str):
        config = info_jobs.get(jid)
        return (
            depths.get(jid, 0),
            -pos.get(jid, (0, 0))[1],
            type(config).__name__ if config else "",
            jid,
        )

    sorted_job_ids = sorted(job_ids, key=job_sort_key)

    for idx, jid in enumerate(sorted_job_ids):
        is_last_node = is_last and (idx == len(sorted_job_ids) - 1)
        config = info_jobs.get(jid)
        job_type = type(config).__name__.replace(".XPMConfig", "") if config else "Unknown"
        depth = depths.get(jid, 0)
        y_coord = pos.get(jid, (0, 0))[1]

        # Use tags from jobs.jsonl, fallback to tags(config)
        jtags = job_tags_dict.get(jid) or (tags(config) if config else {})
        tag_str = ", ".join(f"{k}={v}" for k, v in jtags.items())
        label = f"{job_type} [{jid[:8]}] (depth {depth}, y {y_coord:.1f})"
        if tag_str:
            label += f" ({tag_str})"

        already_visited = jid in visited
        if already_visited:
            label += " (already shown)"

        connector = "└── " if is_last_node else "├── "
        print(f"{prefix}{connector}{label}")  # noqa: T201

        if already_visited:
            continue

        visited.add(jid)
        child_prefix = prefix + ("    " if is_last_node else "│   ")
        next_job_ids = sorted(dependents.get(jid, []), key=job_sort_key)
        if next_job_ids:
            print_robust_tree(
                next_job_ids,
                dependents,
                dependencies,
                info_jobs,
                job_tags_dict,
                visited,
                depths,
                pos,
                child_prefix,
                True,
            )


def plot_dag(
    full_jobs: dict,
    job_tags_dict: dict,
    pos: dict[str, tuple[float, float]],
    reduced_dag: nx.DiGraph,
    box_w: float = BOX_W,
    box_h: float = BOX_H,
    title_suffix: str = "",
    show_plot: bool = True,
    save_path: Path | None = None,
) -> None:
    """Plot the DAG hierarchically using pre-computed layout and reduced graph.

    Args:
        full_jobs: Mapping of job ID to config/task instance.
        job_tags_dict: Mapping of job ID to tags dictionary.
        pos: Node coordinates mapping job ID to (x, y).
        reduced_dag: Transitive reduction of the DAG.
        box_w: Width of node boxes.
        box_h: Height of node boxes.
        title_suffix: Optional suffix appended to figure title (e.g. config details).
        show_plot: Whether to display the plot interactively.
        save_path: Optional path to save the figure image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    from experimaestro import Task, Action, tags

    node_info = {}
    for jid, config in full_jobs.items():
        name = type(config).__name__.replace(".XPMConfig", "")
        ntype = (
            "action"
            if isinstance(config, Action)
            else "task" if isinstance(config, Task) else "config"
        )

        # Get tags and create label
        jtags = job_tags_dict.get(jid) or (tags(config) if config else {})
        tag_str = ", ".join(f"{k}={v}" for k, v in jtags.items())

        node_info[jid] = {
            "name": name,
            "hash": f"[{jid[:8]}]",
            "tags": tag_str if tag_str else None,
            "type": ntype,
        }

    # Calculate dynamic figure size based on the layout
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale factor: 1 unit = 0.8 inches
    scale = 0.8
    fig_w = max(10, (max_x - min_x + box_w * 2) * scale)
    fig_h = max(6, (max_y - min_y + box_h * 2) * scale)

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    title = "Experimaestro DAG (Hierarchical Top-Aligned Flow)"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)

    for u, v in reduced_dag.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        plt.annotate(
            "",
            xy=(x2 - box_w / 2, y2),
            xytext=(x1 + box_w / 2, y1),
            zorder=1,
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                alpha=1.0,
                shrinkA=0,
                shrinkB=0,
                linewidth=1.5,
            ),
        )

    styles = {"config": "#e6ffed", "task": "#e6f4ff", "action": "#fff1f0"}
    for jid, (x, y) in pos.items():
        info = node_info[jid]
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.03,rounding_size=0.2",
                facecolor=styles[info["type"]],
                edgecolor="#666666",
                linewidth=2.0,
                zorder=2,
            )
        )
        if info["tags"]:
            plt.text(
                x,
                y + 0.25,
                info["name"],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#111111",
                zorder=3,
            )
            plt.text(
                x,
                y + 0.03,
                info["hash"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
                zorder=3,
            )
            plt.text(
                x,
                y - 0.23,
                info["tags"],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
                zorder=3,
            )
        else:
            plt.text(
                x,
                y + 0.14,
                info["name"],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#111111",
                zorder=3,
            )
            plt.text(
                x,
                y - 0.14,
                info["hash"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
                zorder=3,
            )

    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=k.capitalize(),
                markerfacecolor=v,
                markersize=10,
                markeredgecolor="gray",
            )
            for k, v in styles.items()
        ],
        loc="upper left",
    )

    plt.xlim(min_x - box_w, max_x + box_w)
    plt.ylim(min_y - box_h, max_y + box_h)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved DAG plot to {save_path}")  # noqa: T201
    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace ID or directory path (default: auto-detected via find_workspace).",
    )
    parser.add_argument(
        "--experiment-id",
        default="MNIST_train_advanced",
        help="Experiment identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run timestamp ID (default: most recent run).",
    )
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Only display Task and Action nodes, skipping static config nodes.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch interactive DAG web viewer and open in browser.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Path where the standalone interactive DAG HTML page should be saved (e.g. dag.html).",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="",
        default=None,
        help="Export DAG as JSON to stdout or file (compatible with Experimaestro web CLI).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Skip the matplotlib interactive plot and only output the text tree.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all unique configuration parameter sets found in the experiment and exit.",
    )
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Render the DAG for a single grid configuration (defaults to config index 0).",
    )
    parser.add_argument(
        "-c",
        "--config-index",
        type=int,
        default=None,
        help="Index of the configuration to display (see --list-configs). Implies --single-config.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter the DAG by tag values (e.g. 'lr=0.001,kernel_size=3').",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include downstream Action and EvaluatedModel nodes when filtering by configuration.",
    )
    parser.add_argument(
        "--layout",
        choices=["dagre", "aligned"],
        default="dagre",
        help="Layout algorithm for interactive web view: 'dagre' (default) or 'aligned' (stage-aligned flow).",
    )
    parser.add_argument(
        "--save-as",
        type=Path,
        default=None,
        help="Path where the DAG output should be saved (auto-detects .html, .json, .png, etc.). "
        "If ending in .html, generates standalone HTML without launching the server.",
    )
    args = parser.parse_args()

    save_plot_path = None
    if args.save_as:
        ext = args.save_as.suffix.lower()
        if ext in (".html", ".htm"):
            args.html = args.save_as
            args.web = False
        elif ext == ".json":
            args.json = str(args.save_as)
            args.web = False
        elif ext in (".png", ".pdf", ".svg", ".jpg", ".jpeg"):
            save_plot_path = args.save_as
            args.plot = False
            args.web = False
        else:
            logging.error(
                "Unsupported file extension '%s' for --save-as. "
                "Supported extensions: .html, .htm, .json, .png, .pdf, .svg, .jpg",
                ext,
            )
            return

    run_info = resolve_run_dir(args.workspace, args.experiment_id, args.run_id)
    if run_info is None:
        return
    run_id, run_dir = run_info

    if args.list_configs:
        configs = get_experiment_configs(run_dir)
        if not configs:
            print(f"No configuration tags found for experiment '{args.experiment_id}' ({run_id}).")  # noqa: T201
            return
        print(f"Available configurations in experiment '{args.experiment_id}' (run {run_id}):")  # noqa: T201
        for idx, cfg_dict in enumerate(configs):
            cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg_dict.items()))
            print(f"  [{idx}] {cfg_str}")  # noqa: T201
        return

    graph_data = load_run_graph(run_dir)
    if graph_data is None:
        return
    full_jobs, dependencies, job_tags_dict = graph_data

    title_suffix = ""
    if args.single_config or args.config_index is not None or args.filter:
        configs = get_experiment_configs(run_dir)
        target_tags, title_suffix = resolve_target_config(args, configs)
        if target_tags is None:
            return
        full_jobs, dependencies = filter_subgraph_by_config(
            full_jobs,
            dependencies,
            job_tags_dict,
            target_tags,
            include_actions=args.include_actions,
        )

    if args.tasks_only:
        full_jobs, dependencies = filter_tasks_only(full_jobs, dependencies)

    workspace_path = str(run_dir.parents[2]) if len(run_dir.parents) >= 3 else ""

    # Export JSON if requested
    if args.json is not None:
        dag_json = build_dag_json(
            full_jobs,
            dependencies,
            job_tags_dict,
            run_dir=run_dir,
            experiment_id=args.experiment_id,
            run_id=run_id,
            workspace=workspace_path,
        )
        json_str = json.dumps(dag_json, indent=2)
        if args.json:  # Target file provided
            out_path = Path(args.json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str)
            print(f"Saved DAG JSON to {out_path}")  # noqa: T201
        else:
            print(json_str)  # noqa: T201
        return

    # Export standalone HTML page if requested
    if args.html is not None:
        dag_json = build_dag_json(
            full_jobs,
            dependencies,
            job_tags_dict,
            run_dir=run_dir,
            experiment_id=args.experiment_id,
            run_id=run_id,
            workspace=workspace_path,
        )
        html_str = generate_dag_html(
            dag_json,
            title=f"Experimaestro DAG - {args.experiment_id}",
            default_layout=args.layout,
        )
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html_str)
        print(f"Saved interactive DAG HTML to {args.html}")  # noqa: T201
        return

    # Launch interactive web server if requested
    if args.web:
        dag_json = build_dag_json(
            full_jobs,
            dependencies,
            job_tags_dict,
            run_dir=run_dir,
            experiment_id=args.experiment_id,
            run_id=run_id,
            workspace=workspace_path,
        )
        serve_dag_web(dag_json, default_layout=args.layout)
        return

    # Default / Matplotlib & Text Tree visualization
    dependents: dict[str, list[str]] = {}
    for jid, deps in dependencies.items():
        for d in deps:
            dependents.setdefault(d, []).append(jid)

    depths, pos, reduced_dag = compute_dag_layout(full_jobs, dependencies, dependents)

    tree_roots = [jid for jid in full_jobs if not dependencies.get(jid)]
    print_robust_tree(
        tree_roots,
        dependents,
        dependencies,
        full_jobs,
        job_tags_dict,
        set(),
        depths,
        pos,
    )

    if args.plot or save_plot_path:
        plot_dag(
            full_jobs,
            job_tags_dict,
            pos,
            reduced_dag,
            title_suffix=title_suffix,
            show_plot=args.plot,
            save_path=save_plot_path,
        )


if __name__ == "__main__":
    main()
