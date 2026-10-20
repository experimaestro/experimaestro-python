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
import types
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
    is_init_task_node,
    load_dag_from_json,
    load_run_graph,
    load_run_status,
    parse_filter_arg,
    resolve_run_dir,
    resolve_target_config,
    stack_dag_nodes,
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
    "is_init_task_node",
    "load_dag_from_json",
    "load_run_graph",
    "load_run_status",
    "parse_filter_arg",
    "plot_dag",
    "print_robust_tree",
    "resolve_run_dir",
    "resolve_target_config",
    "run_dag",
    "serve_dag_web",
    "stack_dag_nodes",
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
    stack_members: dict[str, list[str]] | None = None,
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
        job_type = (
            type(config).__name__.replace(".XPMConfig", "") if config else "Unknown"
        )
        depth = depths.get(jid, 0)
        y_coord = pos.get(jid, (0, 0))[1]
        is_init = is_init_task_node(config)
        members = (stack_members or {}).get(jid, [jid])
        stack_suffix = f" x {len(members)}" if len(members) > 1 else ""
        init_badge = " [INIT]" if is_init else ""

        # Use tags from jobs.jsonl, fallback to tags(config)
        jtags = job_tags_dict.get(jid)
        if not jtags and config:
            try:
                jtags = tags(config)
            except Exception:
                jtags = {}
        jtags = jtags or {}
        tag_str = ", ".join(f"{k}={v}" for k, v in jtags.items())
        label = (
            f"{job_type}{init_badge}{stack_suffix} [{jid[:8]}] "
            f"(depth {depth}, y {y_coord:.1f})"
        )
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
                stack_members=stack_members,
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
    stack_members: dict[str, list[str]] | None = None,
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
        stack_members: Optional mapping of representative job ID to stacked member IDs.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        logging.error(
            "matplotlib is required for graphical plotting. "
            "Please install it using: pip install matplotlib"
        )
        return

    from experimaestro import Action, Task, tags

    node_info = {}
    for jid, config in full_jobs.items():
        name = type(config).__name__.replace(".XPMConfig", "")
        ntype = (
            "action"
            if isinstance(config, Action)
            else "task"
            if isinstance(config, Task)
            else "init"
            if is_init_task_node(config)
            else "config"
        )

        # Get tags and create label
        jtags = job_tags_dict.get(jid)
        if not jtags and config:
            try:
                jtags = tags(config)
            except Exception:
                jtags = {}
        jtags = jtags or {}
        tag_str = ", ".join(f"{k}={v}" for k, v in jtags.items())
        members = (stack_members or {}).get(jid, [jid])

        node_info[jid] = {
            "name": f"{name} [INIT]" if ntype == "init" else name,
            "hash": f"[{jid[:8]}]",
            "tags": tag_str if tag_str else None,
            "type": ntype,
            "stack_count": len(members),
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

    styles = {
        "config": "#e6ffed",
        "task": "#e6f4ff",
        "init": "#f3e8ff",
        "action": "#fff1f0",
    }
    for jid, (x, y) in pos.items():
        info = node_info[jid]
        scount = info["stack_count"]
        if scount > 1:
            n_frames = 3 if scount > 5 else 2
            for layer in range(n_frames, 0, -1):
                dx = layer * 0.35
                dy = -layer * 0.08
                ax.add_patch(
                    FancyBboxPatch(
                        (x - box_w / 2 + dx, y - box_h / 2 + dy),
                        box_w,
                        box_h,
                        boxstyle="round,pad=0.03,rounding_size=0.2",
                        facecolor=styles[info["type"]],
                        edgecolor="#888888",
                        linewidth=1.4,
                        alpha=0.85,
                        zorder=1.5,
                    )
                )
            plt.text(
                x + box_w / 2 + (n_frames * 0.35) + 0.3,
                y + box_h / 2 - 0.1,
                f"x {scount}",
                ha="left",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="#334155",
                zorder=4,
            )

        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.03,rounding_size=0.2",
                facecolor=styles[info["type"]],
                edgecolor="#7c3aed" if info["type"] == "init" else "#666666",
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


def run_dag(
    *,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace: str | None = None,
    web: bool | None = None,
    port: int = 0,
    no_open: bool = False,
    tree: bool = False,
    plot: bool = False,
    save_as: Path | None = None,
    html: Path | None = None,
    json_out: str | None = None,
    tasks_only: bool = False,
    list_configs: bool = False,
    single_config: bool = False,
    config_index: int | None = None,
    filter_tags: str | None = None,
    include_actions: bool = False,
    layout: str = "dagre",
    stack: bool = True,
) -> int:
    """Execute the experiment DAG visualization/export pipeline with given options."""
    save_plot_path = None
    if save_as:
        ext = save_as.suffix.lower()
        if ext in (".html", ".htm"):
            html = save_as
            web = False
        elif ext == ".json":
            json_out = str(save_as)
            web = False
        elif ext in (".png", ".pdf", ".svg", ".jpg", ".jpeg"):
            save_plot_path = save_as
            plot = False
            web = False
        else:
            logging.error(
                "Unsupported file extension '%s' for --save-as. "
                "Supported extensions: .html, .htm, .json, .png, .pdf, .svg, .jpg",
                ext,
            )
            return 1

    # Resolve default web behavior: enabled by default unless --tree, --plot, or file export is active
    if web is None:
        web = not (
            plot or tree or save_as or html or json_out is not None or list_configs
        )

    run_info = resolve_run_dir(workspace, experiment_id, run_id)
    if run_info is None:
        return 1
    run_id, run_dir = run_info
    exp_id = run_dir.parent.name

    if list_configs:
        configs = get_experiment_configs(run_dir)
        if not configs:
            print(f"No configuration tags found for experiment '{exp_id}' ({run_id}).")  # noqa: T201
            return 0
        print(f"Available configurations in experiment '{exp_id}' (run {run_id}):")  # noqa: T201
        for idx, cfg_dict in enumerate(configs):
            cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg_dict.items()))
            print(f"  [{idx}] {cfg_str}")  # noqa: T201
        return 0

    graph_data = load_run_graph(run_dir)
    if graph_data is None:
        return 1
    full_jobs, dependencies, job_tags_dict = graph_data

    title_suffix = ""
    if single_config or config_index is not None or filter_tags:
        configs = get_experiment_configs(run_dir)
        filter_args = types.SimpleNamespace(
            config_index=config_index,
            filter=filter_tags,
            single_config=single_config,
        )
        target_tags, title_suffix = resolve_target_config(filter_args, configs)
        if target_tags is None:
            return 1
        full_jobs, dependencies = filter_subgraph_by_config(
            full_jobs,
            dependencies,
            job_tags_dict,
            target_tags,
            include_actions=include_actions,
        )

    if tasks_only:
        full_jobs, dependencies = filter_tasks_only(full_jobs, dependencies)

    workspace_path = str(run_dir.parents[2]) if len(run_dir.parents) >= 3 else ""

    # Export JSON if requested
    if json_out is not None:
        dag_json = build_dag_json(
            full_jobs,
            dependencies,
            job_tags_dict,
            run_dir=run_dir,
            experiment_id=exp_id,
            run_id=run_id,
            workspace=workspace_path,
        )
        json_str = json.dumps(dag_json, indent=2)
        if json_out:  # Target file provided
            out_path = Path(json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str)
            print(f"Saved DAG JSON to {out_path}")  # noqa: T201
        else:
            print(json_str)  # noqa: T201
        return 0

    # Export standalone HTML page if requested
    if html is not None:
        dag_json = build_dag_json(
            full_jobs,
            dependencies,
            job_tags_dict,
            run_dir=run_dir,
            experiment_id=exp_id,
            run_id=run_id,
            workspace=workspace_path,
        )
        html_str = generate_dag_html(
            dag_json,
            title=f"Experimaestro DAG - {exp_id}",
            default_layout=layout,
        )
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(html_str)
        print(f"Saved interactive DAG HTML to {html}")  # noqa: T201
        return 0

    # Optional terminal text tree visualization
    if tree:
        t_jobs, t_deps, t_tags = full_jobs, dependencies, job_tags_dict
        stack_members = None
        if stack:
            t_jobs, t_deps, t_tags, stack_members = stack_dag_nodes(
                full_jobs, dependencies, job_tags_dict
            )
        dependents: dict[str, list[str]] = {}
        for jid, deps in t_deps.items():
            for d in deps:
                dependents.setdefault(d, []).append(jid)

        depths, pos, _ = compute_dag_layout(t_jobs, t_deps, dependents)

        tree_roots = [jid for jid in t_jobs if not t_deps.get(jid)]
        print_robust_tree(
            tree_roots,
            dependents,
            t_deps,
            t_jobs,
            t_tags,
            set(),
            depths,
            pos,
            stack_members=stack_members,
        )
        if not (web or plot or save_plot_path):
            return 0

    # Launch interactive web server (default mode)
    if web:
        dag_json = build_dag_json(
            full_jobs,
            dependencies,
            job_tags_dict,
            run_dir=run_dir,
            experiment_id=exp_id,
            run_id=run_id,
            workspace=workspace_path,
        )
        serve_dag_web(
            dag_json,
            port=port,
            open_browser=not no_open,
            default_layout=layout,
        )
        return 0

    # Optional Matplotlib & graphical plot visualization (requires matplotlib)
    if plot or save_plot_path:
        p_jobs, p_deps, p_tags = full_jobs, dependencies, job_tags_dict
        stack_members = None
        if stack:
            p_jobs, p_deps, p_tags, stack_members = stack_dag_nodes(
                full_jobs, dependencies, job_tags_dict
            )
        dependents = {}
        for jid, deps in p_deps.items():
            for d in deps:
                dependents.setdefault(d, []).append(jid)

        depths, pos, reduced_dag = compute_dag_layout(
            p_jobs, p_deps, dependents
        )

        plot_dag(
            p_jobs,
            p_tags,
            pos,
            reduced_dag,
            title_suffix=title_suffix,
            show_plot=plot,
            save_path=save_plot_path,
            stack_members=stack_members,
        )
        return 0

    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Experiment identifier (optional positional).",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Workspace ID or directory path (default: auto-detected via find_workspace).",
    )
    parser.add_argument(
        "--experiment-id",
        "-e",
        default=None,
        help="Experiment identifier.",
    )
    parser.add_argument(
        "--run-id",
        "-r",
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
        "-W",
        action="store_true",
        default=None,
        help="Launch interactive DAG web viewer in browser (default).",
    )
    parser.add_argument(
        "--no-web",
        action="store_false",
        dest="web",
        help="Do not launch interactive DAG web viewer.",
    )
    parser.add_argument(
        "--tree",
        "-t",
        action="store_true",
        help="Print text tree representation of the DAG to terminal.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Display matplotlib interactive plot window (requires matplotlib).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for web server (default: random available port).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open browser when launching web viewer.",
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
        "--no-stack",
        action="store_false",
        dest="stack",
        default=True,
        help="Disable stacking of structurally equivalent nodes.",
    )
    parser.add_argument(
        "--save-as",
        type=Path,
        default=None,
        help="Path where the DAG output should be saved (auto-detects .html, .json, .png, etc.). "
        "If ending in .html, generates standalone HTML without launching the server.",
    )
    args = parser.parse_args()

    exp_id = args.experiment or args.experiment_id

    run_dag(
        experiment_id=exp_id,
        run_id=args.run_id,
        workspace=args.workspace,
        web=args.web,
        port=args.port,
        no_open=args.no_open,
        tree=args.tree,
        plot=args.plot,
        save_as=args.save_as,
        html=args.html,
        json_out=args.json,
        tasks_only=args.tasks_only,
        list_configs=args.list_configs,
        single_config=args.single_config,
        config_index=args.config_index,
        filter_tags=args.filter,
        include_actions=args.include_actions,
        layout=args.layout,
        stack=args.stack,
    )


if __name__ == "__main__":
    main()
