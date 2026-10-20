"""Core DAG utilities, graph algorithms, and standardized JSON serialization.

This module provides display-agnostic utilities for reading Experimaestro
experiment graphs from workspace records (``objects.jsonl``, ``jobs.jsonl``,
and ``status.json``), computing hierarchical forward topological layouts,
filtering subgraphs by hyperparameter configurations, and serializing the DAG
into a standardized dual-compatible JSON format.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any
import networkx as nx

from experimaestro import Action, Config, LightweightTask, Task, tags
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.settings import find_workspace, get_workspace
from experimaestro.tools.inspect_params.models import (
    ParamsGraph,
    resolve_serialized_value,
)

# Default layout sizing constants
BOX_W = 25.0
BOX_H = 1.05
MARGIN_X = 4.0  # Gap between boxes horizontally
MARGIN_Y = 0.55  # Gap between boxes vertically


def resolve_run_dir(
    workspace_arg: str | Path | None,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> tuple[str, Path] | None:
    """Find and return (run_id, run_dir) from the workspace.

    Uses the explicitly specified workspace if provided (by ID or path).
    Otherwise, auto-selects the default workspace just as in real experiments
    using Experimaestro's workspace triggers matching the experiment ID.
    """
    if workspace_arg:
        ws_id = str(workspace_arg)
        if get_workspace(ws_id) is not None:
            ws_env = find_workspace(workspace=ws_id, experiment_id=experiment_id)
        else:
            ws_path = Path(workspace_arg).expanduser()
            ws_env = find_workspace(workdir=ws_path, experiment_id=experiment_id)
    else:
        ws_env = find_workspace(experiment_id=experiment_id)

    if getattr(ws_env, "is_remote", False):
        from experimaestro.utils.remote import find_remote_workspace_experiment_run

        try:
            return find_remote_workspace_experiment_run(ws_env, experiment_id, run_id)
        except Exception as e:
            logging.error("Failed to resolve remote experiment run: %s", e)
            return None

    workspace = ws_env.path
    experiments_dir = workspace / "experiments"
    if not experiments_dir.exists():
        logging.error("No 'experiments' directory found in workspace '%s'.", workspace)
        return None

    available_xps = sorted(
        [
            p.name
            for p in experiments_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
    )
    if not experiment_id:
        if len(available_xps) == 1:
            experiment_id = available_xps[0]
            logging.info(
                "Auto-selected single experiment '%s' in workspace '%s'.",
                experiment_id,
                workspace,
            )
        elif len(available_xps) > 1:
            xps_str = ", ".join(f"'{x}'" for x in available_xps)
            logging.error(
                "No experiment specified. Please specify an experiment ID. Available experiments in workspace '%s': %s",
                workspace,
                xps_str,
            )
            return None
        else:
            logging.error("No experiments found in workspace '%s'.", workspace)
            return None

    exp_dir = experiments_dir / experiment_id
    if not exp_dir.exists():
        xps_str = (
            ", ".join(f"'{x}'" for x in available_xps) if available_xps else "(none)"
        )
        logging.error(
            "Experiment '%s' not found in workspace '%s'. Available experiments: %s",
            experiment_id,
            workspace,
            xps_str,
        )
        return None

    available_runs = sorted(
        [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    if not available_runs:
        logging.error(
            "Experiment '%s' in workspace '%s' has no recorded runs.",
            experiment_id,
            workspace,
        )
        return None

    provider = WorkspaceStateProvider(workspace)
    if run_id:
        if run_id not in available_runs:
            runs_str = ", ".join(f"'{r}'" for r in available_runs)
            logging.error(
                "Run '%s' not found for experiment '%s' in workspace '%s'. Available runs: %s",
                run_id,
                experiment_id,
                workspace,
                runs_str,
            )
            return None
        resolved_run_id = run_id
    else:
        try:
            current_run = provider.get_current_run(experiment_id)
        except Exception:
            current_run = None
        resolved_run_id = current_run or available_runs[-1]

    run_dir = exp_dir / resolved_run_id
    if not run_dir.exists():
        logging.error("Run directory does not exist: %s", run_dir)
        return None

    return resolved_run_id, run_dir


def extract_config_tags(
    config: Any, grid_keys: set[str] | None = None
) -> dict[str, Any]:
    """Extract hyperparameter tags from a Config node's xpmvalues.

    If grid_keys is provided, extracts parameters matching grid_keys.
    Otherwise, extracts all scalar primitive parameters (int, float, str, bool).
    """
    if config is None:
        return {}
    xpm = getattr(config, "__xpm__", None)
    if xpm is None or not hasattr(xpm, "xpmvalues"):
        return {}

    raw_params: dict[str, Any] = {}
    try:
        for param, val in xpm.xpmvalues():
            p_name = getattr(param, "name", str(param))
            if isinstance(val, (int, float, str, bool)) and not p_name.startswith("_"):
                raw_params[p_name] = val
    except Exception:
        return {}

    if grid_keys:
        return {k: v for k, v in raw_params.items() if k in grid_keys}
    return raw_params


class _LightweightParam:
    """Minimal parameter descriptor carrying the argument name."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _LightweightXPMInfo:
    """Lightweight metadata container matching ConfigInformation's DAG interface."""

    __slots__ = (
        "identifier",
        "task",
        "init_tasks",
        "dependencies",
        "values",
        "is_init_task",
    )

    def __init__(self, identifier: str, is_init_task: bool = False):
        self.identifier = identifier
        self.task: Any = None
        self.init_tasks: list[Any] = []
        self.dependencies: list[Any] = []
        self.values: dict[str, Any] = {}
        self.is_init_task = is_init_task

    def xpmvalues(self):
        for k, v in self.values.items():
            yield _LightweightParam(k), v

    def tags(self) -> dict[str, Any]:
        return {}


def is_init_task_node(config: Any) -> bool:
    """Return True if a node is a LightweightTask (init_task) and not a full Task or Action."""
    if config is None:
        return False
    if isinstance(config, (Task, Action)):
        return False
    if isinstance(config, LightweightTask):
        return True
    xpm = getattr(config, "__xpm__", None)
    return bool(getattr(xpm, "is_init_task", False))


def _load_lightweight_objects(
    graph: ParamsGraph,
    job_ids: set[str],
    action_ids: set[str],
) -> dict[int, Any]:
    """Construct lightweight Config, LightweightTask, Task, and Action nodes from an indexed ParamsGraph.

    Avoids importing external Python modules or executing remote file paths while
    preserving class names, isinstance(node, (Task, Action, LightweightTask, Config)) semantics,
    and __xpm__ dependency/parameter traversal.
    """
    cls_cache: dict[tuple[type, str, str], type] = {}
    objects: dict[int, Any] = {}

    for obj_id, obj_node in graph.objects.items():
        ident = obj_node.identifier
        module_name = obj_node.module or "_xpm_dag_"
        type_name = obj_node.type_name or "Unknown"

        if ident and ident in action_ids:
            base_cls: type = Action
        elif (ident and ident in job_ids) or obj_node.is_task:
            base_cls = Task
        elif obj_node.is_init_task:
            base_cls = LightweightTask
        else:
            base_cls = Config
            # Fallback to already-imported module in sys.modules if present
            loaded_mod = sys.modules.get(module_name)
            if loaded_mod is not None:
                real_cls = getattr(loaded_mod, type_name, None)
                if isinstance(real_cls, type):
                    if issubclass(real_cls, Action):
                        base_cls = Action
                    elif issubclass(real_cls, Task):
                        base_cls = Task
                    elif issubclass(real_cls, LightweightTask):
                        base_cls = LightweightTask

        cls_key = (base_cls, module_name, type_name)
        cls = cls_cache.get(cls_key)
        if cls is None:
            cls = type(type_name, (base_cls,), {"__module__": module_name})
            cls_cache[cls_key] = cls

        node = object.__new__(cls)
        object.__setattr__(
            node,
            "__xpm__",
            _LightweightXPMInfo(ident, is_init_task=(base_cls is LightweightTask)),
        )
        objects[obj_id] = node

    for obj_id, obj_node in graph.objects.items():
        node = objects[obj_id]
        xpm: _LightweightXPMInfo = node.__xpm__

        if (
            obj_node.task_id is not None
            and obj_node.task_id in objects
            and obj_node.task_id != obj_id
        ):
            xpm.task = objects[obj_node.task_id]

        if obj_node.init_tasks:
            xpm.init_tasks = [
                objects[tid] for tid in obj_node.init_tasks if tid in objects
            ]

        for fname, fval in obj_node.fields.items():
            xpm.values[fname] = resolve_serialized_value(fval, objects)

    return objects


def load_run_graph(
    run_dir: Path,
) -> tuple[dict, dict[str, list[str]], dict[str, dict]] | None:
    """Load objects.jsonl and jobs.jsonl into full_jobs, dependencies, and job_tags_dict."""
    objects_file = run_dir / "objects.jsonl"
    if not objects_file.exists():
        logging.error("Objects record not found: %s", objects_file)
        return None

    graph = ParamsGraph.from_file(objects_file, load_docstrings=False)

    job_tags_dict: dict[str, dict] = {}
    job_ids: set[str] = set()
    grid_keys: set[str] = set()
    jobs_jsonl = run_dir / "jobs.jsonl"
    if jobs_jsonl.exists():
        with jobs_jsonl.open() as f:
            for line in f:
                if line.strip():
                    job_data = json.loads(line)
                    jid = job_data.get("job_id")
                    if jid:
                        job_ids.add(jid)
                        if "tags" in job_data:
                            jtags = job_data["tags"] or {}
                            job_tags_dict[jid] = jtags
                            if jtags:
                                grid_keys.update(jtags.keys())

    status_data = load_run_status(run_dir)
    action_ids: set[str] = set(status_data.get("actions", {}).keys())

    all_configs_dict = _load_lightweight_objects(
        graph, job_ids=job_ids, action_ids=action_ids
    )

    full_jobs = {
        obj_node.identifier: all_configs_dict[obj_id]
        for obj_id, obj_node in graph.objects.items()
        if obj_node.identifier
    }

    direct_deps = {
        jid: get_all_dependencies(config, full_jobs, recursive=False)
        for jid, config in full_jobs.items()
    }
    raw_all_deps = {
        jid: set(get_all_dependencies(config, full_jobs, recursive=True))
        for jid, config in full_jobs.items()
    }
    # Remove PathGenerator back-edges on direct dependencies where a Config's
    # generated path points to the job directory of an enclosing Task/Config.
    for jid, deps in list(direct_deps.items()):
        if not isinstance(full_jobs.get(jid), (Task, Action)):
            direct_deps[jid] = [
                d for d in deps if jid not in raw_all_deps.get(d, set())
            ]

    # Expand transitive Config dependencies from cleaned direct_deps so nested
    # Configs (e.g. ValidationSettings -> AggregatorValidationListener) never
    # inherit stripped PathGenerator back-edges.
    dependencies: dict[str, list[str]] = {}
    for jid in full_jobs:
        expanded: list[str] = []
        seen_deps: set[str] = set()
        queue = list(direct_deps.get(jid, []))
        visited_cfg: set[str] = set()
        while queue:
            d = queue.pop(0)
            if d == jid:
                continue
            if d not in seen_deps:
                seen_deps.add(d)
                expanded.append(d)
            d_obj = full_jobs.get(d)
            if (
                d_obj is not None
                and not isinstance(d_obj, (Task, Action, LightweightTask))
                and not is_init_task_node(d_obj)
                and d not in visited_cfg
            ):
                visited_cfg.add(d)
                for sub_d in direct_deps.get(d, []):
                    if sub_d not in seen_deps:
                        queue.append(sub_d)
        if expanded:
            dependencies[jid] = expanded

    # Populate tags for Config nodes (e.g. CNN architecture params)
    for jid, config in full_jobs.items():
        if jid not in job_tags_dict and config:
            cfg_tags = extract_config_tags(config, grid_keys if grid_keys else None)
            if cfg_tags:
                job_tags_dict[jid] = cfg_tags

    return full_jobs, dependencies, job_tags_dict


def load_run_status(run_dir: Path) -> dict[str, Any]:
    """Load status.json if present in the run directory."""
    status_file = run_dir / "status.json"
    if not status_file.exists():
        return {}
    try:
        with status_file.open() as f:
            return json.load(f)
    except Exception as e:
        logging.warning("Could not read status.json at %s: %s", status_file, e)
        return {}


def get_all_dependencies(
    config, full_jobs: dict, *, recursive: bool = True
) -> list[str]:
    """Walk the config and its fields to find all dependencies."""
    xpm = config.__xpm__
    dep_ids = set()
    current_id = str(xpm.identifier)

    if hasattr(xpm, "task") and xpm.task:
        dep_ids.add(str(xpm.task.__xpm__.identifier))
    if hasattr(xpm, "init_tasks"):
        for init_task in xpm.init_tasks:
            if hasattr(init_task, "__xpm__"):
                dep_ids.add(str(init_task.__xpm__.identifier))

    for dep in getattr(xpm, "dependencies", []):
        if hasattr(dep, "__identifier__"):
            dep_ids.add(str(dep.__identifier__()))
        elif hasattr(dep, "__xpm__"):
            dep_ids.add(str(dep.__xpm__.identifier))
        else:
            dep_ids.add(str(dep))

    processed_configs = set()

    def walk_value(v):
        from experimaestro import Config

        if isinstance(v, Config):
            v_id = str(v.__xpm__.identifier)
            if v_id in full_jobs:
                dep_ids.add(v_id)
                if not recursive:
                    return
            if v_id not in processed_configs:
                processed_configs.add(v_id)
                for _, sub_value in v.__xpm__.xpmvalues():
                    walk_value(sub_value)
        elif isinstance(v, list):
            for item in v:
                walk_value(item)
        elif isinstance(v, dict):
            for item in v.values():
                walk_value(item)
        elif isinstance(v, (str, Path)) and "/jobs/" in str(v):
            match = re.search(r"/jobs/[^/]+/([0-9a-f]{64})", str(v))
            if match:
                v_id = match.group(1)
                if v_id in full_jobs:
                    dep_ids.add(v_id)

    for name, value in xpm.xpmvalues():
        walk_value(value)
    if current_id in dep_ids:
        dep_ids.remove(current_id)
    return list(dep_ids)


def _detect_bottleneck_configs(
    job_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect core bottleneck grid-search configurations from jobs.jsonl records.

    When an experiment contains both core model grid-search tasks (e.g. Learner
    tagged with ``{base, lr, seed}``) and dataset/evaluation fan-out tasks
    (e.g. IndexBuilder tagged with ``{index_documents}``, 1st-stage Evaluate
    tagged with ``{dataset, first_stage}``, and 2nd-stage Evaluate tagged with
    ``{base, lr, seed, dataset, first_stage, checkpoint}``), this function
    identifies the bottleneck grid-search schema so that configuration filtering
    selects one complete model run (including all its evaluation datasets) rather
    than individual dataset jobs.
    """
    tagged_records = [r for r in job_records if r.get("tags")]
    if not tagged_records:
        return []

    by_schema: dict[frozenset[str], list[dict[str, Any]]] = {}
    schema_task_ids: dict[frozenset[str], set[str]] = {}
    for r in tagged_records:
        tdict = r["tags"]
        schema = frozenset(tdict.keys())
        by_schema.setdefault(schema, []).append(tdict)
        if r.get("task_id"):
            schema_task_ids.setdefault(schema, set()).add(r["task_id"])

    schemas = list(by_schema.keys())
    if len(schemas) > 1:
        # Look for shared bottleneck schemas across distinct task stages via
        # pairwise schema intersections I = S_a ∩ S_b (which covers both strict
        # subsets S_a ⊂ S_b like Learner -> Evaluate and non-subset stage overlaps
        # like IndexBuilder{model, doc} -> Evaluate{model, dataset}).
        candidate_configs: dict[frozenset[str], list[dict[str, Any]]] = {}
        for idx_a, s_a in enumerate(schemas):
            tids_a = schema_task_ids.get(s_a, set())
            uniq_a = {tuple(sorted(d.items())) for d in by_schema[s_a]}
            for s_b in schemas[idx_a + 1 :]:
                tids_b = schema_task_ids.get(s_b, set())
                if tids_a and tids_b and tids_a == tids_b:
                    continue
                inter = s_a & s_b
                if not inter:
                    continue
                uniq_b = {tuple(sorted(d.items())) for d in by_schema[s_b]}
                proj_a = {
                    tuple(sorted((k, d[k]) for k in inter))
                    for d in by_schema[s_a]
                }
                proj_b = {
                    tuple(sorted((k, d[k]) for k in inter))
                    for d in by_schema[s_b]
                }
                shared_proj = proj_a & proj_b
                union_proj = proj_a | proj_b
                # Require shared bottleneck configurations across stages and a 1->N fan-out
                if shared_proj and max(len(uniq_a), len(uniq_b)) > len(union_proj):
                    if inter not in candidate_configs:
                        configs_for_inter: list[dict[str, Any]] = []
                        seen_inter: set[tuple] = set()
                        for d in by_schema[s_a] + by_schema[s_b]:
                            proj_dict = {k: d[k] for k in inter}
                            key = tuple(sorted(proj_dict.items()))
                            if key in shared_proj and key not in seen_inter:
                                seen_inter.add(key)
                                configs_for_inter.append(proj_dict)
                        if configs_for_inter:
                            candidate_configs[inter] = configs_for_inter

        if candidate_configs:
            # Prefer largest bottleneck schema key-set, breaking ties by strongest compression (fewest configs)
            target_schema = max(
                candidate_configs.keys(),
                key=lambda s: (len(s), -len(candidate_configs[s])),
            )
            return candidate_configs[target_schema]

    # Default: deduplicate all non-empty tag dictionaries in order
    configs = []
    seen = set()
    for r in tagged_records:
        tdict = r["tags"]
        key = tuple(sorted(tdict.items()))
        if key not in seen:
            seen.add(key)
            configs.append(tdict)
    return configs


def get_experiment_configs(run_dir: Path) -> list[dict]:
    """Retrieve unique bottleneck grid-search tag dictionaries from jobs.jsonl in a run directory."""
    jobs_jsonl = run_dir / "jobs.jsonl"
    if not jobs_jsonl.exists():
        return []

    job_records: list[dict[str, Any]] = []
    with jobs_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            job_records.append(json.loads(line))

    configs = _detect_bottleneck_configs(job_records)
    if configs:
        param_keys = sorted(set().union(*(c.keys() for c in configs)))
        logging.info(
            "Detected %d grid search configuration(s) (parameters: %s)",
            len(configs),
            ", ".join(param_keys),
        )
    return configs


def parse_filter_arg(filter_str: str) -> dict[str, str | int | float]:
    """Parse a comma-separated key=val filter string into typed values."""
    parsed: dict[str, str | int | float] = {}
    for item in filter_str.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            parsed[k] = int(v)
        except ValueError:
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = v
    return parsed


def filter_subgraph_by_config(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    job_tags_dict: dict[str, dict],
    target_tags: dict,
    include_actions: bool = False,
) -> tuple[dict, dict[str, list[str]]]:
    """Extract a subgraph for a single configuration or a subset of configurations."""
    # 1. Identify jobs whose tags match all key-values in target_tags
    matching = [
        jid
        for jid, tags_val in job_tags_dict.items()
        if jid in full_jobs
        and all(tags_val.get(k) == v for k, v in target_tags.items())
    ]
    if not matching:
        return {}, {}

    # 2. If filtering matched only upstream tasks (no matched task depends on another
    # matched task, e.g. --filter index_documents=...), traverse downstream Task
    # dependents whose tags do not conflict with target_tags.
    from experimaestro import Action

    selected_nodes = set(matching)
    matched_tasks = [
        jid for jid in matching if isinstance(full_jobs.get(jid), Task)
    ]
    matched_task_set = set(matched_tasks)
    has_multi_stage_match = any(
        any(d in matched_task_set for d in dependencies.get(jid, []))
        for jid in matched_tasks
    )
    if matched_tasks and not has_multi_stage_match:
        dependents: dict[str, list[str]] = {}
        for jid, deps in dependencies.items():
            if jid not in full_jobs:
                continue
            for d in deps:
                dependents.setdefault(d, []).append(jid)

        down_stack = list(matched_tasks)
        while down_stack:
            curr = down_stack.pop()
            for child in dependents.get(curr, []):
                if child in selected_nodes:
                    continue
                cfg = full_jobs.get(child)
                if not isinstance(cfg, Task):
                    continue
                child_tags = job_tags_dict.get(child) or {}
                if all(
                    child_tags.get(k) is None or child_tags.get(k) == v
                    for k, v in target_tags.items()
                ):
                    selected_nodes.add(child)
                    down_stack.append(child)

    # 3. Traverse upstream ancestors (CNN, Datasets, ParameterLoader, FilterUnassessedTopics)
    stack = list(selected_nodes)
    while stack:
        curr = stack.pop()
        for parent in dependencies.get(curr, []):
            if parent in full_jobs and parent not in selected_nodes:
                selected_nodes.add(parent)
                stack.append(parent)

    # 4. If requested, include downstream actions and evaluated models that depend on matching jobs
    if include_actions:
        for jid, config in full_jobs.items():
            if isinstance(config, Action) or type(config).__name__.startswith(
                "EvaluatedModel"
            ):
                deps = dependencies.get(jid, [])
                if any(m in deps for m in selected_nodes):
                    selected_nodes.add(jid)

    # 4. Construct filtered jobs and dependencies restricted to selected_nodes
    filtered_jobs = {jid: full_jobs[jid] for jid in selected_nodes}
    filtered_dependencies = {
        jid: [p for p in dependencies.get(jid, []) if p in selected_nodes]
        for jid in selected_nodes
    }
    filtered_dependencies = {k: v for k, v in filtered_dependencies.items() if v}
    return filtered_jobs, filtered_dependencies


def filter_tasks_only(
    full_jobs: dict, dependencies: dict[str, list[str]]
) -> tuple[dict, dict[str, list[str]]]:
    """Filter the graph to include only Task, Action, and LightweightTask (init_task) nodes."""
    is_task = {
        jid: isinstance(c, (Task, Action, LightweightTask)) or is_init_task_node(c)
        for jid, c in full_jobs.items()
    }
    task_deps: dict[str, list[str]] = {}
    for jid in full_jobs:
        if not is_task[jid]:
            continue
        tdeps, q, processed = set(), list(dependencies.get(jid, [])), set()
        while q:
            d = q.pop()
            if d in processed:
                continue
            processed.add(d)
            if is_task.get(d):
                tdeps.add(d)
            else:
                q.extend(dependencies.get(d, []))
        task_deps[jid] = list(tdeps)
    filtered_jobs = {jid: c for jid, c in full_jobs.items() if is_task[jid]}
    return filtered_jobs, task_deps


def stack_dag_nodes(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    job_tags_dict: dict[str, dict],
    grid_keys: set[str] | None = None,
) -> tuple[dict, dict[str, list[str]], dict[str, dict], dict[str, list[str]]]:
    """Stack structurally equivalent nodes sharing class name, grid tags, and parent/child groups.

    Uses iterative bisimulation refinement on the transitively reduced DAG so that
    parallel chains (e.g. per-dataset index -> 1st-stage evaluate -> 2nd-stage evaluate)
    collapse into stacked nodes while keeping distinct core grid-search branches separate.

    Returns:
        Tuple of ``(stacked_jobs, stacked_dependencies, stacked_tags_dict, stack_members)``.
    """
    if not full_jobs:
        return {}, {}, {}, {}

    dag = nx.DiGraph()
    for jid in full_jobs:
        dag.add_node(jid)
    for jid, deps in dependencies.items():
        if jid in dag:
            for d in deps:
                if d in dag and d != jid:
                    dag.add_edge(d, jid)

    reduced_dag = (
        nx.transitive_reduction(dag)
        if nx.is_directed_acyclic_graph(dag)
        else dag
    )

    # Infer bottleneck grid_keys if not explicitly supplied
    if grid_keys is None:
        mock_records = [
            {
                "job_id": jid,
                "task_id": type(c).__name__,
                "tags": job_tags_dict.get(jid) or {},
            }
            for jid, c in full_jobs.items()
            if isinstance(c, Task) and job_tags_dict.get(jid)
        ]
        detected = _detect_bottleneck_configs(mock_records)
        if detected:
            # Only treat as distinguishing grid_keys if there are multiple schemas
            all_keys = set().union(*(r["tags"].keys() for r in mock_records))
            det_keys = set().union(*(c.keys() for c in detected))
            if det_keys < all_keys:
                grid_keys = det_keys

    base_sig: dict[str, tuple] = {}
    for jid, config in full_jobs.items():
        cname = type(config).__name__.replace(".XPMConfig", "")
        ntype = (
            "action"
            if isinstance(config, Action)
            else "task"
            if isinstance(config, Task)
            else "init"
            if is_init_task_node(config)
            else "config"
        )
        jtags = job_tags_dict.get(jid) or {}
        if grid_keys:
            gtag_sig = tuple(
                sorted((k, str(v)) for k, v in jtags.items() if k in grid_keys)
            )
        else:
            gtag_sig = ()
        base_sig[jid] = (cname, ntype, gtag_sig)

    # Iterative Weisfeiler-Lehman refinement on (base_sig, set(parent_colors), set(child_colors))
    uniq_init = {v: i for i, v in enumerate(sorted(set(base_sig.values()), key=str))}
    color: dict[str, int] = {jid: uniq_init[base_sig[jid]] for jid in full_jobs}

    for _ in range(len(full_jobs) + 1):
        new_sig: dict[str, tuple] = {}
        for jid in full_jobs:
            p_cols = tuple(sorted({color[p] for p in reduced_dag.predecessors(jid)}))
            c_cols = tuple(sorted({color[c] for c in reduced_dag.successors(jid)}))
            new_sig[jid] = (base_sig[jid], p_cols, c_cols)
        uniq_map = {v: i for i, v in enumerate(sorted(set(new_sig.values()), key=str))}
        new_color = {jid: uniq_map[new_sig[jid]] for jid in full_jobs}
        if new_color == color:
            break
        color = new_color

    groups: dict[int, list[str]] = {}
    for jid in sorted(full_jobs.keys()):
        groups.setdefault(color[jid], []).append(jid)

    member_to_rep: dict[str, str] = {}
    stacked_jobs: dict[str, Any] = {}
    stack_members: dict[str, list[str]] = {}
    stacked_tags_dict: dict[str, dict] = {}

    for _, members in groups.items():
        rep_id = members[0]
        stacked_jobs[rep_id] = full_jobs[rep_id]
        stack_members[rep_id] = members
        for m in members:
            member_to_rep[m] = rep_id

        if len(members) == 1:
            if rep_id in job_tags_dict:
                stacked_tags_dict[rep_id] = dict(job_tags_dict[rep_id])
        else:
            all_tkeys: list[str] = []
            for m in members:
                for k in (job_tags_dict.get(m) or {}).keys():
                    if k not in all_tkeys:
                        all_tkeys.append(k)
            merged_tags: dict[str, Any] = {}
            for k in all_tkeys:
                vals = [
                    (job_tags_dict.get(m) or {}).get(k)
                    for m in members
                    if k in (job_tags_dict.get(m) or {})
                ]
                uniq_vals = {str(v) for v in vals}
                if len(uniq_vals) == 1 and len(vals) == len(members):
                    merged_tags[k] = vals[0]
                else:
                    merged_tags[k] = f"<{len(uniq_vals)} values>"
            if merged_tags:
                stacked_tags_dict[rep_id] = merged_tags

    stacked_dependencies: dict[str, list[str]] = {}
    for rep_id, members in stack_members.items():
        rep_deps: list[str] = []
        seen_deps: set[str] = {rep_id}
        for m in members:
            for d in dependencies.get(m, []):
                if d in member_to_rep:
                    d_rep = member_to_rep[d]
                    if d_rep not in seen_deps:
                        seen_deps.add(d_rep)
                        rep_deps.append(d_rep)
        if rep_deps:
            stacked_dependencies[rep_id] = rep_deps

    return stacked_jobs, stacked_dependencies, stacked_tags_dict, stack_members


def compute_dag_depths(
    dag: nx.DiGraph,
    full_jobs: dict,
    align_types: bool = True,
    aligned_class_names: set[str] | None = None,
) -> dict[str, int]:
    """Compute horizontal depths for nodes in a DAG with robust stage alignment.

    Implements a 3-pass algorithm:
    1. Base Forward Topological Leveling:
       depth(v) = max_{u in deps(v)}(depth(u) + 1), roots at 0.
    2. Grid Stage Alignment:
       Aligns tasks of each type in aligned_class_names to max depth across peers.
    3. Deep Dependency Forward Push:
       Topological push sweep ensuring depth(succ) >= depth(node) + 1 for all edges.
    """
    if aligned_class_names is None:
        aligned_class_names = {"Learn", "ParameterLoader", "Evaluate"}

    # Pass 1: Base Forward Topological Leveling
    depths: dict[str, int] = {}
    topo_order = list(nx.topological_sort(dag))
    for node in topo_order:
        preds = list(dag.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = max(depths[p] for p in preds) + 1

    # Pass 2: Grid Stage Alignment
    if align_types:
        for tname in aligned_class_names:
            nodes_t = [
                n
                for n in dag.nodes
                if n in full_jobs and type(full_jobs[n]).__name__.split(".")[0] == tname
            ]
            if nodes_t:
                max_d = max(depths[n] for n in nodes_t)
                for n in nodes_t:
                    depths[n] = max_d

    # Pass 3: Deep Dependency Forward Push
    for node in topo_order:
        for succ in dag.successors(node):
            if depths[succ] <= depths[node]:
                depths[succ] = depths[node] + 1

    return depths


def compute_dag_layout(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    dependents: dict[str, list[str]],
    align_types: bool = True,
) -> tuple[dict[str, int], dict[str, tuple[float, float]], nx.DiGraph]:
    """Compute depths and (X, Y) coordinates for the DAG to maximize straight arrows."""
    dag = nx.DiGraph()
    for jid in full_jobs:
        dag.add_node(jid)
    for u, targets in dependents.items():
        for v in targets:
            if u in dag and v in dag:
                dag.add_edge(u, v)

    depths = compute_dag_depths(dag, full_jobs, align_types=align_types)

    reduced_dag = nx.transitive_reduction(dag)
    preds: dict[str, list[str]] = {n: [] for n in full_jobs}
    for u, v in reduced_dag.edges():
        preds[v].append(u)

    pos: dict[str, tuple[float, float]] = {}
    unique_depths = sorted(set(depths.values()))
    nodes_per_type: dict[str, list[str]] = {}
    for d in unique_depths:
        nodes_at_depth = [jid for jid, depth in depths.items() if depth == d]

        # Compute desired Y (average of parents)
        desired_y: dict[str, float] = {}
        for jid in nodes_at_depth:
            p_ys = [pos[p][1] for p in preds[jid] if p in pos]
            tname = type(full_jobs[jid]).__name__.split(".")[0]
            sibling_ys = [pos[s][1] for s in nodes_per_type.get(tname, []) if s in pos]
            p_ys.extend(sibling_ys)
            desired_y[jid] = sum(p_ys) / len(p_ys) if p_ys else 0.0

        nodes_at_depth.sort(
            key=lambda j: (-desired_y[j], type(full_jobs[j]).__name__, j)
        )

        for i, jid in enumerate(nodes_at_depth):
            pos[jid] = (d * (BOX_W + MARGIN_X), -i * (BOX_H + MARGIN_Y))

        for jid in nodes_at_depth:
            tname = type(full_jobs[jid]).__name__.split(".")[0]
            nodes_per_type.setdefault(tname, []).append(jid)

    return depths, pos, reduced_dag


def resolve_target_config(
    args: argparse.Namespace,
    configs: list[dict],
) -> tuple[dict | None, str]:
    """Resolve target configuration tags and title suffix from CLI options."""
    if args.config_index is not None:
        if not (0 <= args.config_index < len(configs)):
            logging.error(
                "Config index %d out of range [0..%d]. Use --list-configs to see options.",
                args.config_index,
                len(configs) - 1,
            )
            return None, ""
        target_tags = configs[args.config_index]
        return target_tags, ", ".join(
            f"{k}={v}" for k, v in sorted(target_tags.items())
        )

    if args.filter:
        filter_tags = parse_filter_arg(args.filter)
        if args.single_config:
            for cfg_cand in configs:
                if all(cfg_cand.get(k) == v for k, v in filter_tags.items()):
                    return cfg_cand, ", ".join(
                        f"{k}={v}" for k, v in sorted(cfg_cand.items())
                    )
            logging.error("No configuration matched filter: %s", args.filter)
            return None, ""
        return filter_tags, f"Filter: {args.filter}"

    if args.single_config:
        if not configs:
            logging.error("No configurations found to filter.")
            return None, ""
        return configs[0], ", ".join(f"{k}={v}" for k, v in sorted(configs[0].items()))

    return None, ""


def build_dag_json(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    job_tags_dict: dict[str, dict],
    run_dir: Path | None = None,
    experiment_id: str = "",
    run_id: str = "",
    workspace: str = "",
) -> dict[str, Any]:
    """Serialize the experiment DAG into a standardized, dual-compatible JSON format.

    Supports both snake_case and camelCase field aliases so it can be consumed
    transparently by draw_dag web renderers, experimaestro CLI, and web UI.
    """
    from experimaestro import Task, Action

    # Read jobs metadata from jobs.jsonl and status.json if available
    job_task_ids: dict[str, str] = {}
    job_timestamps: dict[str, float] = {}
    if run_dir:
        jobs_jsonl = run_dir / "jobs.jsonl"
        if jobs_jsonl.exists():
            with jobs_jsonl.open() as f:
                for line in f:
                    if line.strip():
                        try:
                            jd = json.loads(line)
                            jid = jd.get("job_id")
                            if jid:
                                if "task_id" in jd:
                                    job_task_ids[jid] = jd["task_id"]
                                if "timestamp" in jd:
                                    job_timestamps[jid] = jd["timestamp"]
                        except json.JSONDecodeError:
                            pass

    workspace_val = str(workspace) if workspace else ""
    if not workspace_val and run_dir and len(run_dir.parents) >= 3:
        workspace_val = str(run_dir.parents[2])

    status_data = load_run_status(run_dir) if run_dir else {}
    job_states: dict[str, str] = status_data.get("job_states", {})

    configs = get_experiment_configs(run_dir) if run_dir else []
    grid_keys = set().union(*(c.keys() for c in configs)) if configs else None

    # Compute deterministic layout matching flat plot
    dependents: dict[str, list[str]] = {}
    for jid, deps in dependencies.items():
        for d in deps:
            dependents.setdefault(d, []).append(jid)

    depths, pos, reduced_dag = compute_dag_layout(full_jobs, dependencies, dependents)

    nodes: list[dict[str, Any]] = []
    for jid, config in full_jobs.items():
        name = type(config).__name__.replace(".XPMConfig", "")
        is_action = isinstance(config, Action)
        is_task = isinstance(config, Task)
        is_init = is_init_task_node(config)
        ntype = (
            "action"
            if is_action
            else "task"
            if is_task
            else "init"
            if is_init
            else "config"
        )

        # Determine task status
        if jid in job_states:
            status = job_states[jid]
        elif is_init:
            status = "init"
        elif is_task or is_action:
            status = "done"  # If in finished run but not explicitly state-tracked
        else:
            status = "config"

        # Tags
        jtags = job_tags_dict.get(jid)
        if not jtags and config:
            try:
                jtags = dict(tags(config))
            except Exception:
                jtags = {}
        if not jtags and config:
            jtags = extract_config_tags(config, grid_keys)
        deps = dependencies.get(jid, [])

        task_id = job_task_ids.get(jid, name)
        if is_task or is_action:
            if workspace_val:
                full_path = str(Path(workspace_val) / "jobs" / task_id / jid)
            else:
                full_path = f"jobs/{task_id}/{jid}"
            display_path = f".../{task_id}/{jid[:20]}..."
        else:
            full_path = ""
            display_path = ""

        coord = pos.get(jid, (0.0, 0.0))
        depth_val = depths.get(jid, 0)
        row_val = round(-coord[1] / (BOX_H + MARGIN_Y)) if (BOX_H + MARGIN_Y) > 0 else 0

        node_dict = {
            # Dual identifier keys
            "id": jid,
            "jobId": jid,
            # Dual task naming keys
            "name": name,
            "task_id": task_id,
            "taskId": task_id,
            "type": ntype,
            "status": status,
            "init": is_init,
            "tags": jtags,
            # Dual dependency keys
            "depends_on": deps,
            "dependsOn": deps,
            "path": full_path,
            "display_path": display_path,
            "locator": full_path,
            "timestamp": job_timestamps.get(jid),
            # Layout coordinates matching flat plot
            "depth": depth_val,
            "col": depth_val,
            "row": row_val,
            "x": coord[0],
            "y": coord[1],
        }
        nodes.append(node_dict)

    edges: list[dict[str, str]] = [
        {"source": u, "target": v} for u, v in reduced_dag.edges()
    ]

    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "workspace": workspace_val,
        "nodes": nodes,
        "edges": edges,
        "configs": configs,
        "grid_keys": sorted(grid_keys) if grid_keys else [],
    }


def load_dag_from_json(path_or_str: str | Path) -> dict[str, Any]:
    """Load an ExperimentGraph JSON from a file path or JSON string."""
    p = Path(path_or_str) if isinstance(path_or_str, (str, Path)) else None
    if p and p.exists():
        with p.open() as f:
            return json.load(f)
    return json.loads(str(path_or_str))
