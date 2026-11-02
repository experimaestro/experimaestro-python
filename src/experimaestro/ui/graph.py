"""Dependency graph of an experiment run

The graph holds the tasks, actions and configurations of a run, built from its
serialized objects (``objects.jsonl``) without importing the experiment code. It is what the web UI graph view, the ``experimaestro experiments dag``
command and its standalone HTML export display.

A :class:`GraphView` is what is actually drawn: the graph restricted to a grid
search configuration and/or to jobs, with structurally equivalent nodes (e.g.
the same evaluation run on many datasets) stacked into one, and redundant
(transitive) edges removed. Views are computed here and only here: the web UI
asks for them, and the HTML export embeds all of them.
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from experimaestro.scheduler.interfaces import ExperimentJobInformation
from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.ui.graph")

#: Matches a job directory in a path: ``.../jobs/<task id>/<job identifier>``
JOB_PATH_RE = re.compile(r"/jobs/[^/]+/([0-9a-f]{64})")


class NodeKind(StrEnum):
    TASK = "task"
    ACTION = "action"
    #: Lightweight task run before a task (``init_tasks``)
    INIT = "init"
    CONFIG = "config"

    @property
    def is_job(self) -> bool:
        """Whether the node is kept when only showing jobs"""
        return self is not NodeKind.CONFIG


@dataclass
class GraphNode:
    id: str
    """Identifier (the job identifier for tasks)"""

    name: str
    """Class name"""

    kind: NodeKind

    task_id: str | None = None
    """Task identifier, for jobs"""

    tags: dict[str, Any] = field(default_factory=dict)

    depends_on: list[str] = field(default_factory=list)
    """Identifiers of the nodes this one directly depends on"""

    object_id: int | None = None
    """Object ID in the serialized objects pool"""

    def state_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": str(self.kind),
            "taskId": self.task_id,
            "tags": self.tags,
            "dependsOn": self.depends_on,
            "objectId": self.object_id,
        }


@dataclass
class ViewNode:
    members: list[str]
    """Identifiers of the stacked nodes, the first one standing for the stack"""

    tags: dict[str, Any] | None = None
    """Tags merged over the members (only for stacks)"""

    @property
    def id(self) -> str:
        return self.members[0]


@dataclass
class GraphView:
    nodes: list[ViewNode]
    edges: list[tuple[str, str]]
    """(source, target) pairs between stack identifiers, without redundant edges"""

    def state_dict(self, index: dict[str, int]) -> dict[str, Any]:
        """Serialize, referring to nodes by their position in the graph"""

        def node(n: ViewNode) -> dict[str, Any]:
            d: dict[str, Any] = {"members": [index[m] for m in n.members]}
            if n.tags is not None:
                d["tags"] = n.tags
            return d

        return {
            "nodes": [node(n) for n in self.nodes],
            "edges": [[index[s], index[t]] for s, t in self.edges],
        }


def view_key(config: int | None, tasks_only: bool, stack: bool) -> str:
    """Key of a view in :meth:`ExperimentGraph.views` (mirrored in the web app)"""
    return f"{'all' if config is None else config}:{int(tasks_only)}:{int(stack)}"


@dataclass
class ExperimentGraph:
    experiment_id: str
    run_id: str
    nodes: dict[str, GraphNode]

    configs: list[dict[str, Any]] = field(default_factory=list)
    """Grid search configurations (tag values), see :func:`detect_grid_configs`"""

    objects: dict[int, dict[str, Any]] = field(default_factory=dict)
    """Serialized object definitions indexed by numeric ID"""

    @property
    def grid_keys(self) -> set[str]:
        return {k for c in self.configs for k in c}

    def state_dict(self) -> dict[str, Any]:
        return {
            "experimentId": self.experiment_id,
            "runId": self.run_id,
            "nodes": [n.state_dict() for n in self.nodes.values()],
            "configs": self.configs,
            "objects": self.objects,
        }

    def index(self) -> dict[str, int]:
        """Position of each node in the serialized node list"""
        return {node_id: ix for ix, node_id in enumerate(self.nodes)}

    # --- Views

    def view(
        self,
        config: int | dict[str, Any] | None = None,
        *,
        tasks_only: bool = False,
        stack: bool = True,
    ) -> GraphView:
        """Compute what is drawn

        :param config: Grid search configuration to restrict to, given by its
            index in :attr:`configs` or by tag values (None for the whole graph)
        :param tasks_only: Only show jobs (configurations are skipped, their
            dependencies being followed through)
        :param stack: Stack structurally equivalent nodes
        """
        deps = self.job_dependencies() if tasks_only else self.dependencies()
        if tasks_only:
            deps = {n: d for n, d in deps.items() if self.nodes[n].kind.is_job}

        if config is not None:
            target = self.configs[config] if isinstance(config, int) else config
            selected = self._select_config(target)
            deps = {n: [d for d in deps[n] if d in selected] for n in deps}
            deps = {n: d for n, d in deps.items() if n in selected}

        edges = transitive_reduction(deps)

        if not stack:
            return GraphView([ViewNode([n]) for n in deps], edges)
        return self._stack(list(deps), edges)

    def stages(self, view: GraphView) -> list[list[ViewNode]]:
        """View nodes by stage: a node comes after all its dependencies"""
        parents: dict[str, list[str]] = {n.id: [] for n in view.nodes}
        for s, t in view.edges:
            parents[t].append(s)
        order = _topological_order(parents) or list(parents)
        depth: dict[str, int] = {}
        for n in order:
            depth[n] = max((depth.get(p, 0) + 1 for p in parents[n]), default=0)
        stages: list[list[ViewNode]] = [
            [] for _ in range(max(depth.values(), default=-1) + 1)
        ]
        for vn in view.nodes:
            stages[depth[vn.id]].append(vn)
        return stages

    def views(self) -> dict[str, GraphView]:
        """All the views, keyed by :func:`view_key`"""
        configs: list[int | None] = [None, *range(len(self.configs))]
        return {
            view_key(config, tasks_only, stack): self.view(
                config, tasks_only=tasks_only, stack=stack
            )
            for config in configs
            for tasks_only in (False, True)
            for stack in (False, True)
        }

    def dependencies(self) -> dict[str, list[str]]:
        return {
            n.id: [d for d in n.depends_on if d in self.nodes]
            for n in self.nodes.values()
        }

    def job_dependencies(self) -> dict[str, list[str]]:
        """Dependencies of each node, following through configurations

        A job using a configuration produced by a task (e.g. a trained model)
        then depends on that task.
        """
        result: dict[str, list[str]] = {}
        for node in self.nodes.values():
            found: dict[str, None] = {}
            seen: set[str] = set()
            queue = list(node.depends_on)
            while queue:
                dep_id = queue.pop(0)
                dep = self.nodes.get(dep_id)
                if dep is None or dep_id in seen or dep_id == node.id:
                    continue
                seen.add(dep_id)
                if dep.kind.is_job:
                    found[dep_id] = None
                else:
                    queue.extend(dep.depends_on)
            result[node.id] = list(found)
        return result

    def _select_config(self, target: dict[str, Any]) -> set[str]:
        """Nodes of a grid search configuration

        These are the nodes whose tags match the configuration, with their
        ancestors. When the configuration only matches one stage of jobs (no
        matching job depends on another), the downstream jobs whose tags do not
        contradict it are included as well.
        """

        def matches(node: GraphNode, *, partial: bool = False) -> bool:
            return all(
                node.tags.get(k) == v or (partial and node.tags.get(k) is None)
                for k, v in target.items()
            )

        job_deps = self.job_dependencies()
        selected = {n.id for n in self.nodes.values() if n.tags and matches(n)}
        matched_jobs = {n for n in selected if self.nodes[n].kind == NodeKind.TASK}

        if matched_jobs and not any(
            d in matched_jobs for n in matched_jobs for d in job_deps[n]
        ):
            dependents: dict[str, list[str]] = {}
            for n, deps in job_deps.items():
                for d in deps:
                    dependents.setdefault(d, []).append(n)
            stack = list(matched_jobs)
            while stack:
                for child in dependents.get(stack.pop(), []):
                    node = self.nodes[child]
                    if (
                        child not in selected
                        and node.kind == NodeKind.TASK
                        and matches(node, partial=True)
                    ):
                        selected.add(child)
                        stack.append(child)

        stack = list(selected)
        while stack:
            for parent in self.nodes[stack.pop()].depends_on:
                if parent in self.nodes and parent not in selected:
                    selected.add(parent)
                    stack.append(parent)
        return selected

    def _stack(self, node_ids: list[str], edges: list[tuple[str, str]]) -> GraphView:
        """Stack nodes that cannot be told apart

        Nodes are stacked when they have the same class, kind and grid search
        tags, and their parents and children are (recursively) stacked
        together — a bisimulation computed by colour refinement.
        """
        grid_keys = self.grid_keys
        parents: dict[str, list[str]] = {n: [] for n in node_ids}
        children: dict[str, list[str]] = {n: [] for n in node_ids}
        for s, t in edges:
            children[s].append(t)
            parents[t].append(s)

        def signature(node: GraphNode) -> tuple:
            grid_tags = sorted(
                (k, str(v)) for k, v in node.tags.items() if k in grid_keys
            )
            return (node.name, str(node.kind), tuple(grid_tags))

        color = _renumber({n: signature(self.nodes[n]) for n in node_ids})
        while True:
            new_color = _renumber(
                {
                    n: (
                        color[n],
                        tuple(sorted({color[p] for p in parents[n]})),
                        tuple(sorted({color[c] for c in children[n]})),
                    )
                    for n in node_ids
                }
            )
            # Refinement only splits classes: stop when the count is stable
            if len(set(new_color.values())) == len(set(color.values())):
                break
            color = new_color

        groups: dict[int, list[str]] = {}
        for n in sorted(node_ids):
            groups.setdefault(color[n], []).append(n)

        rep = {m: members[0] for members in groups.values() for m in members}
        nodes = [
            ViewNode(
                members,
                _merge_tags([self.nodes[m].tags for m in members])
                if len(members) > 1
                else None,
            )
            for members in groups.values()
        ]
        stacked = {
            n.id: sorted({rep[p] for m in n.members for p in parents[m]} - {n.id})
            for n in nodes
        }
        return GraphView(nodes, transitive_reduction(stacked))


def _renumber(signatures: dict[str, Any]) -> dict[str, int]:
    """Map each distinct signature to a small integer"""
    numbers = {s: ix for ix, s in enumerate(sorted(set(signatures.values()), key=str))}
    return {n: numbers[s] for n, s in signatures.items()}


def _merge_tags(all_tags: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the tags of stacked nodes, summarizing those that differ"""
    merged: dict[str, Any] = {}
    for key in dict.fromkeys(k for tags in all_tags for k in tags):
        values = [tags[key] for tags in all_tags if key in tags]
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in values}
        if len(distinct) == 1 and len(values) == len(all_tags):
            merged[key] = values[0]
        else:
            merged[key] = f"<{len(distinct)} values>"
    return merged


def transitive_reduction(deps: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Edges (dependency, dependent) without those implied by others

    Dependencies outside ``deps`` are ignored. If the graph has a cycle, all
    the edges are returned.
    """
    edges = [
        (d, n)
        for n, ds in deps.items()
        for d in dict.fromkeys(ds)
        if d in deps and d != n
    ]
    order = _topological_order(deps)
    if order is None:
        return edges

    # Ancestors as bitsets, in topological order
    bit = {n: 1 << ix for ix, n in enumerate(order)}
    ancestors: dict[str, int] = {}
    for n in order:
        mask = 0
        for d in deps[n]:
            if d in deps and d != n:
                mask |= ancestors[d] | bit[d]
        ancestors[n] = mask

    def redundant(d: str, n: str) -> bool:
        return any(
            ancestors[other] & bit[d]
            for other in deps[n]
            if other != d and other in deps and other != n
        )

    return [(d, n) for d, n in edges if not redundant(d, n)]


def _topological_order(deps: dict[str, list[str]]) -> list[str] | None:
    """Dependencies first, or None if there is a cycle"""
    remaining = {
        n: len({d for d in ds if d in deps and d != n}) for n, ds in deps.items()
    }
    dependents: dict[str, list[str]] = {}
    for n, ds in deps.items():
        for d in {d for d in ds if d in deps and d != n}:
            dependents.setdefault(d, []).append(n)

    order = [n for n, count in remaining.items() if count == 0]
    for n in order:
        for child in dependents.get(n, []):
            remaining[child] -= 1
            if remaining[child] == 0:
                order.append(child)
    return order if len(order) == len(deps) else None


# --- Grid search configurations


def detect_grid_configs(job_tags: Iterable[tuple[str, dict[str, Any]]]) -> list[dict]:
    """Grid search configurations of an experiment, from its job tags

    :param job_tags: (task identifier, tags) of each job

    When stages are tagged differently (e.g. a learner tagged with
    ``{lr, seed}`` and its evaluations with ``{lr, seed, dataset}``), the
    configurations are the values of the tags shared between stages with a 1→N
    fan-out, so that one configuration selects one model with all its
    evaluations. Otherwise, they are the distinct tag values.
    """
    by_schema: dict[frozenset[str], list[dict[str, Any]]] = {}
    schema_tasks: dict[frozenset[str], set[str]] = {}
    for task_id, tags in job_tags:
        if not tags:
            continue
        schema = frozenset(tags)
        by_schema.setdefault(schema, []).append(tags)
        schema_tasks.setdefault(schema, set()).add(task_id)

    def key(tags: dict[str, Any], keys: Iterable[str]) -> tuple:
        return tuple(sorted((k, json.dumps(tags[k], default=str)) for k in keys))

    candidates: dict[frozenset[str], list[dict[str, Any]]] = {}
    schemas = list(by_schema)
    for ix, schema_a in enumerate(schemas):
        for schema_b in schemas[ix + 1 :]:
            shared = schema_a & schema_b
            if not shared or schema_tasks[schema_a] == schema_tasks[schema_b]:
                continue
            proj_a = {key(t, shared) for t in by_schema[schema_a]}
            proj_b = {key(t, shared) for t in by_schema[schema_b]}
            distinct_a = {key(t, schema_a) for t in by_schema[schema_a]}
            distinct_b = {key(t, schema_b) for t in by_schema[schema_b]}
            common = proj_a & proj_b
            # Shared values with a fan-out between the two stages
            if not common or max(len(distinct_a), len(distinct_b)) <= len(
                proj_a | proj_b
            ):
                continue
            if shared not in candidates:
                configs: dict[tuple, dict[str, Any]] = {}
                for tags in by_schema[schema_a] + by_schema[schema_b]:
                    k = key(tags, shared)
                    if k in common:
                        configs.setdefault(k, {s: tags[s] for s in shared})
                candidates[shared] = list(configs.values())

    if candidates:
        # Most keys first, then the fewest configurations
        best = max(candidates, key=lambda s: (len(s), -len(candidates[s])))
        return candidates[best]

    configs = {}
    for tags_list in by_schema.values():
        for tags in tags_list:
            configs.setdefault(key(tags, tags), tags)
    return list(configs.values())


# --- Loading


def load_experiment_graph(
    provider: StateProvider, experiment_id: str, run_id: str | None = None
) -> ExperimentGraph:
    """Read the graph of an experiment run through a state provider

    :raises FileNotFoundError: if the run has no ``objects.jsonl``
    """
    if run_id is None:
        run_id = provider.get_current_run(experiment_id)
    objects = provider.get_run_objects(experiment_id, run_id)
    jobs = provider.get_experiment_job_info(experiment_id, run_id)
    action_ids: set[str] = set()
    for run in provider.get_experiment_runs(experiment_id):
        if run.run_id == run_id:
            action_ids = set(run.actions)
    return build_experiment_graph(
        experiment_id, run_id or "", objects, jobs, action_ids
    )


def load_serialized_graph(
    source: Path | str | list[dict[str, Any]] | dict[str, Any],
    experiment_id: str | None = None,
    run_id: str = "serialized",
    *,
    promote_sink_configs: bool = True,
) -> ExperimentGraph:
    """Build an ExperimentGraph from a serialized object file, directory, or data payload.

    Supports:
    - Path to ``experimaestro.json`` (e.g. Hugging Face exported model)
    - Path to ``params.json`` (job parameters)
    - Path to ``objects.jsonl`` (streaming records)
    - Directory containing any of the above files
    - Raw deserialized payload (``list[dict]`` or ``dict`` with ``"objects"``)

    :param source: File path, directory path, or deserialized dictionary/list
    :param experiment_id: Name/ID for the graph (defaults to directory or file name)
    :param run_id: Run identifier (defaults to "serialized")
    :param promote_sink_configs: Whether to promote terminal deliverable configs
        (e.g. exported ModuleLoader / model config) to Action nodes so they
        are displayed as the final pipeline stage in task-only views.
    """
    raw_objs: list[dict[str, Any]]
    inferred_name = experiment_id

    if isinstance(source, (str, Path)):
        p = Path(source).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Serialized graph source not found: {source}")

        if inferred_name is None:
            s_str = str(source)
            if m := re.search(r"models--([^/]+)--([^/]+)", s_str):
                inferred_name = f"{m.group(1)}/{m.group(2)}"
            elif p.is_dir():
                inferred_name = p.name
            else:
                parent_name = p.parent.name
                if parent_name in ("experimaestro.json", "params.json", "objects.jsonl") or p.name in ("experimaestro.json", "params.json", "objects.jsonl"):
                    inferred_name = parent_name
                else:
                    inferred_name = p.stem

        if p.is_dir():
            cand = next(
                (
                    p / name
                    for name in ("experimaestro.json", "params.json", "objects.jsonl")
                    if (p / name).is_file()
                ),
                None,
            )
            if cand is None:
                raise FileNotFoundError(
                    f"No experimaestro.json, params.json, or objects.jsonl found in directory: {p}"
                )
            target_file = cand
        else:
            target_file = p

        text = target_file.read_text(encoding="utf-8")
        if target_file.suffix == ".jsonl" or target_file.name == "objects.jsonl":
            raw_objs = [
                json.loads(line) for line in text.splitlines() if line.strip()
            ]
        else:
            payload = json.loads(text)
            if isinstance(payload, list):
                raw_objs = payload
            elif isinstance(payload, dict):
                raw_objs = payload.get("objects", [payload] if "id" in payload else [])
            else:
                raise ValueError(
                    f"Unexpected JSON format in {target_file}: expected list or dict"
                )
    elif isinstance(source, list):
        raw_objs = source
        if inferred_name is None:
            inferred_name = "serialized"
    elif isinstance(source, dict):
        raw_objs = source.get("objects", [source] if "id" in source else [])
        if inferred_name is None:
            inferred_name = "serialized"
    else:
        raise TypeError(f"Unsupported source type for serialized graph: {type(source)}")

    graph = build_experiment_graph(
        inferred_name or "serialized", run_id, raw_objs, {}, set()
    )

    if promote_sink_configs and len(graph.nodes) > 0:
        dependents: dict[str, list[str]] = {n: [] for n in graph.nodes}
        for n, node in graph.nodes.items():
            for d in node.depends_on:
                if d in dependents:
                    dependents[d].append(n)

        sink_configs = [
            n
            for n, deps in dependents.items()
            if not deps and graph.nodes[n].kind == NodeKind.CONFIG
        ]
        for sink_id in sink_configs:
            graph.nodes[sink_id].kind = NodeKind.ACTION

    return graph


def build_experiment_graph(
    experiment_id: str,
    run_id: str,
    serialized: list[dict[str, Any]],
    jobs: dict[str, ExperimentJobInformation],
    action_ids: set[str],
) -> ExperimentGraph:
    """Build the graph of a run from its serialized objects

    :param serialized: The records of the run's ``objects.jsonl``
    :param jobs: The jobs of the run (task identifier and tags)
    :param action_ids: Identifiers of the run's actions
    """
    objects = {int(o["id"]): o for o in serialized}
    identifier = {
        oid: str(o["identifier"]) for oid, o in objects.items() if o.get("identifier")
    }

    # Objects pointed to as a task (by the configurations they output) or init task
    task_oids = {int(o["task"]) for o in objects.values() if o.get("task") is not None}
    init_oids = {int(t) for o in objects.values() for t in o.get("init-tasks", [])}

    def kind(oid: int) -> NodeKind:
        ident = identifier[oid]
        if ident in action_ids:
            return NodeKind.ACTION
        if (
            ident in jobs
            or oid in task_oids
            or objects[oid].get("init-tasks")
            or objects[oid].get("stub")
        ):
            return NodeKind.TASK
        if oid in init_oids:
            return NodeKind.INIT
        return NodeKind.CONFIG

    def references(value: Any, found: dict[str, None], visited: set[int]) -> None:
        """Collect the nodes a serialized value refers to"""
        if isinstance(value, list):
            for item in value:
                references(item, found, visited)
        elif isinstance(value, dict):
            match value.get("type"):
                case "python":
                    oid = int(value["value"])
                    if oid in identifier:
                        found[identifier[oid]] = None
                    elif oid in objects and oid not in visited:
                        visited.add(oid)
                        references(objects[oid].get("fields", {}), found, visited)
                case "path" | "path.serialized":
                    references(value.get("value"), found, visited)
                case _:
                    for item in value.values():
                        references(item, found, visited)
        elif isinstance(value, str) and "/jobs/" in value:
            if m := JOB_PATH_RE.search(value):
                found[m.group(1)] = None

    grid_keys = {k for job in jobs.values() for k in job.tags}
    nodes: dict[str, GraphNode] = {}
    for oid, ident in identifier.items():
        if ident in nodes:
            continue
        o = objects[oid]
        found: dict[str, None] = {}
        references(o.get("fields", {}), found, set())
        for ref in [o.get("task"), *o.get("init-tasks", [])]:
            if ref is not None and int(ref) in identifier:
                found[identifier[int(ref)]] = None
        found.pop(ident, None)

        node_kind = kind(oid)
        job = jobs.get(ident)
        if job is not None:
            tags = dict(job.tags)
        else:
            tags = _config_tags(o.get("fields", {}), grid_keys)

        node_name = o.get("type") or str(o.get("typename", "?")).rsplit(".", 1)[-1]
        if o.get("stub"):
            node_name = f"{node_name} (stub)"

        task_id = (
            job.task_id
            if job is not None
            else (
                o.get("typename") or f"{o.get('module', '')}.{o.get('type', '')}"
                if node_kind.is_job
                else None
            )
        )

        nodes[ident] = GraphNode(
            id=ident,
            name=node_name,
            kind=node_kind,
            task_id=task_id,
            tags=tags,
            depends_on=list(found),
            object_id=oid,
        )

    _remove_back_edges(nodes)

    detected_configs = (
        detect_grid_configs((job.task_id, job.tags) for job in jobs.values())
        if jobs
        else detect_grid_configs(
            (n.task_id or "", n.tags)
            for n in nodes.values()
            if n.kind == NodeKind.TASK and n.tags
        )
    )

    return ExperimentGraph(
        experiment_id=experiment_id,
        run_id=run_id,
        nodes=nodes,
        configs=detected_configs,
        objects=objects,
    )


def _config_tags(fields: dict[str, Any], grid_keys: set[str]) -> dict[str, Any]:
    """Scalar parameters of a configuration, restricted to the tagged ones if any"""
    return {
        k: v
        for k, v in fields.items()
        if isinstance(v, (int, float, str, bool))
        and not k.startswith("_")
        and (not grid_keys or k in grid_keys)
    }


def _remove_back_edges(nodes: dict[str, GraphNode]) -> None:
    """Break the cycles created by generated paths

    A configuration within a task can hold a path generated in the task
    directory: it then refers to the task that refers to it. Edges from a
    configuration to a node that (transitively) depends on it are dropped.
    """
    for scc in _strongly_connected_components(nodes):
        if len(scc) == 1:
            continue
        for n in scc:
            if not nodes[n].kind.is_job:
                nodes[n].depends_on = [d for d in nodes[n].depends_on if d not in scc]


def _strongly_connected_components(nodes: dict[str, GraphNode]) -> list[set[str]]:
    """Tarjan's algorithm (iterative)"""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[set[str]] = []

    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(nodes[root].depends_on))]
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        while work:
            n, children = work[-1]
            for child in children:
                if child not in nodes:
                    continue
                if child not in index:
                    index[child] = low[child] = len(index)
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(nodes[child].depends_on)))
                    break
                if child in on_stack:
                    low[n] = min(low[n], index[child])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[n])
                if low[n] == index[n]:
                    component = set()
                    while True:
                        m = stack.pop()
                        on_stack.discard(m)
                        component.add(m)
                        if m == n:
                            break
                    components.append(component)
    return components


# --- Standalone export


def run_statuses(
    provider: StateProvider, experiment_id: str, run_id: str | None = None
) -> dict[str, str]:
    """Displayed status of each job of a run (as in the web UI and the TUI)"""
    from experimaestro.scheduler.job_status import display_state

    experiment = next(
        (
            run
            for run in provider.get_experiment_runs(experiment_id)
            if run_id is None or run.run_id == run_id
        ),
        None,
    )
    return {
        job.identifier: display_state(job, experiment).name.lower()
        for job in provider.get_jobs(experiment_id, run_id)
    }


def run_progress(
    provider: StateProvider, experiment_id: str, run_id: str | None = None
) -> dict[str, float]:
    """Progress (between 0 and 1) of the running jobs of a run that report one

    This is the top level progress, as displayed in the job lists.
    """
    from experimaestro.scheduler.interfaces import JobState

    progress: dict[str, float] = {}
    for job in provider.get_jobs(experiment_id, run_id):
        if job.state == JobState.RUNNING and job.progress:
            # Progress information, or its serialized form
            top = job.progress[0]
            progress[job.identifier] = (
                top["progress"] if isinstance(top, dict) else top.progress
            )
    return progress


def export_state(
    graph: ExperimentGraph,
    statuses: dict[str, str],
    progress: dict[str, float] | None = None,
) -> dict[str, Any]:
    """What the standalone page displays: the graph with all its views"""
    index = graph.index()
    return {
        "graph": graph.state_dict(),
        "views": {key: view.state_dict(index) for key, view in graph.views().items()},
        "statuses": statuses,
        "progress": progress or {},
    }


def standalone_html(
    graph: ExperimentGraph,
    statuses: dict[str, str],
    progress: dict[str, float] | None = None,
) -> str:
    """A self-contained HTML page displaying the graph

    The page embeds the web UI graph bundle (``webui/data/dag.{js,css}``) and
    the graph data, and needs no server.

    :param progress: Progress of the running jobs (see :func:`run_progress`)
    """
    from importlib.resources import files

    data = files("experimaestro.webui") / "data"
    script = (
        (data / "dag.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    )
    style = (data / "dag.css").read_text(encoding="utf-8")
    # "<" is escaped so that the data cannot close the script element
    payload = json.dumps(export_state(graph, statuses, progress)).replace(
        "<", "\\u003c"
    )
    title = html.escape(f"{graph.experiment_id} ({graph.run_id})")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>body {{ margin: 0; font-family: system-ui, sans-serif; }}</style>
<style>{style}</style>
</head>
<body>
<div id="root"></div>
<script id="xpm-graph" type="application/json">{payload}</script>
<script>{script}</script>
</body>
</html>
"""


def to_mermaid(
    graph: ExperimentGraph,
    view: GraphView | None = None,
    *,
    tasks_only: bool = True,
    stack: bool = True,
    orientation: str = "TD",
) -> str:
    """Generate Mermaid flowchart markdown for the given graph or view.

    :param graph: The experiment dependency graph
    :param view: Explicit GraphView to render (if None, computed with tasks_only and stack)
    :param tasks_only: Only show tasks/jobs if view is not provided
    :param stack: Stack similar nodes if view is not provided
    :param orientation: Flowchart orientation ("TD" or "LR")
    """
    if view is None:
        has_tasks = any(n.kind == NodeKind.TASK for n in graph.nodes.values())
        effective_tasks_only = tasks_only if has_tasks else False
        view = graph.view(tasks_only=effective_tasks_only, stack=stack)

    lines = [f"graph {orientation}"]
    node_id_map: dict[str, str] = {}
    for idx, vn in enumerate(view.nodes):
        node_ref = f"node_{idx}"
        node_id_map[vn.id] = node_ref
        node = graph.nodes[vn.id]
        label = node.name
        if len(vn.members) > 1:
            label += f" (×{len(vn.members)})"
        safe_label = label.replace('"', '\\"')
        lines.append(f'    {node_ref}["{safe_label}"]')

    for s, t in view.edges:
        if s in node_id_map and t in node_id_map:
            lines.append(f"    {node_id_map[s]} --> {node_id_map[t]}")

    return "\n".join(lines)
