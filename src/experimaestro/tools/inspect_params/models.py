"""Core data models and dependency graph for Experimaestro parameters inspector.

Represents serialized objects, handles reference traversal, cycle detection,
path resolution, and Python code synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from experimaestro.core.identifier import Identifier

logger = logging.getLogger("experimaestro")


def normalize_identifier(raw_ident: Any) -> str:
    """Normalize a serialized identifier (string hash or Identifier state dict)."""
    if isinstance(raw_ident, dict):
        return str(Identifier.from_state_dict(raw_ident))
    return str(raw_ident) if raw_ident is not None else ""


def resolve_serialized_value(value: Any, objects: Mapping[int, Any]) -> Any:
    """Resolve a serialized JSON field value against a mapping of loaded objects."""
    if isinstance(value, list):
        return [resolve_serialized_value(x, objects) for x in value]

    if isinstance(value, dict):
        if "type" not in value:
            return {k: resolve_serialized_value(v, objects) for k, v in value.items()}

        vtype = value["type"]
        if vtype == "python":
            ref_id = int(value["value"])
            if ref_id not in objects:
                logger.warning(
                    "Referenced object id %s not found in loaded objects", ref_id
                )
                return None
            return objects[ref_id]

        if vtype in ("path", "path.serialized"):
            return Path(value["value"])

        if vtype == "set":
            return [
                resolve_serialized_value(x, objects) for x in value.get("items", [])
            ]

        if vtype == "enum":
            return value.get("value")

        return value.get("value")

    return value


@dataclass
class ResolvedPath:
    """Representation of a path reference resolved against job or workspace directories.

    Attributes:
        raw_value: Original string representation from the parameters JSON.
        base: Base reference origin ('job', 'workspace', or 'absolute').
        resolved: Fully resolved filesystem Path.
        exists: Whether the file or directory exists on the local filesystem.
        is_dir: Whether the resolved path is a directory.
        size_bytes: File size in bytes if it exists and is a regular file.
        mtime: Last modification timestamp if accessible.
    """

    raw_value: str
    base: str  # 'job' | 'workspace' | 'absolute'
    resolved: Path
    exists: bool = False
    is_dir: bool = False
    size_bytes: Optional[int] = None
    mtime: Optional[datetime] = None

    @classmethod
    def resolve(
        cls, raw_val: str, base: str, job_dir: Optional[Path], ws_dir: Optional[Path]
    ) -> ResolvedPath:
        """Resolve a raw path string relative to job or workspace directory.

        Resolution precedence:
        1. If base is 'job' and job_dir is provided: resolve relative to job_dir.
        2. If base is 'workspace' and ws_dir is provided: resolve relative to ws_dir.
        3. If raw path is relative and job_dir is provided: resolve relative to job_dir.
        4. Otherwise, resolve raw path directly.
        """
        p = Path(raw_val)
        res = (
            (job_dir / p).resolve()
            if base == "job" and job_dir
            else (ws_dir / p).resolve()
            if base == "workspace" and ws_dir
            else (job_dir / p).resolve()
            if not p.is_absolute() and job_dir
            else p.resolve()
        )
        exists = res.exists()
        is_dir = res.is_dir() if exists else False
        size, mtime = None, None
        if exists and not is_dir:
            try:
                st = res.stat()
                size, mtime = st.st_size, datetime.fromtimestamp(st.st_mtime)
            except OSError:
                pass
        return cls(raw_val, base, res, exists, is_dir, size, mtime)


@dataclass
class ObjectNode:
    """A deserialized configuration object or task within the Experimaestro graph.

    Attributes:
        id: Unique numeric object identifier.
        module: Python module declaring the object's class.
        type_name: Short class name of the object.
        qualified_name: Full module.class or qualified typename.
        identifier: Content hash or task identifier string.
        is_task: True if this object represents an executable task.
        is_root_task: True if this is the primary root task of the job.
        init_tasks: IDs of prerequisite tasks that must run before this task.
        fields: Key-value mapping of parameters, sub-configs, or paths.
        incoming_refs: List of (source_object_id, field_name) referencing this object.
        outgoing_refs: List of (field_name, target_object_id) referenced by this object.
        docstring: Optional docstring extracted from the underlying Python class.
    """

    id: int
    module: str
    type_name: str
    qualified_name: str
    identifier: str
    is_task: bool
    is_root_task: bool
    init_tasks: List[int]
    fields: Dict[str, Any]
    incoming_refs: List[Tuple[int, str]] = field(default_factory=list)
    outgoing_refs: List[Tuple[str, int]] = field(default_factory=list)
    docstring: Optional[str] = None
    task_id: Optional[int] = None
    is_init_task: bool = False

    @classmethod
    def from_definition(
        cls, o: Dict[str, Any], *, load_docstring: bool = True
    ) -> ObjectNode:
        """Create an ObjectNode from a raw serialized definition dictionary."""
        oid = int(o["id"])
        mod, typ = o.get("module", ""), o.get("type", "")
        qname = o.get("typename") or f"{mod}.{typ}" if mod else typ or str(oid)
        ident = normalize_identifier(o.get("identifier", ""))
        inits = [int(x) for x in o.get("init-tasks", [])]
        t_ptr = o.get("task")
        task_id = int(t_ptr) if t_ptr is not None else None
        is_task = (task_id is not None and task_id == oid) or bool(inits)

        doc = None
        if load_docstring and mod and typ:
            try:
                cls_obj = getattr(importlib.import_module(mod), typ, None)
                doc = cls_obj.__doc__.strip() if cls_obj and cls_obj.__doc__ else None
            except Exception:
                pass

        raw_fields = o.get("fields", {})
        return cls(
            oid,
            mod,
            typ,
            qname,
            ident,
            is_task,
            False,
            inits,
            raw_fields if isinstance(raw_fields, dict) else {},
            docstring=doc,
            task_id=task_id,
        )


class ParamsGraph:
    """Parses, indexes, and queries the object dependency graph of an Experimaestro job.

    Supports single-job dictionaries (`params.json`), multi-task list payloads
    (such as Hugging Face Hub `experimaestro.json` exports), and streaming
    experiment records (`objects.jsonl`).

    Attributes:
        raw_data: Original deserialized JSON data.
        job_path: Path to the directory containing the parameters file.
        workspace_path: Resolved path to the enclosing workspace directory.
        objects: Mapping of numeric object IDs to ObjectNode instances.
        task_object_ids: Set of object IDs referenced via `task` or `init-tasks`.
        root_task_id: Object ID of the primary root task, if determined.
        version: Schema version number.
        experimaestro_version: Experimaestro version string.
        carbon: Carbon emission tracking configuration dictionary.
    """

    workspace_path: Optional[Path]

    def __init__(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        job_path: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
        *,
        load_docstrings: bool = True,
    ):
        """Initialize the graph from a dictionary or list of raw object specifications.

        Args:
            data: Deserialized params.json dict or experimaestro.json / objects.jsonl list.
            job_path: Optional filesystem path to the enclosing job directory.
            workspace_path: Optional workspace path (auto-discovered if omitted).
            load_docstrings: Whether to import modules to extract class docstrings.
        """
        self.raw_data = data
        self.job_path = job_path.resolve() if job_path else None
        if isinstance(data, list):
            ws = None
            self.version = 0
            self.experimaestro_version = "unknown"
            self.carbon = {}
            raw_objs = data
        else:
            ws = data.get("workspace")
            self.version = data.get("version", 0)
            self.experimaestro_version = data.get("experimaestro", "unknown")
            self.carbon = data.get("carbon", {})
            raw_objs = data.get("objects", [])

        self.workspace_path = (
            Path(ws).resolve()
            if ws
            else workspace_path.resolve()
            if workspace_path
            else self._find_ws(self.job_path)
        )
        self.objects: Dict[int, ObjectNode] = {}
        self.task_object_ids: Set[int] = set()
        self.init_task_object_ids: Set[int] = set()
        self.root_task_id: Optional[int] = None
        self._parse(raw_objs, load_docstrings=load_docstrings)

    @staticmethod
    def _find_ws(path: Optional[Path]) -> Optional[Path]:
        """Search upwards from a directory for the nearest `.experimaestro` root."""
        if not path:
            return None
        for p in [path] + list(path.parents):
            if (p / ".experimaestro").is_dir():
                return p
        return None

    @classmethod
    def from_file(cls, path: Path, *, load_docstrings: bool = True) -> ParamsGraph:
        """Load a parameter graph from a file or directory.

        Accepts direct paths to `params.json`, `experimaestro.json`, or `objects.jsonl`,
        or a directory containing any of these files.
        """
        p = path.resolve()
        if p.is_dir():
            if (p / "params.json").is_file():
                f = p / "params.json"
            elif (p / "experimaestro.json").is_file():
                f = p / "experimaestro.json"
            elif (p / "objects.jsonl").is_file():
                f = p / "objects.jsonl"
            else:
                f = p / "params.json"
        else:
            f = p
        if not f.exists():
            raise FileNotFoundError(f"params.json not found at: {f}")

        text = f.read_text(encoding="utf-8")
        if f.suffix == ".jsonl" or f.name == "objects.jsonl":
            payload: Union[Dict[str, Any], List[Dict[str, Any]]] = [
                json.loads(line) for line in text.splitlines() if line.strip()
            ]
        else:
            payload = json.loads(text)
        return cls(payload, job_path=f.parent, load_docstrings=load_docstrings)

    def _parse(
        self, raw_objs: List[Dict[str, Any]], *, load_docstrings: bool = True
    ) -> None:
        """Construct ObjectNode instances and resolve reference graphs."""
        target_hash = self.job_path.name if self.job_path else None
        cand_root = None

        for o in raw_objs:
            node = ObjectNode.from_definition(o, load_docstring=load_docstrings)
            self.objects[node.id] = node
            if node.task_id is not None:
                self.task_object_ids.add(node.task_id)
            for it in node.init_tasks:
                self.init_task_object_ids.add(it)
            if target_hash and node.identifier == target_hash:
                cand_root = node.id

        for tid in self.task_object_ids:
            if tid in self.objects:
                self.objects[tid].is_task = True

        for it_id in self.init_task_object_ids:
            if it_id in self.objects and not self.objects[it_id].is_task:
                self.objects[it_id].is_init_task = True

        self.root_task_id = (
            cand_root if cand_root else (int(raw_objs[-1]["id"]) if raw_objs else None)
        )
        if self.root_task_id and self.root_task_id in self.objects:
            self.objects[self.root_task_id].is_root_task = True
            self.objects[self.root_task_id].is_task = True

        def _extract_refs(val: Any, prefix: str = "") -> List[Tuple[str, int]]:
            refs: List[Tuple[str, int]] = []
            if isinstance(val, dict):
                if val.get("type") == "python" and "value" in val:
                    refs.append((prefix, int(val["value"])))
                else:
                    for k_sub, v_sub in val.items():
                        sub_p = f"{prefix}.{k_sub}" if prefix else k_sub
                        refs.extend(_extract_refs(v_sub, sub_p))
            elif isinstance(val, list):
                for idx, item in enumerate(val):
                    sub_p = f"{prefix}[{idx}]"
                    refs.extend(_extract_refs(item, sub_p))
            return refs

        for oid, n in self.objects.items():
            for it in n.init_tasks:
                if it in self.objects:
                    n.outgoing_refs.append(("__init_task__", it))
                    self.objects[it].incoming_refs.append((oid, "__init_task__"))
            for k, v in n.fields.items():
                for rk, tid in _extract_refs(v, k):
                    if tid in self.objects:
                        n.outgoing_refs.append((rk, tid))
                        self.objects[tid].incoming_refs.append((oid, rk))

    def resolve_path(self, spec: Any) -> Optional[ResolvedPath]:
        """Attempt to resolve a parameter value as a path specification."""
        if isinstance(spec, dict) and spec.get("type") in ("path", "path.serialized"):
            return ResolvedPath.resolve(
                spec.get("value", ""),
                spec.get("base", "absolute"),
                self.job_path,
                self.workspace_path,
            )
        return None

    def to_python_snippet(
        self, oid: int, indent: int = 0, seen: Optional[Set[int]] = None
    ) -> str:
        """Synthesize a reproducible Python `.C(...)` configuration snippet."""
        seen = seen or set()
        node = self.objects.get(oid)
        if not node:
            return f"#{oid}"
        if oid in seen:
            return f"{node.type_name}.C(...)  # cycle to #{oid}"
        seen.add(oid)

        sp, spi = "    " * indent, "    " * (indent + 1)
        args = [
            f"{spi}{k}={self._fmt_py(v, indent + 1, set(seen))},"
            for k, v in sorted(node.fields.items())
        ]
        return f"{node.type_name}.C(\n" + "\n".join(args) + f"\n{sp})"

    def _fmt_py(self, v: Any, indent: int, seen: Set[int]) -> str:
        """Format an individual field value for Python code synthesis."""
        if isinstance(v, dict):
            t = v.get("type")
            if t == "python":
                return self.to_python_snippet(int(v["value"]), indent, seen)
            if t in ("path", "path.serialized"):
                b = v.get("base")
                return f'Path("{v.get("value")}")' + (f"  # base: {b}" if b else "")
            if t == "enum":
                return f"{v.get('enum')}.{v.get('value')}"
            return (
                "{"
                + ", ".join(
                    f'"{k}": {self._fmt_py(val, indent, seen)}' for k, val in v.items()
                )
                + "}"
            )
        if isinstance(v, list):
            return "[" + ", ".join(self._fmt_py(x, indent, seen) for x in v) + "]"
        return repr(v)
