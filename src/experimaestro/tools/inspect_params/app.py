# ruff: noqa: T201
"""Interactive Textual TUI widgets and application for Experimaestro parameters inspector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import pyperclip  # type: ignore[import-untyped]
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual import work
from textual.widgets import Footer, Header, Input, Static, Tree
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseJob
    from experimaestro.scheduler.state_provider import StateProvider

from experimaestro.tools.inspect_params.models import (
    ParamsGraph,
    ResolvedPath,
)
from experimaestro.tui import APP_TCSS_PATH

logger = logging.getLogger("experimaestro")


class NodeKind(Enum):
    """Enumeration of semantic node classifications in the parameters tree."""

    ROOT_TASK = "root"
    OBJECT = "obj"
    CATEGORY = "cat"
    FIELD_PRIMITIVE = "prim"
    FIELD_PATH = "path"
    FIELD_COLLECTION = "coll"
    METADATA = "meta"
    CARBON = "carbon"
    CYCLE = "cycle"
    SHARED_REF = "shared"


@dataclass
class NodeData:
    """Payload metadata attached to a Textual Tree node.

    Attributes:
        kind: Semantic classification of the tree item.
        title: Display title or name.
        object_id: ID of the underlying Experimaestro object, if applicable.
        field_name: Parameter field name if this node represents a field.
        field_value: Raw or resolved value of the parameter.
        resolved_path: ResolvedPath instance if this node represents a path.
        target_object_id: Primary object ID if this node is a shared reference or jump target.
        cycle_chain: Sequence of object IDs forming a circular reference chain.
        is_shared: Whether this object is referenced multiple times in the DAG.
    """

    kind: NodeKind
    title: str
    object_id: Optional[int] = None
    field_name: Optional[str] = None
    field_value: Any = None
    resolved_path: Optional[ResolvedPath] = None
    target_object_id: Optional[int] = None
    cycle_chain: Optional[List[int]] = None
    is_shared: bool = False


class ParamsTree(Tree[NodeData]):
    """Hierarchical Task Configuration Tree widget.

    Renders Experimaestro tasks, configs, hyperparameters, paths, and collections
    in an interactive collapsible tree structure with cycle detection and shared
    DAG reference indication.
    """

    def __init__(self, graph: ParamsGraph) -> None:
        """Initialize the parameters tree widget.

        Args:
            graph: The indexed ParamsGraph instance to display.
        """
        super().__init__("Parameters")
        self.graph = graph
        self.group_fields: bool = True
        self.filter_query: str = ""
        self.show_absolute_paths: bool = False
        self.primary_nodes: Dict[int, TreeNode[NodeData]] = {}
        self.matched_nodes: List[TreeNode[NodeData]] = []
        self.match_count: int = 0

    def on_mount(self) -> None:
        """Configure tree display settings and build initial tree structure."""
        self.show_root = False
        self.guide_depth = 3
        self.rebuild()

    def set_filter(self, q: str) -> None:
        """Filter the tree view by search query string.

        Args:
            q: Query string to filter fields or objects.
        """
        self.filter_query = q.strip().lower()
        self.rebuild()

    def toggle_grouping(self) -> None:
        """Toggle between categorized field groups and alphabetical sorting."""
        self.group_fields = not self.group_fields
        self.rebuild()

    def toggle_paths(self) -> None:
        """Toggle between absolute and relative path display."""
        self.show_absolute_paths = not self.show_absolute_paths
        self.rebuild()

    def _matches(self, *texts: Any) -> bool:
        """Check if any of the provided strings contain the active search query."""
        if not self.filter_query:
            return False
        q = self.filter_query
        return any(isinstance(t, str) and q in t.lower() for t in texts)

    def _apply_highlight(self, text: Text) -> None:
        """Highlight active search query terms in a Rich Text instance."""
        if self.filter_query:
            text.highlight_words(
                [self.filter_query],
                style="bold black on yellow",
                case_sensitive=False,
            )

    def rebuild(self) -> None:
        """Clear and rebuild the entire tree from the underlying graph."""
        self.clear()
        self.primary_nodes.clear()
        self.matched_nodes.clear()
        self.match_count = 0

        # Metadata header
        meta = self.root.add(
            Text("ℹ️  Metadata & Environment", style="bold cyan"),
            data=NodeData(NodeKind.METADATA, "Metadata"),
        )
        meta.add_leaf(
            Text(f"Workspace: {self.graph.workspace_path or 'unknown'}", style="dim"),
            data=NodeData(NodeKind.METADATA, "Workspace"),
        )
        meta.add_leaf(
            Text(
                f"v{self.graph.version} · experimaestro {self.graph.experimaestro_version}",
                style="dim",
            ),
            data=NodeData(NodeKind.METADATA, "Version"),
        )
        if self.graph.carbon.get("enabled"):
            meta.add_leaf(
                Text(
                    f"🌱 Carbon: {self.graph.carbon.get('provider')} (ON)",
                    style="green",
                ),
                data=NodeData(NodeKind.CARBON, "Carbon", field_value=self.graph.carbon),
            )

        # Root task
        seen: Set[int] = set()
        rid = self.graph.root_task_id
        if rid and rid in self.graph.objects:
            self._render_obj(
                self.root, rid, ancestors=(), seen=seen, label="📦 Root Task"
            )
        else:
            self.root.add_leaf("[dim]No root task found[/dim]")

        # Other root tasks / components if present (e.g. in multi-task graphs / experimaestro.json)
        other_roots = [
            oid
            for oid, node in self.graph.objects.items()
            if oid not in seen and not node.incoming_refs
        ]
        if other_roots:
            other_container = self.root.add(
                Text(
                    f"📋 Other Tasks & Configurations ({len(other_roots)})",
                    style="bold cyan",
                ),
                data=NodeData(NodeKind.CATEGORY, "Other Tasks"),
            )
            for roid in other_roots:
                rnode = self.graph.objects[roid]
                lbl = "⚡ Task" if rnode.is_task else "🧩 Config"
                self._render_obj(
                    other_container, roid, ancestors=(), seen=seen, label=lbl
                )

        # Fallback for disconnected cycles or orphaned nodes
        orphans = [oid for oid in self.graph.objects if oid not in seen]
        if orphans:
            orphan_container = self.root.add(
                Text(
                    f"🧩 Additional Objects ({len(orphans)})",
                    style="bold yellow",
                ),
                data=NodeData(NodeKind.CATEGORY, "Additional Objects"),
            )
            for roid in orphans:
                if roid not in seen:
                    self._render_obj(
                        orphan_container, roid, ancestors=(), seen=seen, label="Object"
                    )

        if self.filter_query:
            if self.match_count == 0:
                self.root.add_leaf(
                    Text(
                        f"❌ No matching parameters or configs for '{self.filter_query}'",
                        style="dim red",
                    ),
                    data=NodeData(NodeKind.METADATA, "No Match"),
                )
            else:
                for n in self.matched_nodes:
                    if n.children:
                        n.expand()
                    curr = n.parent
                    while curr is not None:
                        curr.expand()
                        curr = curr.parent
                self.move_cursor(self.matched_nodes[0])
                self.scroll_to_node(self.matched_nodes[0])
        else:
            for c in self.root.children:
                c.expand()

    @staticmethod
    def _order_fields(fields: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Order fields prioritizing core payload components over auxiliary metadata."""
        CORE_KEYS = {
            "model": 0,
            "value": 0,
            "task": 1,
            "module": 1,
            "dataset": 2,
            "data": 2,
        }
        AUX_KEYS = {
            "settings": 10,
            "hooks": 11,
            "listeners": 12,
            "callbacks": 13,
        }
        return sorted(
            fields.items(),
            key=lambda kv: (
                CORE_KEYS.get(kv[0], 5),
                AUX_KEYS.get(kv[0], 5),
                kv[0],
            ),
        )

    def _render_obj(
        self,
        parent: TreeNode[NodeData],
        oid: int,
        ancestors: Tuple[int, ...],
        seen: Set[int],
        label: str = "",
    ) -> TreeNode[NodeData]:
        """Render an ObjectNode into the tree.

        Handles cycle detection, shared DAG references, field categorization,
        and recursive rendering of child components and init tasks.

        Args:
            parent: Parent tree node to attach to.
            oid: Numeric ID of the object to render.
            ancestors: Tuple of ancestor object IDs for cycle detection.
            seen: Set of already rendered object IDs in the tree.
            label: Optional field or role label prefix.

        Returns:
            The constructed TreeNode.
        """
        node = self.graph.objects[oid]
        hid = node.identifier[:8] if node.identifier else "no-id"

        # Loop / Cycle guard
        if oid in ancestors:
            txt = Text()
            if label:
                txt.append(f"{label}: ", style="bold white")
            txt.append("🔁 [CYCLE] ", style="bold red")
            txt.append(f"{node.type_name} [#{oid} · {hid}]", style="dim red")
            return parent.add_leaf(
                txt,
                data=NodeData(
                    NodeKind.CYCLE,
                    node.type_name,
                    object_id=oid,
                    cycle_chain=list(ancestors) + [oid],
                ),
            )

        # Shared DAG reference
        is_shared = len(node.incoming_refs) > 1
        if oid in seen:
            txt = Text()
            if label:
                txt.append(f"{label}: ", style="bold white")
            txt.append("↪ [Shared] ", style="bold yellow")
            txt.append(f"{node.type_name} [#{oid} · {hid}]", style="dim yellow")
            tn = parent.add(
                txt,
                data=NodeData(
                    NodeKind.SHARED_REF,
                    node.type_name,
                    object_id=oid,
                    target_object_id=oid,
                    is_shared=True,
                ),
            )
            tn.add_leaf(
                Text("(Press 'j' to jump to primary)", style="dim italic"),
                data=NodeData(NodeKind.SHARED_REF, "Hint", target_object_id=oid),
            )
            return tn

        seen.add(oid)
        icon = "📦" if node.is_root_task else ("⚡" if node.is_task else "🧩")
        st = (
            "bold magenta"
            if node.is_root_task
            else ("bold cyan" if node.is_task else "bold blue")
        )
        txt = Text()
        if label:
            txt.append(f"{label}: ", style="bold white")
        txt.append(f"{icon} {node.type_name} ", style=st)
        txt.append(f"[{hid}] ", style="dim")
        if is_shared:
            txt.append(f"(shared x{len(node.incoming_refs)})", style="italic yellow")

        is_match = self._matches(
            node.type_name, node.qualified_name, node.identifier, label
        )
        if is_match:
            self._apply_highlight(txt)

        tree_node = parent.add(
            txt,
            data=NodeData(
                NodeKind.ROOT_TASK if node.is_root_task else NodeKind.OBJECT,
                node.qualified_name,
                object_id=oid,
                is_shared=is_shared,
            ),
        )
        self.primary_nodes[oid] = tree_node
        if is_match:
            self.matched_nodes.append(tree_node)
            self.match_count += 1
        next_anc = ancestors + (oid,)

        # Categorize fields
        sub_c, paths, hparams, colls = [], [], [], []
        for fn, fv in self._order_fields(node.fields):
            if isinstance(fv, dict) and fv.get("type") == "python":
                sub_c.append((fn, fv))
            elif isinstance(fv, dict) and fv.get("type") in ("path", "path.serialized"):
                paths.append((fn, fv))
            elif isinstance(fv, (list, dict)) and not (
                isinstance(fv, dict) and fv.get("type") == "enum"
            ):
                colls.append((fn, fv))
            else:
                hparams.append((fn, fv))

        def _add_inits(target: TreeNode[NodeData]) -> None:
            if node.init_tasks:
                itn = target.add(
                    Text(
                        f"⚡ Init Tasks ({len(node.init_tasks)})", style="bold magenta"
                    ),
                    data=NodeData(NodeKind.CATEGORY, "Init Tasks"),
                )
                for i, it in enumerate(node.init_tasks):
                    if it in self.graph.objects:
                        self._render_obj(itn, it, next_anc, seen, label=f"[{i}]")
                itn.expand()

        if self.group_fields and (len(node.fields) > 3 or node.init_tasks):
            for gtitle, gstyle, items in [
                ("⚙️ Hyperparameters", "bold yellow", hparams),
                ("📁 Paths & Outputs", "bold blue", paths),
                ("🧩 Sub-Configurations", "bold cyan", sub_c),
            ]:
                if items:
                    gn = tree_node.add(
                        Text(gtitle, style=gstyle),
                        data=NodeData(NodeKind.CATEGORY, gtitle),
                    )
                    for fn, fv in items:
                        self._render_val(gn, fn, fv, next_anc, seen)
                    gn.expand()

            for fn, fv in colls:
                self._render_val(tree_node, fn, fv, next_anc, seen)
            _add_inits(tree_node)
        else:
            for fn, fv in self._order_fields(node.fields):
                self._render_val(tree_node, fn, fv, next_anc, seen)
            _add_inits(tree_node)

        return tree_node

    def _render_val(
        self,
        parent: TreeNode[NodeData],
        fn: str,
        v: Any,
        anc: Tuple[int, ...],
        seen: Set[int],
    ) -> None:
        """Render an individual field value (sub-object, path, collection, or primitive).

        Args:
            parent: Parent tree node to attach to.
            fn: Field name.
            v: Field value.
            anc: Tuple of ancestor object IDs.
            seen: Set of visited object IDs.
        """
        if (
            isinstance(v, dict)
            and v.get("type") == "python"
            and "value" in v
            and int(v["value"]) in self.graph.objects
        ):
            self._render_obj(parent, int(v["value"]), anc, seen, label=fn)
            return

        rp = self.graph.resolve_path(v)
        if rp:
            is_match = self._matches(fn, str(rp.resolved), rp.raw_value)
            txt = Text(f"{fn}: ", style="bold white")
            pstr = (
                str(rp.resolved)
                if self.show_absolute_paths
                else f"{rp.raw_value} [{rp.base}]"
            )
            sz = (
                f" ({self._fmt_sz(rp.size_bytes)})"
                if rp.exists and rp.size_bytes is not None
                else ""
            )
            txt.append(
                f"{'📁' if rp.is_dir else '📄'} {pstr}{sz} ", style="underline blue"
            )
            if not rp.exists:
                txt.append("[not found]", style="red")
            if is_match:
                self._apply_highlight(txt)

            leaf_node = parent.add_leaf(
                txt,
                data=NodeData(
                    NodeKind.FIELD_PATH,
                    fn,
                    field_name=fn,
                    field_value=v,
                    resolved_path=rp,
                ),
            )
            if is_match:
                self.matched_nodes.append(leaf_node)
                self.match_count += 1
            return

        if isinstance(v, (list, dict)) and not (
            isinstance(v, dict) and v.get("type") == "enum"
        ):
            is_match = self._matches(fn)
            if isinstance(v, list):
                desc = f"[{len(v)} items]"
                txt = Text(f"{fn}: ", style="bold white") + Text(desc, style="cyan")
                if is_match:
                    self._apply_highlight(txt)
                cn = parent.add(
                    txt,
                    data=NodeData(
                        NodeKind.FIELD_COLLECTION, fn, field_name=fn, field_value=v
                    ),
                )
                if is_match:
                    self.matched_nodes.append(cn)
                    self.match_count += 1
                for k_idx, val in enumerate(v):
                    self._render_val(cn, f"[{k_idx}]", val, anc, seen)
            else:
                desc = f"{{{len(v)} entries}}"
                txt = Text(f"{fn}: ", style="bold white") + Text(desc, style="cyan")
                if is_match:
                    self._apply_highlight(txt)
                cn = parent.add(
                    txt,
                    data=NodeData(
                        NodeKind.FIELD_COLLECTION, fn, field_name=fn, field_value=v
                    ),
                )
                if is_match:
                    self.matched_nodes.append(cn)
                    self.match_count += 1
                for k_key, val in sorted(v.items()):
                    self._render_val(cn, str(k_key), val, anc, seen)
            return

        is_match = self._matches(fn, str(v))
        txt = Text(f"{fn}: ", style="bold white")
        if isinstance(v, dict) and v.get("type") == "enum":
            txt.append(f"{v.get('enum')}.{v.get('value')}", style="bold yellow")
        elif isinstance(v, str):
            txt.append(f'"{v}"', style="green")
        elif isinstance(v, bool):
            txt.append(str(v), style="bold yellow")
        elif isinstance(v, (int, float)):
            txt.append(str(v), style="magenta")
        elif v is None:
            txt.append("null", style="dim")
        else:
            txt.append(str(v), style="white")

        if is_match:
            self._apply_highlight(txt)

        leaf_node = parent.add_leaf(
            txt,
            data=NodeData(NodeKind.FIELD_PRIMITIVE, fn, field_name=fn, field_value=v),
        )
        if is_match:
            self.matched_nodes.append(leaf_node)
            self.match_count += 1

    @staticmethod
    def _fmt_sz(sz: int) -> str:
        """Format file size in human-readable B, KB, or MB units.

        Args:
            sz: Size in bytes.

        Returns:
            Human-readable size string.
        """
        if sz < 1024:
            return f"{sz} B"
        if sz < 1024 * 1024:
            return f"{sz / 1024:.1f} KB"
        return f"{sz / (1024 * 1024):.1f} MB"


class NodeDetailPane(VerticalScroll):
    """Right-hand detail inspector or raw JSON viewer widget.

    Displays structured cards for objects, paths, primitives, carbon metrics,
    and metadata, or syntax-highlighted raw JSON.
    """

    def __init__(self, graph: ParamsGraph) -> None:
        """Initialize the detail inspector pane.

        Args:
            graph: The ParamsGraph instance being inspected.
        """
        super().__init__(id="detail-pane")
        self.graph = graph
        self.current_data: Optional[NodeData] = None
        self.show_raw_json: bool = False

    def compose(self) -> ComposeResult:
        """Compose child widgets for the detail pane."""
        yield Static(id="detail-content")

    def toggle_raw_json(self) -> None:
        """Toggle between formatted inspector card view and raw JSON view."""
        self.show_raw_json = not self.show_raw_json
        self.update_detail(self.current_data)

    def update_detail(self, data: Optional[NodeData]) -> None:
        """Update displayed card or raw JSON based on selected tree node data.

        Args:
            data: NodeData of the selected tree item, or None.
        """
        self.current_data = data
        w = self.query_one("#detail-content", Static)
        if not data:
            w.update(Panel("No node selected", title="Detail Inspector"))
            return

        if self.show_raw_json:
            target = (
                self.graph.objects[data.object_id].fields
                if data.object_id and data.object_id in self.graph.objects
                else {data.field_name: data.field_value}
                if data.field_name
                else self.graph.raw_data
            )
            w.update(
                Panel(
                    Syntax(
                        json.dumps(target, indent=2),
                        "json",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    title="Raw JSON ('J' toggle)",
                )
            )
            return

        k = data.kind
        if k in (
            NodeKind.OBJECT,
            NodeKind.ROOT_TASK,
            NodeKind.SHARED_REF,
            NodeKind.CYCLE,
        ):
            self._render_obj_card(w, data)
        elif k == NodeKind.FIELD_PATH and data.resolved_path:
            self._render_path_card(w, data)
        elif k == NodeKind.FIELD_PRIMITIVE:
            rows = [
                ("Field", str(data.field_name)),
                (
                    "Type",
                    type(data.field_value).__name__
                    if data.field_value is not None
                    else "null",
                ),
                ("Value", f"[bold yellow]{repr(data.field_value)}[/bold yellow] ('c')"),
            ]
            w.update(self._make_table(rows))
        elif k == NodeKind.CARBON:
            c = data.field_value or {}
            rows = [
                (
                    "Tracking",
                    "[green]ENABLED[/green]" if c.get("enabled") else "Disabled",
                ),
                ("Provider", str(c.get("provider"))),
                ("Region", str(c.get("region") or "Auto")),
                ("Interval", f"{c.get('report_interval_s')}s"),
            ]
            w.update(self._make_table(rows, "🌱 Carbon Tracking"))
        elif k == NodeKind.METADATA:
            rows = [
                ("Workspace", str(self.graph.workspace_path or "unknown")),
                ("Job Dir", str(self.graph.job_path or "unknown")),
                ("Schema", f"v{self.graph.version}"),
                ("Experimaestro", self.graph.experimaestro_version),
                ("Objects", str(len(self.graph.objects))),
            ]
            w.update(self._make_table(rows, "ℹ️ Environment"))
        else:
            w.update(Panel(repr(data.field_value), title=data.title))

    def _render_obj_card(self, w: Static, d: NodeData) -> None:
        """Render detailed object card with properties, Python code, and references.

        Args:
            w: Target Static widget.
            d: Selected tree node data.
        """
        oid = d.object_id or d.target_object_id
        node = self.graph.objects.get(oid) if oid else None
        if not node:
            w.update(Panel("Unknown Object", style="red"))
            return

        role = (
            "Root Task 📦"
            if node.is_root_task
            else ("Init Task ⚡" if node.is_task else "Config 🧩")
        )
        refs = (
            ", ".join(f"#{c}.{f}" for c, f in node.incoming_refs)
            if node.incoming_refs
            else "None (Root / Unreferenced)"
        )
        rows = [
            ("Type", f"[bold white]{node.type_name}[/bold white]"),
            ("Full Name", f"[cyan]{node.qualified_name}[/cyan]"),
            ("Object ID", f"#{node.id}"),
            ("Role", role),
            ("Identifier", f"[bold yellow]{node.identifier}[/bold yellow] ('c')"),
            ("Fields", f"{len(node.fields)} fields"),
        ]
        parts: List[Any] = [self._make_table(rows)]

        if d.kind == NodeKind.CYCLE:
            parts.insert(
                0,
                Panel(
                    f"[bold red]Circular Reference![/bold red]\nChain: {' ➔ '.join(f'#{c}' for c in (d.cycle_chain or []))}",
                    style="red",
                    title="⚠️ Loop Guard",
                ),
            )
        if node.docstring:
            parts.append(Panel(node.docstring, title="Docstring", style="dim italic"))

        code = self.graph.to_python_snippet(node.id)
        parts.append(
            Panel(
                Syntax(code, "python", theme="monokai", word_wrap=True),
                title="Python .C(...) ('y' copy)",
                subtitle="Press 'J' for Raw JSON",
            )
        )
        parts.append(self._make_table([("Referenced by", refs)]))
        w.update(Group(*parts))

    def _render_path_card(self, w: Static, d: NodeData) -> None:
        """Render detailed path card with filesystem metadata and preview.

        Args:
            w: Target Static widget.
            d: Selected tree node data.
        """
        rp = d.resolved_path
        if not rp:
            return
        st = "[green]✓ Exists[/green]" if rp.exists else "[red]✗ Not Found[/red]"
        rows = [
            ("Field", f"[bold white]{d.field_name}[/bold white]"),
            ("Base", rp.base),
            ("Raw Value", rp.raw_value),
            ("Resolved", f"[cyan]{rp.resolved}[/cyan] ('c')"),
            ("Status", st),
        ]
        if rp.exists:
            rows.append(("Kind", "Directory 📁" if rp.is_dir else "File 📄"))
            if rp.size_bytes is not None:
                rows.append(("Size", ParamsTree._fmt_sz(rp.size_bytes)))
            if rp.mtime:
                rows.append(("Modified", rp.mtime.strftime("%Y-%m-%d %H:%M:%S")))

        parts: List[Any] = [self._make_table(rows)]
        if rp.exists and not rp.is_dir and (rp.size_bytes or 0) < 500_000:
            try:
                txt = "\n".join(
                    rp.resolved.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[:12]
                )
                parts.append(
                    Panel(txt, title=f"Preview: {rp.resolved.name}", style="dim")
                )
            except Exception:
                pass
        w.update(Group(*parts))

    @staticmethod
    def _make_table(rows: List[Tuple[str, str]], title: Optional[str] = None) -> Any:
        """Construct a formatted two-column table or panel.

        Args:
            rows: List of (key, value) string pairs.
            title: Optional panel title.

        Returns:
            Rich Table or Panel instance.
        """
        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Key", style="bold cyan", width=14)
        tbl.add_column("Value")
        for k, v in rows:
            tbl.add_row(k, v)
        return Panel(tbl, title=title) if title else tbl


# ---------------------------------------------------------------------------
# Standalone App
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inspector Screen & App
# ---------------------------------------------------------------------------


class ParamsInspectorScreen(Screen, inherit_bindings=False):
    """Textual Screen for inspecting experimaestro parameters.

    Supports both standalone CLI usage and embedding within ExperimaestroUI.
    Can load parameters from a local file, pre-parsed ParamsGraph, or
    on-demand via StateProvider (for remote clusters).
    """

    CSS_PATH = APP_TCSS_PATH

    BINDINGS = [
        Binding("g", "toggle_grouping", "Group Fields", show=True),
        Binding("p", "toggle_paths", "Paths", show=True),
        Binding("e", "expand_all", "Expand All", show=True),
        Binding("E", "collapse_all", "Collapse", show=True),
        Binding("slash", "toggle_search", "Search", key_display="/", show=True),
        Binding("c", "copy_value", "Copy", show=True),
        Binding("y", "copy_python", "Copy Python", show=True),
        Binding("J", "toggle_raw_json", "Raw JSON", key_display="J", show=True),
        Binding("j", "jump_primary", "Jump", show=True),
        Binding("escape", "dismiss_or_quit", "Back", show=True, priority=True),
        Binding("q", "dismiss_or_quit", "Back", show=False, priority=True),
    ]

    def __init__(
        self,
        params_target: Path | str | ParamsGraph | None = None,
        job: Optional["BaseJob"] = None,
        state_provider: Optional["StateProvider"] = None,
        title: Optional[str] = None,
    ):
        """Initialize the parameters inspector screen.

        Args:
            params_target: Filesystem path or already-parsed ParamsGraph.
            job: BaseJob instance for StateProvider retrieval.
            state_provider: StateProvider for fetching parameters.
            title: Custom title for the screen header.
        """
        super().__init__()
        self.job = job
        self.state_provider = state_provider
        self.custom_title = title
        self.params_path: Optional[Path] = None
        self.graph: Optional[ParamsGraph] = None
        self._loading = False

        if isinstance(params_target, ParamsGraph):
            self.graph = params_target
            if self.graph.job_path:
                self.params_path = self.graph.job_path
        elif params_target is not None:
            self.params_path = Path(params_target).resolve()
            self.graph = ParamsGraph.from_file(self.params_path)
        elif self.job is not None and self.state_provider is not None:
            if not self.state_provider.is_remote:
                p = self.state_provider.get_job_params_file(self.job)
                if p:
                    self.params_path = p
                    self.graph = ParamsGraph.from_file(p)
                else:
                    self._loading = True
            else:
                self._loading = True
        else:
            raise ValueError(
                "Either params_target or (job and state_provider) must be provided"
            )

        if self.custom_title:
            self.title = self.custom_title
        elif self.params_path and self.graph:
            jname = self.graph.job_path.name if self.graph.job_path else "job"
            self.sub_title = f"{jname} ({self.params_path.parent.parent.name if self.graph.job_path else ''})"

    def compose(self) -> ComposeResult:
        """Compose top-level layout of the screen."""
        yield Header()
        if self._loading:
            yield Static(
                "Syncing parameters from remote...",
                id="loading",
                classes="loading-message",
            )
        else:
            with Horizontal(id="mode-bar"):
                yield Static(" ⚙️ Config Tree ", id="view-title", classes="title-badge")
                yield Static(
                    " [g] Group  ·  [p] Paths  ·  [J] Raw JSON  ·  [/] Search",
                    id="mode-hints",
                )
            with Horizontal(id="search-bar-container", classes="hidden"):
                yield Input(placeholder="Filter parameters...", id="search-input")
                yield Static("", id="search-status")
            with Horizontal(id="split-container"):
                with Vertical(id="tree-container"):
                    yield ParamsTree(self.graph)
                yield NodeDetailPane(self.graph)
        yield Footer()

    def on_mount(self) -> None:
        """Set initial focus or start remote sync on mount."""
        if self._loading:
            self._start_remote_sync()
        else:
            self.query_one(ParamsTree).focus()

    @work(thread=True)
    def _start_remote_sync(self) -> None:
        """Fetch parameters file via StateProvider in background thread."""
        try:
            if self.state_provider and self.job:
                p = self.state_provider.get_job_params_file(self.job)
                if p and p.is_file():
                    self.params_path = p
                    self.graph = ParamsGraph.from_file(p)
                    self.app.call_from_thread(self._on_sync_success)
                    return
            self.app.call_from_thread(
                self._on_sync_failure, "Parameters file not found"
            )
        except Exception as e:
            self.app.call_from_thread(self._on_sync_failure, str(e))

    def _on_sync_success(self) -> None:
        """Mount inspector widgets once remote parameters have been retrieved."""
        self._loading = False
        try:
            loading = self.query_one("#loading", Static)
            loading.remove()
        except Exception:
            pass

        footer = self.query_one(Footer)
        mode_bar = Horizontal(
            Static(" ⚙️ Config Tree ", id="view-title", classes="title-badge"),
            Static(
                " [g] Group  ·  [p] Paths  ·  [J] Raw JSON  ·  [/] Search",
                id="mode-hints",
            ),
            id="mode-bar",
        )
        search_bar = Horizontal(
            Input(placeholder="Filter parameters...", id="search-input"),
            Static("", id="search-status"),
            id="search-bar-container",
            classes="hidden",
        )
        split = Horizontal(
            Vertical(ParamsTree(self.graph), id="tree-container"),
            NodeDetailPane(self.graph),
            id="split-container",
        )
        self.mount(mode_bar, before=footer)
        self.mount(search_bar, before=footer)
        self.mount(split, before=footer)

        if not self.custom_title and self.params_path and self.graph:
            jname = self.graph.job_path.name if self.graph.job_path else "job"
            self.sub_title = f"{jname} ({self.params_path.parent.parent.name if self.graph.job_path else ''})"

        tree = self.query_one(ParamsTree)
        tree.focus()

    def _on_sync_failure(self, error_msg: str) -> None:
        """Display error message when sync fails."""
        try:
            loading = self.query_one("#loading", Static)
            loading.update(f"⚠️ {error_msg}\n\nPress Esc or 'q' to go back.")
        except Exception:
            pass

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update detail inspector pane when cursor moves over a node."""
        self.query_one(NodeDetailPane).update_detail(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Activate selected node, jumping to primary definition if shared."""
        if event.node.data and event.node.data.target_object_id:
            self.action_jump_primary()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter tree view as query string changes."""
        tree = self.query_one(ParamsTree)
        tree.set_filter(event.value)
        inp = self.query_one("#search-input", Input)
        status = self.query_one("#search-status", Static)
        hints = self.query_one("#mode-hints", Static)

        q = event.value.strip()
        if q:
            if tree.match_count == 0:
                inp.add_class("error")
                inp.remove_class("valid")
                status.remove_class("valid")
                status.add_class("error")
                status.update(" (not found)")
                hints.update(
                    f" [g] Group  ·  [p] Paths  ·  [/] Search: '{q}' [bold red](not found)[/bold red]"
                )
            else:
                inp.remove_class("error")
                inp.add_class("valid")
                status.remove_class("error")
                status.add_class("valid")
                status.update(f" ({tree.match_count} matched)")
                hints.update(
                    f" [g] Group  ·  [p] Paths  ·  [/] Search: '{q}' [green]({tree.match_count} matched)[/green]"
                )
        else:
            inp.remove_class("error")
            inp.remove_class("valid")
            status.remove_class("error")
            status.remove_class("valid")
            status.update("")
            hints.update(" [g] Group  ·  [p] Paths  ·  [J] Raw JSON  ·  [/] Search")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Hide search bar and return focus to tree upon query submission."""
        self.query_one("#search-bar-container").add_class("hidden")
        self.query_one(ParamsTree).focus()

    def action_toggle_grouping(self) -> None:
        """Toggle categorized field groups versus flat alphabetical ordering."""
        tree = self.query_one(ParamsTree)
        tree.toggle_grouping()
        self.notify(
            f"Field grouping: {'Categorized' if tree.group_fields else 'Alphabetical'}"
        )

    def action_toggle_paths(self) -> None:
        """Toggle absolute filesystem path versus raw relative path display."""
        tree = self.query_one(ParamsTree)
        tree.toggle_paths()
        self.notify(
            f"Path display: {'Absolute' if tree.show_absolute_paths else 'Relative'}"
        )

    def action_toggle_raw_json(self) -> None:
        """Toggle raw JSON view versus structured inspector card in detail pane."""
        pane = self.query_one(NodeDetailPane)
        pane.toggle_raw_json()
        self.notify(
            f"Detail pane: {'Raw JSON' if pane.show_raw_json else 'Inspector Card'}"
        )

    def action_expand_all(self) -> None:
        """Recursively expand all nodes in the parameter tree."""

        def _exp(n: TreeNode[NodeData]) -> None:
            n.expand()
            for c in n.children:
                _exp(c)

        _exp(self.query_one(ParamsTree).root)
        self.notify("Expanded all")

    def action_collapse_all(self) -> None:
        """Collapse all child nodes in the parameter tree."""

        def _col(n: TreeNode[NodeData]) -> None:
            for c in n.children:
                _col(c)
                c.collapse()

        _col(self.query_one(ParamsTree).root)
        self.notify("Collapsed all")

    def action_toggle_search(self) -> None:
        """Toggle visibility and focus of the search filter input."""
        box = self.query_one("#search-bar-container")
        if "hidden" in box.classes:
            box.remove_class("hidden")
            self.query_one("#search-input", Input).focus()
        else:
            box.add_class("hidden")
            self.query_one(ParamsTree).focus()

    def action_copy_value(self) -> None:
        """Copy the identifier, resolved path, or raw value of the selected node."""
        node = self.query_one(ParamsTree).cursor_node
        if not node or not node.data:
            return
        d = node.data
        val = (
            self.graph.objects[d.object_id].identifier
            if d.object_id
            and d.kind in (NodeKind.OBJECT, NodeKind.ROOT_TASK, NodeKind.SHARED_REF)
            else str(d.resolved_path.resolved)
            if d.resolved_path
            else str(d.field_value)
            if d.field_value is not None
            else ""
        )
        if val:
            pyperclip.copy(val)
            self.notify(f"Copied: {val[:32]}...")

    def action_copy_python(self) -> None:
        """Copy the reproducible Python constructor snippet of the selected object."""
        node = self.query_one(ParamsTree).cursor_node
        oid = node.data.object_id if node and node.data else self.graph.root_task_id
        if oid and oid in self.graph.objects:
            pyperclip.copy(self.graph.to_python_snippet(oid))
            self.notify("Copied Python .C(...) constructor!")

    def action_jump_primary(self) -> None:
        """Navigate to the primary definition node in the tree from a shared reference."""
        node = self.query_one(ParamsTree).cursor_node
        if not node or not node.data:
            return
        tid = node.data.target_object_id or node.data.object_id
        tree = self.query_one(ParamsTree)
        if tid and tid in tree.primary_nodes:
            target = tree.primary_nodes[tid]

            # 1. Expand all ancestors from root down to target so it is visible
            curr = target.parent
            while curr is not None:
                if not curr.is_expanded:
                    curr.expand()
                curr = curr.parent

            # 2. Expand target itself so its parameters/fields are revealed
            target.expand()
            for child in target.children:
                if child.data and child.data.kind == NodeKind.CATEGORY:
                    child.expand()

            # 3. Move cursor to target and scroll into view
            tree.move_cursor(target)
            tree.scroll_to_node(target)
            self.query_one(NodeDetailPane).update_detail(target.data)
            self.notify(f"Jumped to primary definition #{tid}")

    def action_dismiss_or_quit(self) -> None:
        """Dismiss the search bar if visible, or exit the application/screen."""
        if not self._loading:
            try:
                box = self.query_one("#search-bar-container")
                if "hidden" not in box.classes:
                    box.add_class("hidden")
                    inp = self.query_one("#search-input", Input)
                    inp.value = ""
                    inp.remove_class("error")
                    inp.remove_class("valid")
                    status = self.query_one("#search-status", Static)
                    status.remove_class("error")
                    status.remove_class("valid")
                    status.update("")
                    self.query_one("#mode-hints", Static).update(
                        " [g] Group  ·  [p] Paths  ·  [J] Raw JSON  ·  [/] Search"
                    )
                    tree = self.query_one(ParamsTree)
                    tree.set_filter("")
                    tree.focus()
                    return
            except Exception:
                pass

        if len(self.app.screen_stack) > 1:
            self.dismiss()
        else:
            self.app.exit()


class ParamsInspectorApp(App):
    """Standalone Textual App for inspecting experimaestro params.json.

    Features interactive tree navigation, categorization, path resolution,
    fuzzy search filtering, Python snippet copying, and raw JSON inspection.
    """

    TITLE = "Experimaestro Parameters Inspector"
    CSS_PATH = APP_TCSS_PATH

    def __init__(self, params_path: Path):
        """Initialize the parameters inspector application.

        Args:
            params_path: Filesystem path to the parameters JSON file.
        """
        super().__init__()
        self.params_path = Path(params_path).resolve()
        self.inspector_screen = ParamsInspectorScreen(self.params_path)

    def get_default_screen(self) -> Screen:
        """Return the inspector screen as the default screen."""
        return self.inspector_screen

    @property
    def graph(self) -> Optional[ParamsGraph]:
        """Accessor for the underlying ParamsGraph."""
        return self.inspector_screen.graph

    def action_expand_all(self) -> None:
        """Forward expand_all to inspector screen."""
        self.inspector_screen.action_expand_all()

    def action_collapse_all(self) -> None:
        """Forward collapse_all to inspector screen."""
        self.inspector_screen.action_collapse_all()

    def action_toggle_grouping(self) -> None:
        """Forward toggle_grouping to inspector screen."""
        self.inspector_screen.action_toggle_grouping()

    def action_toggle_paths(self) -> None:
        """Forward toggle_paths to inspector screen."""
        self.inspector_screen.action_toggle_paths()

    def action_toggle_raw_json(self) -> None:
        """Forward toggle_raw_json to inspector screen."""
        self.inspector_screen.action_toggle_raw_json()

    def action_toggle_search(self) -> None:
        """Forward toggle_search to inspector screen."""
        self.inspector_screen.action_toggle_search()

    def action_copy_value(self) -> None:
        """Forward copy_value to inspector screen."""
        self.inspector_screen.action_copy_value()

    def action_copy_python(self) -> None:
        """Forward copy_python to inspector screen."""
        self.inspector_screen.action_copy_python()

    def action_jump_primary(self) -> None:
        """Forward jump_primary to inspector screen."""
        self.inspector_screen.action_jump_primary()
