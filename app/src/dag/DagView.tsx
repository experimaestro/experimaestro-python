// Experiment dependency graph view, shared by the web UI (TaskGraph) and the
// standalone HTML export (dag-standalone): it only displays what it is given
import React, { useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  MarkerType,
  MiniMap,
  Node,
  NodeProps,
  Position,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "@dagrejs/dagre";

import "./dag.css";
import {
  ExperimentGraph,
  GraphNode,
  GraphView,
  Tags,
  ViewParams,
  viewKey,
} from "./types";

// Statuses (scheduler/job_status.py display states), most important first:
// a stack takes the colour of its most important member status
const STATUSES: Array<[string, string]> = [
  ["error", "#c62828"],
  ["running", "#1565c0"],
  ["scheduled", "#00838f"],
  ["ready", "#00838f"],
  ["waiting", "#ef6c00"],
  ["unscheduled", "#757575"],
  ["done", "#2e7d32"],
];
const STATUS_COLOR = new Map(STATUSES);
const UNKNOWN_COLOR = "#9e9e9e";

const LEGEND = ["done", "running", "ready", "waiting", "error", "unscheduled"];

const NODE_W = 220;
const MAX_TAGS = 6;

type Orientation = "LR" | "TB";

type NodeData = {
  // Identifiers of the stacked nodes
  members: string[];
  name: string;
  kind: string;
  count: number;
  tags: Array<[string, string]>;
  // Number of members per status (jobs only)
  statuses: Array<[string, number]>;
  orientation: Orientation;
  dimmed: boolean;
  selected: boolean;
};

const formatValue = (v: unknown): string =>
  typeof v === "string" ? v : JSON.stringify(v);

const tagList = (tags: Tags): Array<[string, string]> =>
  Object.entries(tags).map(([k, v]) => [k, formatValue(v)]);

const isJob = (node: GraphNode) => node.kind !== "config";

const statusColor = (status: string | undefined) =>
  (status && STATUS_COLOR.get(status)) || UNKNOWN_COLOR;

const nodeHeight = (tagCount: number, job: boolean) =>
  34 + Math.ceil(Math.min(tagCount, MAX_TAGS + 1) / 2) * 20 + (job ? 8 : 0);

const DagNode = ({ data }: NodeProps<NodeData>) => {
  const horizontal = data.orientation === "LR";
  const main = data.statuses.length > 0 ? data.statuses[0][0] : undefined;
  const total = data.statuses.reduce((s, [, n]) => s + n, 0);
  const classes = [
    "xpm-dag-node",
    `xpm-dag-${data.kind}`,
    data.count > 1 ? "xpm-dag-stacked" : "",
    data.dimmed ? "xpm-dag-dimmed" : "",
    data.selected ? "xpm-dag-selected" : "",
  ];
  return (
    <div
      className={classes.join(" ")}
      style={main ? { borderColor: statusColor(main) } : undefined}
      title={data.name}
      data-members={data.members.join(" ")}
    >
      <Handle type="target" position={horizontal ? Position.Left : Position.Top} />
      <div className="xpm-dag-title">
        <span className="xpm-dag-name">{data.name}</span>
        {data.count > 1 && <span className="xpm-dag-count">×{data.count}</span>}
      </div>
      {data.tags.length > 0 && (
        <div className="xpm-dag-tags">
          {data.tags.slice(0, MAX_TAGS).map(([k, v]) => (
            <span key={k} className="xpm-dag-tag" title={`${k}: ${v}`}>
              <span className="xpm-dag-tag-name">{k}</span>
              <span className="xpm-dag-tag-value">{v}</span>
            </span>
          ))}
          {data.tags.length > MAX_TAGS && (
            <span className="xpm-dag-tag">+{data.tags.length - MAX_TAGS}</span>
          )}
        </div>
      )}
      {total > 0 && (
        <div className="xpm-dag-statusbar">
          {data.statuses.map(([status, n]) => (
            <span
              key={status}
              title={`${n} ${status}`}
              style={{ flexGrow: n, background: statusColor(status) }}
            />
          ))}
        </div>
      )}
      <Handle
        type="source"
        position={horizontal ? Position.Right : Position.Bottom}
      />
    </div>
  );
};

const nodeTypes = { dag: DagNode };

// Positions of the view nodes (top-left corners), computed with dagre
function layout(
  graph: ExperimentGraph,
  view: GraphView,
  orientation: Orientation,
): Array<{ x: number; y: number; height: number }> {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: orientation, nodesep: 24, ranksep: 70 });

  const heights = view.nodes.map((vn) => {
    const node = graph.nodes[vn.members[0]];
    return nodeHeight(Object.keys(vn.tags ?? node.tags).length, isJob(node));
  });
  view.nodes.forEach((vn, ix) =>
    g.setNode(String(vn.members[0]), { width: NODE_W, height: heights[ix] }),
  );
  view.edges.forEach(([s, t]) => g.setEdge(String(s), String(t)));
  dagre.layout(g);

  return view.nodes.map((vn, ix) => {
    const pos = g.node(String(vn.members[0]));
    return { x: pos.x - NODE_W / 2, y: pos.y - heights[ix] / 2, height: heights[ix] };
  });
}

// Representatives of the ancestors and descendants of a view node
function lineage(view: GraphView, rep: number): Set<number> {
  const parents = new Map<number, number[]>();
  const children = new Map<number, number[]>();
  view.edges.forEach(([s, t]) => {
    parents.set(t, [...(parents.get(t) ?? []), s]);
    children.set(s, [...(children.get(s) ?? []), t]);
  });
  const found = new Set([rep]);
  for (const next of [parents, children]) {
    const stack = [rep];
    while (stack.length > 0) {
      for (const n of next.get(stack.pop()!) ?? []) {
        if (!found.has(n)) {
          found.add(n);
          stack.push(n);
        }
      }
    }
  }
  return found;
}

export type DagViewProps = {
  graph: ExperimentGraph;
  // null while the view is being computed
  view: GraphView | null;
  params: ViewParams;
  onParamsChange: (params: ViewParams) => void;
  statusOf: (jobId: string) => string | undefined;
  // Opens the details of a job (web UI only)
  onOpenJob?: (node: GraphNode) => void;
  error?: string | null;
  height?: string;
};

const Details = ({
  graph,
  members,
  statusOf,
  onOpenJob,
  onClose,
}: {
  graph: ExperimentGraph;
  members: number[];
  statusOf: (jobId: string) => string | undefined;
  onOpenJob?: (node: GraphNode) => void;
  onClose: () => void;
}) => {
  const first = graph.nodes[members[0]];
  return (
    <div className="xpm-dag-details">
      <div className="xpm-dag-details-header">
        <strong>{first.name}</strong>
        <span className="xpm-dag-kind">{first.kind}</span>
        {members.length > 1 && <span>×{members.length}</span>}
        <button className="xpm-dag-close" onClick={onClose} title="Close">
          ×
        </button>
      </div>
      {members.map((ix) => {
        const node = graph.nodes[ix];
        const status = isJob(node) ? statusOf(node.id) : undefined;
        return (
          <div key={node.id} className="xpm-dag-member">
            <div className="xpm-dag-member-header">
              {status && (
                <span
                  className="xpm-dag-status"
                  style={{ background: statusColor(status) }}
                >
                  {status}
                </span>
              )}
              <code title={node.id}>{node.id.slice(0, 12)}…</code>
              {onOpenJob && node.taskId && (
                <button className="xpm-dag-open" onClick={() => onOpenJob(node)}>
                  details
                </button>
              )}
            </div>
            {node.taskId && <div className="xpm-dag-taskid">{node.taskId}</div>}
            <div className="xpm-dag-tags">
              {tagList(node.tags).map(([k, v]) => (
                <span key={k} className="xpm-dag-tag">
                  <span className="xpm-dag-tag-name">{k}</span>
                  <span className="xpm-dag-tag-value">{v}</span>
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const Toolbar = ({
  graph,
  params,
  onParamsChange,
  orientation,
  setOrientation,
}: {
  graph: ExperimentGraph;
  params: ViewParams;
  onParamsChange: (params: ViewParams) => void;
  orientation: Orientation;
  setOrientation: (o: Orientation) => void;
}) => (
  <div className="xpm-dag-toolbar">
    {graph.configs.length > 0 && (
      <select
        value={params.config === null ? "" : String(params.config)}
        onChange={(e) =>
          onParamsChange({
            ...params,
            config: e.target.value === "" ? null : Number(e.target.value),
          })
        }
      >
        <option value="">All configurations ({graph.configs.length})</option>
        {graph.configs.map((config, ix) => (
          <option key={ix} value={ix}>
            {tagList(config)
              .map(([k, v]) => `${k}=${v}`)
              .join(", ")}
          </option>
        ))}
      </select>
    )}
    <label>
      <input
        type="checkbox"
        checked={params.tasksOnly}
        onChange={(e) => onParamsChange({ ...params, tasksOnly: e.target.checked })}
      />
      Jobs only
    </label>
    <label>
      <input
        type="checkbox"
        checked={params.stack}
        onChange={(e) => onParamsChange({ ...params, stack: e.target.checked })}
      />
      Stack similar nodes
    </label>
    <button onClick={() => setOrientation(orientation === "LR" ? "TB" : "LR")}>
      {orientation === "LR" ? "Left → right" : "Top → bottom"}
    </button>
    <span className="xpm-dag-legend">
      {LEGEND.map((status) => (
        <span key={status}>
          <span
            className="xpm-dag-swatch"
            style={{ borderColor: statusColor(status) }}
          />
          {status}
        </span>
      ))}
      <span>
        <span className="xpm-dag-swatch xpm-dag-swatch-config" />
        configuration
      </span>
    </span>
  </div>
);

const GraphCanvas = ({
  graph,
  view,
  statusOf,
  selected,
  setSelected,
  orientation,
}: {
  graph: ExperimentGraph;
  view: GraphView;
  statusOf: (jobId: string) => string | undefined;
  selected: number | null;
  setSelected: (rep: number | null) => void;
  orientation: Orientation;
}) => {
  const positions = useMemo(
    () => layout(graph, view, orientation),
    [graph, view, orientation],
  );
  const highlighted = useMemo(
    () => (selected === null ? null : lineage(view, selected)),
    [view, selected],
  );

  const nodes: Node<NodeData>[] = view.nodes.map((vn, ix) => {
    const rep = vn.members[0];
    const node = graph.nodes[rep];
    const counts = new Map<string, number>();
    if (isJob(node)) {
      vn.members.forEach((m) => {
        const status = statusOf(graph.nodes[m].id) ?? "unknown";
        counts.set(status, (counts.get(status) ?? 0) + 1);
      });
    }
    const order = (s: string) => {
      const i = STATUSES.findIndex(([name]) => name === s);
      return i < 0 ? STATUSES.length : i;
    };
    return {
      id: String(rep),
      type: "dag",
      position: { x: positions[ix].x, y: positions[ix].y },
      style: { width: NODE_W },
      data: {
        members: vn.members.map((m) => graph.nodes[m].id),
        name: node.name,
        kind: node.kind,
        count: vn.members.length,
        tags: tagList(vn.tags ?? node.tags),
        statuses: [...counts.entries()].sort((a, b) => order(a[0]) - order(b[0])),
        orientation,
        dimmed: highlighted !== null && !highlighted.has(rep),
        selected: rep === selected,
      },
    };
  });

  const edges: Edge[] = view.edges.map(([s, t]) => {
    const active = highlighted !== null && highlighted.has(s) && highlighted.has(t);
    return {
      id: `${s}-${t}`,
      source: String(s),
      target: String(t),
      markerEnd: { type: MarkerType.ArrowClosed },
      style: {
        stroke: active ? "#1565c0" : "#9e9e9e",
        strokeWidth: active ? 2 : 1,
        opacity: highlighted !== null && !active ? 0.25 : 1,
      },
    };
  });

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={(_, n) => setSelected(Number(n.id))}
      onPaneClick={() => setSelected(null)}
      fitView
      minZoom={0.05}
      nodesDraggable={false}
      nodesConnectable={false}
    >
      <Background />
      <Controls showInteractive={false} />
      <MiniMap
        pannable
        zoomable
        nodeColor={(n) => {
          const statuses = (n.data as NodeData).statuses;
          return statuses.length > 0 ? statusColor(statuses[0][0]) : "#cfd8dc";
        }}
      />
    </ReactFlow>
  );
};

export const DagView = ({
  graph,
  view,
  params,
  onParamsChange,
  statusOf,
  onOpenJob,
  error,
  height = "75vh",
}: DagViewProps) => {
  const [orientation, setOrientation] = useState<Orientation>("LR");
  const [selected, setSelected] = useState<number | null>(null);

  const selectedNode =
    view && selected !== null
      ? view.nodes.find((vn) => vn.members[0] === selected)
      : undefined;

  const changeParams = (p: ViewParams) => {
    setSelected(null);
    onParamsChange(p);
  };

  return (
    <div className="xpm-dag" style={{ height }}>
      <Toolbar
        graph={graph}
        params={params}
        onParamsChange={changeParams}
        orientation={orientation}
        setOrientation={setOrientation}
      />
      <div className="xpm-dag-body">
        <div className="xpm-dag-canvas">
          {error ? (
            <div className="xpm-dag-message">{error}</div>
          ) : view === null ? (
            <div className="xpm-dag-message">Loading…</div>
          ) : view.nodes.length === 0 ? (
            <div className="xpm-dag-message">Nothing to display</div>
          ) : (
            // Remounted (and so fitted to the view) when what is shown changes
            <ReactFlowProvider key={`${viewKey(params)}-${orientation}`}>
              <GraphCanvas
                graph={graph}
                view={view}
                statusOf={statusOf}
                selected={selected}
                setSelected={setSelected}
                orientation={orientation}
              />
            </ReactFlowProvider>
          )}
        </div>
        {selectedNode && (
          <Details
            graph={graph}
            members={selectedNode.members}
            statusOf={statusOf}
            onOpenJob={onOpenJob}
            onClose={() => setSelected(null)}
          />
        )}
      </div>
    </div>
  );
};

export default DagView;
