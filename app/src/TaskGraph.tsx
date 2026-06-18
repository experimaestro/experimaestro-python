import React, { useMemo, useState, useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  Position,
  MarkerType,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "@dagrejs/dagre";

import { Job } from "./reducers";
import { useAppSelector } from "./store";
import JobModal from "./JobModal";
import client from "./client";

const STATUS_COLORS: Array<[string, string]> = [
  ["done", "#2e7d32"],
  ["running", "#f9a825"],
  ["ready", "#1565c0"],
  ["waiting", "#ef6c00"],
  ["error", "#c62828"],
];

const statusColor = (status: string): string => {
  const found = STATUS_COLORS.find(([s]) => s === status);
  return found ? found[1] : "#616161";
};

const NODE_W = 200;
const NODE_BASE_H = 40;
const TAG_ROW_H = 18;

const nodeLabel = (name: string, tags: Job["tags"]) => (
  <div style={{ lineHeight: 1.2 }}>
    <div style={{ fontWeight: 600 }}>{name}</div>
    {tags.length > 0 && (
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          marginTop: 3,
        }}
      >
        {tags.map((tag) => (
          <span key={tag[0]} className="tag" title={`${tag[0]}: ${tag[1]}`}>
            <span className="name">{tag[0]}</span>
            <span className="value">{String(tag[1])}</span>
          </span>
        ))}
      </div>
    )}
  </div>
);

function buildGraph(jobs: Job[]): { nodes: Node[]; edges: Edge[] } {
  const present = new Set(jobs.map((j) => j.jobId));

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 30, ranksep: 70 });

  const nodeHeight = (j: Job) =>
    j.tags.length > 0 ? NODE_BASE_H + TAG_ROW_H : NODE_BASE_H;

  jobs.forEach((j) => g.setNode(j.jobId, { width: NODE_W, height: nodeHeight(j) }));

  const edges: Edge[] = [];
  jobs.forEach((j) => {
    (j.dependsOn || []).forEach((dep) => {
      if (!present.has(dep)) return;
      g.setEdge(dep, j.jobId);
      edges.push({
        id: `${dep}->${j.jobId}`,
        source: dep,
        target: j.jobId,
        markerEnd: { type: MarkerType.ArrowClosed },
        style: { stroke: "#888" },
      });
    });
  });

  dagre.layout(g);

  const nodes: Node[] = jobs.map((j) => {
    const pos = g.node(j.jobId);
    const h = nodeHeight(j);
    const name = j.taskId.split(".").pop() || j.taskId;
    const color = statusColor(j.status);
    return {
      id: j.jobId,
      data: { label: nodeLabel(name, j.tags), color },
      position: { x: pos.x - NODE_W / 2, y: pos.y - h / 2 },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      style: {
        width: NODE_W,
        background: "white",
        color: "#222",
        border: `3px solid ${color}`,
        borderRadius: 6,
        fontSize: 12,
        padding: 4,
      },
    };
  });

  return { nodes, edges };
}

const Legend = () => (
  <div
    style={{
      display: "flex",
      gap: "0.75rem",
      flexWrap: "wrap",
      marginBottom: "0.5rem",
      fontSize: "0.8rem",
    }}
  >
    {STATUS_COLORS.map(([status, color]) => (
      <span key={status} style={{ display: "inline-flex", alignItems: "center" }}>
        <span
          style={{
            display: "inline-block",
            width: 14,
            height: 14,
            borderRadius: 3,
            background: "white",
            border: `3px solid ${color}`,
            marginRight: 4,
          }}
        />
        {status}
      </span>
    ))}
  </div>
);

export default () => {
  const jobs = useAppSelector((state) => state.db.jobs);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);
  const [showJob, setShowJob] = useState<string>();

  const filtered = useMemo(
    () =>
      jobs.ids
        .map((id) => jobs.byId[id])
        .filter((j) => {
          if (currentExperiment === null) return true;
          return !!j.experimentIds && j.experimentIds.includes(currentExperiment);
        }),
    [jobs.ids, jobs.byId, currentExperiment],
  );

  const { nodes, edges } = useMemo(() => buildGraph(filtered), [filtered]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const job = jobs.byId[node.id];
      if (job) {
        client.job_details(job.jobId, job.taskId);
        setShowJob(node.id);
      }
    },
    [jobs.byId],
  );

  if (filtered.length === 0) {
    return <p className="text-muted">No jobs to display.</p>;
  }

  return (
    <div>
      <Legend />
      <div style={{ height: "75vh", border: "1px solid #ddd", borderRadius: 6 }}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={onNodeClick}
            fitView
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={true}
          >
            <Background />
            <Controls />
            <MiniMap
              nodeColor={(n) => (n.data?.color as string) || "#888"}
              pannable
              zoomable
            />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
      {showJob && jobs.byId[showJob] && (
        <JobModal onHide={() => setShowJob(undefined)} job={jobs.byId[showJob]} />
      )}
    </div>
  );
};
