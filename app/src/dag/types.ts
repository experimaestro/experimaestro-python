// Experiment dependency graph, as built by experimaestro.ui.graph

export type NodeKind = "task" | "action" | "init" | "config";

export type Tags = { [key: string]: unknown };

export interface GraphNode {
  id: string;
  name: string;
  kind: NodeKind;
  taskId: string | null;
  tags: Tags;
  dependsOn: string[];
}

export interface ExperimentGraph {
  experimentId: string;
  runId: string;
  nodes: GraphNode[];
  // Grid search configurations (tag values)
  configs: Tags[];
}

// A drawn node: members are positions in ExperimentGraph.nodes, the first one
// standing for the stack; tags are only given for stacks (merged)
export interface ViewNode {
  members: number[];
  tags?: Tags;
}

export interface GraphView {
  nodes: ViewNode[];
  // (source, target) pairs of stack representatives
  edges: [number, number][];
}

export interface ViewParams {
  // Grid search configuration index, null for the whole graph
  config: number | null;
  tasksOnly: boolean;
  stack: boolean;
}

// Mirrors experimaestro.ui.graph.view_key
export const viewKey = (p: ViewParams): string =>
  `${p.config === null ? "all" : p.config}:${p.tasksOnly ? 1 : 0}:${p.stack ? 1 : 0}`;

// What the standalone page embeds (experimaestro.ui.graph.export_state)
export interface ExportState {
  graph: ExperimentGraph;
  views: { [key: string]: GraphView };
  statuses: { [jobId: string]: string };
  // Progress (between 0 and 1) of the running jobs that report one
  progress?: { [jobId: string]: number };
}
