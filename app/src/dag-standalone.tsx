// Standalone graph page (experimaestro dag --html): the graph, all its views
// and the job statuses are embedded in the page
import React, { useState } from "react";
import { createRoot } from "react-dom/client";

import DagView from "./dag/DagView";
import { ExportState, ViewParams, viewKey } from "./dag/types";

const state: ExportState = JSON.parse(
  document.getElementById("xpm-graph")!.textContent!,
);

const Page = () => {
  const [params, setParams] = useState<ViewParams>({
    config: null,
    tasksOnly: true,
    stack: true,
  });
  const { graph } = state;
  return (
    <div style={{ padding: "0.5rem 1rem" }}>
      <h1 style={{ fontSize: "1.2rem", margin: 0 }}>
        {graph.experimentId} <small style={{ color: "#6e7781" }}>{graph.runId}</small>
      </h1>
      <DagView
        graph={graph}
        view={state.views[viewKey(params)] ?? null}
        params={params}
        onParamsChange={setParams}
        statusOf={(jobId) => state.statuses[jobId]}
        progressOf={(jobId) => state.progress?.[jobId]}
        height="calc(100vh - 3.5rem)"
      />
    </div>
  );
};

createRoot(document.getElementById("root")!).render(<Page />);
