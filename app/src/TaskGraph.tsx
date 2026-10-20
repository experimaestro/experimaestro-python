import React, { useEffect, useRef, useState } from "react";

import { useAppSelector } from "./store";
import JobModal from "./JobModal";
import client, { ExperimentGraphResult } from "./client";
import DagView from "./dag/DagView";
import { ViewParams } from "./dag/types";

// Dependency graph of the selected experiment run: the graph and its views
// are computed by the server, the job statuses come from the store
export default () => {
  const jobs = useAppSelector((state) => state.db.jobs);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);
  const currentRun = useAppSelector((state) => state.db.currentRun);
  const [params, setParams] = useState<ViewParams>({
    config: null,
    tasksOnly: true,
    stack: true,
  });
  const [result, setResult] = useState<ExperimentGraphResult | null>(null);
  const [showJob, setShowJob] = useState<string>();
  const requestId = useRef(0);

  useEffect(() => {
    client.onExperimentGraph = (payload) => {
      if (payload.requestId === requestId.current) setResult(payload);
    };
    return () => {
      client.onExperimentGraph = null;
    };
  }, []);

  // The graph only changes when jobs are submitted, not when their state does
  const jobCount = jobs.ids.length;
  useEffect(() => {
    if (currentExperiment === null) return;
    requestId.current += 1;
    client.experiment_graph(requestId.current, currentExperiment, currentRun, params);
  }, [currentExperiment, currentRun, params, jobCount]);

  // Start afresh when switching experiment
  useEffect(() => setResult(null), [currentExperiment, currentRun]);

  if (currentExperiment === null) {
    return <p className="text-muted">Select an experiment to see its graph.</p>;
  }
  if (result?.error) {
    return <p className="text-muted">{result.error}</p>;
  }
  if (!result?.graph) {
    return <p className="text-muted">Loading the graph…</p>;
  }

  return (
    <>
      <DagView
        graph={result.graph}
        view={result.view ?? null}
        params={params}
        onParamsChange={setParams}
        statusOf={(jobId) => jobs.byId[jobId]?.status}
        onOpenJob={(node) => {
          if (jobs.byId[node.id]) {
            client.job_details(node.id, node.taskId ?? undefined);
            setShowJob(node.id);
          }
        }}
      />
      {showJob && jobs.byId[showJob] && (
        <JobModal onHide={() => setShowJob(undefined)} job={jobs.byId[showJob]} />
      )}
    </>
  );
};
