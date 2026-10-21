import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { useAppSelector, useNavigate } from "./store";
import JobModal from "./JobModal";
import client, { ExperimentGraphResult } from "./client";
import DagView from "./dag/DagView";
import { ViewParams } from "./dag/types";

// Delay (ms) without new jobs before the graph is requested again
const GRAPH_UPDATE_DELAY = 1000;

// Space (px) kept below the graph
const BOTTOM_MARGIN = 16;

/** Height filling the window below the element (updated on resize) */
function useRemainingHeight(): [React.RefObject<HTMLDivElement>, string] {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState("75vh");
  useLayoutEffect(() => {
    const update = () => {
      if (!ref.current) return;
      const top = ref.current.getBoundingClientRect().top + window.scrollY;
      setHeight(`max(20rem, calc(100vh - ${top + BOTTOM_MARGIN}px))`);
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  });
  return [ref, height];
}

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
  const route = useAppSelector((state) => state.db.route);
  const navigate = useNavigate();
  const requestId = useRef(0);
  // First request for the displayed experiment, run and view: the answers to
  // later requests are shown as they arrive (the latest may take a while)
  const firstRequest = useRef(1);
  const shownRequest = useRef(0);
  const [container, height] = useRemainingHeight();

  useEffect(() => {
    client.onExperimentGraph = (payload) => {
      if (payload.requestId < firstRequest.current) return;
      if (payload.requestId <= shownRequest.current) return;
      shownRequest.current = payload.requestId;
      setResult(payload);
    };
    return () => {
      client.onExperimentGraph = null;
    };
  }, []);

  // The graph only changes when jobs are submitted to the displayed
  // experiment, not when their state does
  const jobCount = useMemo(
    () =>
      jobs.ids.filter((id) =>
        jobs.byId[id].experimentIds?.includes(currentExperiment ?? ""),
      ).length,
    [jobs, currentExperiment],
  );

  // Read by the delayed request, which must use the current values
  const latest = useRef({ currentExperiment, currentRun, params, jobCount });
  latest.current = { currentExperiment, currentRun, params, jobCount };
  const requestedCount = useRef(-1);
  const request = () => {
    const { currentExperiment, currentRun, params, jobCount } = latest.current;
    if (currentExperiment === null) return;
    requestId.current += 1;
    requestedCount.current = jobCount;
    client.experiment_graph(requestId.current, currentExperiment, currentRun, params);
  };

  // Another experiment, run or view: request it right away
  useEffect(() => {
    firstRequest.current = requestId.current + 1;
    request();
  }, [currentExperiment, currentRun, params]);

  // Jobs submitted: request once they stop arriving (they come in bursts)
  useEffect(() => {
    if (jobCount === requestedCount.current) return;
    const timer = window.setTimeout(request, GRAPH_UPDATE_DELAY);
    return () => window.clearTimeout(timer);
  }, [jobCount]);

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
    <div ref={container}>
      <DagView
        height={height}
        graph={result.graph}
        view={result.view ?? null}
        params={params}
        onParamsChange={setParams}
        statusOf={(jobId) => jobs.byId[jobId]?.status}
        onOpenJob={(node) => {
          const job = jobs.byId[node.id];
          if (job) {
            navigate({
              ...route,
              job: { taskId: job.taskId, jobId: job.jobId, tab: "details" },
            });
          }
        }}
      />
      <JobModal />
    </div>
  );
};
