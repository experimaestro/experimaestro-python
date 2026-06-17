import React, { useEffect } from "react";
import Form from "react-bootstrap/Form";
import { useDispatch } from "react-redux";

import { useAppSelector } from "./store";
import { actions } from "./reducers";
import client from "./client";

export const RunSelector: React.FC = () => {
  const dispatch = useDispatch();
  const currentExperiment = useAppSelector((s) => s.db.currentExperiment);
  const currentRun = useAppSelector((s) => s.db.currentRun);
  const runs = useAppSelector((s) => s.db.runs);

  // Fetch the run history whenever the selected experiment changes
  useEffect(() => {
    if (currentExperiment) {
      client.request_runs(currentExperiment);
    }
  }, [currentExperiment]);

  if (!currentExperiment || runs.length <= 1) {
    return null;
  }

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const runId = event.target.value || null;
    dispatch(actions.selectRun(runId));
    // Reload the jobs for the selected run
    dispatch(actions.clearJobs());
    client.refresh(currentExperiment, runId || undefined);
  };

  return (
    <Form.Select
      value={currentRun || ""}
      onChange={handleChange}
      style={{ width: "auto", marginLeft: "0.5rem" }}
      size="sm"
      title="Run"
    >
      <option value="">Current run</option>
      {runs.map((run) => {
        const id = run.run_id || run.current_run_id || "";
        const star = run.isCurrent ? "★ " : "";
        const status = run.status ? ` [${run.status}]` : "";
        return (
          <option key={id} value={id}>
            {star}
            {id}
            {status} ({run.finished_jobs}/{run.total_jobs})
          </option>
        );
      })}
    </Form.Select>
  );
};
