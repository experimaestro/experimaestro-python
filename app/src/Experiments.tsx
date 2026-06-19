import React from "react";
import { Table, Badge } from "react-bootstrap";

import { useAppSelector } from "./store";
import { Experiment } from "./reducers";
import { formatShortDate, formatDuration, formatCO2 } from "./format";

function statusInfo(exp: Experiment): { icon: string; color: string; label: string } {
  if (exp.failed_jobs > 0) return { icon: "❌", color: "danger", label: "failed" };
  if (exp.status === "running" || exp.finished_jobs < exp.total_jobs)
    return { icon: "🏃", color: "primary", label: "running" };
  return { icon: "✓", color: "success", label: "done" };
}

export default ({ onOpen }: { onOpen: (id: string) => void }) => {
  const experiments = useAppSelector((s) => s.db.experiments);

  if (experiments.ids.length === 0) {
    return <p className="text-muted mt-3">No experiments yet.</p>;
  }

  return (
    <div className="mt-3">
      <h5 className="mb-3">Experiments</h5>
      <Table hover responsive className="align-middle experiments-table">
        <thead>
          <tr>
            <th>Status</th>
            <th>Experiment</th>
            <th>Jobs</th>
            <th>Host</th>
            <th>Started</th>
            <th>Duration</th>
            <th>CO₂</th>
          </tr>
        </thead>
        <tbody>
          {experiments.ids.map((id) => {
            const exp = experiments.byId[id];
            const st = statusInfo(exp);
            return (
              <tr
                key={id}
                style={{ cursor: "pointer" }}
                onClick={() => onOpen(id)}
                title="Open jobs"
              >
                <td>
                  <span title={st.label}>{st.icon}</span>
                </td>
                <td>
                  <span className="fw-semibold">{exp.experiment_id}</span>
                  {exp.current_run_id && (
                    <div className="text-muted small">{exp.current_run_id}</div>
                  )}
                </td>
                <td>
                  <Badge bg={st.color}>
                    {exp.finished_jobs}/{exp.total_jobs}
                  </Badge>
                  {exp.failed_jobs > 0 && (
                    <Badge bg="danger" className="ms-1">
                      {exp.failed_jobs} failed
                    </Badge>
                  )}
                </td>
                <td className="text-muted small">{exp.hostname || "—"}</td>
                <td className="text-muted small">{formatShortDate(exp.started_at)}</td>
                <td className="text-muted small">
                  {formatDuration(exp.started_at, exp.ended_at)}
                </td>
                <td className="text-muted small">{formatCO2(exp.carbon?.co2kg)}</td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </div>
  );
};
