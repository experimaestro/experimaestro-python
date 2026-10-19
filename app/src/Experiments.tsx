import React, { useState } from "react";
import { Table, Badge, Modal, Button } from "react-bootstrap";

import { useAppSelector } from "./store";
import { Experiment } from "./reducers";
import { formatShortDate, formatDuration, formatCO2 } from "./format";
import client from "./client";
import { useMessages } from "./ui/messages";

function statusInfo(exp: Experiment): { icon: string; color: string; label: string } {
  if (exp.failed_jobs > 0) return { icon: "❌", color: "danger", label: "failed" };
  if (exp.status === "running" || exp.finished_jobs < exp.total_jobs)
    return { icon: "🏃", color: "primary", label: "running" };
  return { icon: "✓", color: "success", label: "done" };
}

type Confirm = { experimentId: string; kind: "kill" | "delete" };

export default ({ onOpen }: { onOpen: (id: string) => void }) => {
  const experiments = useAppSelector((s) => s.db.experiments);
  const [confirm, setConfirm] = useState<Confirm>();
  const { info } = useMessages();

  if (experiments.ids.length === 0) {
    return <p className="text-muted mt-3">No experiments yet.</p>;
  }

  function cancelConfirm() {
    setConfirm(undefined);
    info("Action cancelled");
  }

  function runConfirm() {
    if (!confirm) return;
    if (confirm.kind === "kill") client.experiment_kill(confirm.experimentId);
    else client.experiment_delete(confirm.experimentId);
    setConfirm(undefined);
  }

  return (
    <div className="mt-3">
      <h5 className="mb-3">Experiments</h5>

      {confirm && (
        <Modal show={true} onHide={cancelConfirm}>
          <Modal.Header closeButton>
            <Modal.Title>Are you sure?</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {confirm.kind === "kill" ? (
              <>
                Kill all running jobs of experiment <b>{confirm.experimentId}</b>?
              </>
            ) : (
              <>
                Delete experiment <b>{confirm.experimentId}</b>?
                <div className="text-danger small mt-2">
                  This removes the experiment runs and events from disk. Job
                  directories are kept (they become orphans).
                </div>
              </>
            )}
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={cancelConfirm}>
              Cancel
            </Button>
            <Button
              variant={confirm.kind === "delete" ? "danger" : "primary"}
              onClick={runConfirm}
            >
              {confirm.kind === "kill" ? "Kill" : "Delete"}
            </Button>
          </Modal.Footer>
        </Modal>
      )}

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
            <th></th>
          </tr>
        </thead>
        <tbody>
          {experiments.ids.map((id) => {
            const exp = experiments.byId[id];
            const st = statusInfo(exp);
            const isRunning = exp.status === "running";
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
                <td
                  className="experiment-actions text-nowrap"
                  onClick={(e) => e.stopPropagation()}
                  title=""
                >
                  <i
                    className="fa fa-skull-crossbones action"
                    title="Kill all running jobs"
                    onClick={() => setConfirm({ experimentId: id, kind: "kill" })}
                  />
                  <i
                    className={`fa fa-trash action${isRunning ? " disabled" : ""}`}
                    title={
                      isRunning
                        ? "Cannot delete: experiment is still running"
                        : "Delete experiment"
                    }
                    onClick={() => {
                      if (!isRunning) setConfirm({ experimentId: id, kind: "delete" });
                    }}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </div>
  );
};
