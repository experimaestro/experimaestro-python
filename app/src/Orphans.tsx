import React, { useEffect } from "react";
import { Button, Table } from "react-bootstrap";

import { useAppSelector } from "./store";
import { OrphanJob } from "./reducers";
import client from "./client";
import { StatusBadge } from "./StatusBadge";

function formatBytes(n: number): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

const Orphans: React.FC = () => {
  const orphans = useAppSelector((s) => s.db.orphans);

  useEffect(() => {
    client.request_orphans();
  }, []);

  const strays = orphans.filter((j) => j.isStray);

  const killAll = () => {
    strays.forEach((j) => client.orphan_kill(j.jobId, j.taskId));
  };
  const deleteAll = () => {
    orphans
      .filter((j) => !j.isStray)
      .forEach((j) => client.orphan_delete(j.jobId, j.taskId));
  };

  return (
    <div>
      <div className="d-flex align-items-center mb-2" style={{ gap: "0.5rem" }}>
        <h5 className="mb-0">Orphan &amp; stray jobs</h5>
        <Button size="sm" variant="outline-secondary" onClick={() => client.request_orphans()}>
          <i className="fas fa-rotate-right" /> Refresh
        </Button>
        {strays.length > 0 && (
          <Button size="sm" variant="outline-danger" onClick={killAll}>
            Kill all running ({strays.length})
          </Button>
        )}
        <Button size="sm" variant="outline-danger" onClick={deleteAll}>
          Delete all finished
        </Button>
      </div>

      {orphans.length === 0 ? (
        <p className="text-muted">No orphan or stray jobs found.</p>
      ) : (
        <Table size="sm" hover>
          <thead>
            <tr>
              <th>Status</th>
              <th>Task</th>
              <th>Job</th>
              <th>Size</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {orphans.map((job: OrphanJob) => (
              <tr key={job.jobId}>
                <td>
                  <StatusBadge status={job.status} />
                </td>
                <td>{job.taskId}</td>
                <td>
                  <code>{job.jobId.substring(0, 12)}</code>
                </td>
                <td>{formatBytes(job.sizeBytes)}</td>
                <td>
                  {job.isStray && (
                    <Button
                      size="sm"
                      variant="outline-danger"
                      className="me-1"
                      onClick={() => client.orphan_kill(job.jobId, job.taskId)}
                      title="Kill"
                    >
                      <i className="fa fa-skull-crossbones" />
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline-secondary"
                    onClick={() => client.orphan_delete(job.jobId, job.taskId)}
                    title="Delete"
                  >
                    <i className="fa fa-trash" />
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </div>
  );
};

export default Orphans;
