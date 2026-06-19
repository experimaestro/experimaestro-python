import React from "react";
import { Badge, ProgressBar } from "react-bootstrap";

// Map a job status to a Bootstrap badge colour.
const COLOR: { [key: string]: string } = {
  running: "primary",
  done: "success",
  error: "danger",
  waiting: "warning",
  ready: "info",
  scheduled: "info",
  unscheduled: "secondary",
  transient: "secondary",
};

const DARK_TEXT = new Set(["warning", "info", "light"]);

export const StatusBadge = ({
  status,
  progress,
}: {
  status: string;
  progress?: number;
}) => {
  if (status === "running") {
    const pct = Math.round((progress ?? 0) * 100);
    return (
      <div className="status-badge-running">
        <Badge bg="primary">running {pct}%</Badge>
        <ProgressBar
          now={pct}
          variant="primary"
          className="status-badge-progress"
        />
      </div>
    );
  }
  const bg = COLOR[status] || "secondary";
  return (
    <Badge bg={bg} text={DARK_TEXT.has(bg) ? "dark" : undefined}>
      {status}
    </Badge>
  );
};

export default StatusBadge;
