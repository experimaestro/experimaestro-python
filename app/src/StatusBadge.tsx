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

// The failure of an error state (from the server, shared with the TUI)
export type Failure = {
  failureReason?: string | null;
  failureIcon?: string | null;
  failureDescription?: string | null;
};

export const StatusBadge = ({
  status,
  progress,
  compact = false,
  failure,
}: {
  status: string;
  progress?: number;
  // A single small badge (without the progress bar)
  compact?: boolean;
  // Failure reason: its icon is shown in the badge, its description on hover
  failure?: Failure;
}) => {
  if (status === "running" && compact) {
    const pct = Math.round((progress ?? 0) * 100);
    return <Badge bg="primary">running {pct}%</Badge>;
  }
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
  const reason = failure?.failureReason;
  const badge = (
    <Badge bg={bg} text={DARK_TEXT.has(bg) ? "dark" : undefined}>
      {status}
    </Badge>
  );
  if (!reason) return badge;
  // The failure icon after the badge (some icons, as ❌, are red like the badge)
  return (
    <span
      className="status-badge-failure"
      data-failure={reason}
      title={`${status}: ${failure?.failureDescription || reason.toLowerCase()}`}
    >
      {badge}
      {failure?.failureIcon && (
        <span className="status-badge-icon">{failure.failureIcon}</span>
      )}
    </span>
  );
};

export default StatusBadge;
