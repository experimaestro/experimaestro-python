import React from "react";
import { copyToClibpoard } from "./clipboard";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";
import { formatShortDate, formatDuration, formatCO2 } from "./format";
import { StatusBadge } from "./StatusBadge";

export default ({
  job,
  onKill,
  onDelete,
  onShow,
  onLogs,
}: {
  job: Job;
  onKill: () => void;
  onDelete: () => void;
  onShow: () => void;
  onLogs: () => void;
}) => {
  const { success, error } = useMessages();
  const progress = job.progress.length > 0 ? job.progress[0].progress : 0;

  return (
    <tr className="job-row">
      <td className="job-status-cell">
        <StatusBadge status={job.status} progress={progress} />
      </td>

      <td>
        <span
          className="clipboard task-id"
          onClick={() =>
            copyToClibpoard(job.taskId)
              .then(() => success("Task id copied"))
              .catch((e) => error("Error when copying: " + e))
          }
        >
          {job.taskId}
        </span>
        <div>
          <span
            className="clipboard job-id"
            title={job.locator}
            onClick={() =>
              copyToClibpoard(job.locator)
                .then(() => success("Job path copied"))
                .catch((e) => error("Error when copying job path: " + e))
            }
          >
            {job.jobId.substring(0, 12)}
          </span>
        </div>
      </td>

      <td>
        {job.tags.map((tag) => (
          <span key={tag[0]} className="tag">
            <span className="name">{tag[0]}</span>
            <span className="value">{tag[1]}</span>
          </span>
        ))}
      </td>

      <td className="text-muted small text-nowrap">{formatShortDate(job.submitted)}</td>
      <td className="text-muted small text-nowrap">
        {formatDuration(job.start, job.end)}
      </td>
      <td className="text-muted small text-nowrap">{formatCO2(job.carbon?.co2kg)}</td>

      <td className="job-actions text-nowrap">
        <i className="fas fa-eye action" title="Details" onClick={onShow} />
        <i className="fas fa-file-lines action" title="Logs" onClick={onLogs} />
        {job.status === "running" && (
          <i
            className="fa fa-skull-crossbones action"
            title="Kill"
            onClick={onKill}
          />
        )}
        <i className="fa fa-trash action" title="Delete" onClick={onDelete} />
      </td>
    </tr>
  );
};
