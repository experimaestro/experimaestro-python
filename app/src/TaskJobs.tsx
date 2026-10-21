import React from "react";
import { copyToClibpoard } from "./clipboard";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";
import { formatShortDate, formatDuration, formatCO2 } from "./format";
import { StatusBadge } from "./StatusBadge";
import { TagBadges } from "./ui/tags";

/** Number of columns of the job table */
export const JOB_COLUMNS = 5;

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
    <>
      <tr className="job-row" data-job-id={job.jobId}>
        <td className="text-nowrap">
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
          <span
            className="clipboard job-id ms-2"
            title={job.locator}
            onClick={() =>
              copyToClibpoard(job.locator)
                .then(() => success("Job path copied"))
                .catch((e) => error("Error when copying job path: " + e))
            }
          >
            {job.jobId.substring(0, 12)}
          </span>
        </td>

        {/* Cells spanning the two lines of the job */}
        <td rowSpan={2} className="job-status">
          <StatusBadge status={job.status} progress={progress} compact />
        </td>
        <td className="text-muted small text-nowrap">{formatShortDate(job.submitted)}</td>
        <td rowSpan={2} className="text-muted small text-nowrap">
          {formatCO2(job.carbon?.co2kg)}
        </td>
        <td rowSpan={2} className="job-actions text-nowrap">
          <i className="fas fa-eye action" title="Details" onClick={onShow} />
          <i className="fas fa-file-lines action" title="Logs" onClick={onLogs} />
          {job.status === "running" && (
            <i className="fa fa-skull-crossbones action" title="Kill" onClick={onKill} />
          )}
          <i className="fa fa-trash action" title="Delete" onClick={onDelete} />
        </td>
      </tr>
      {/* Always present (even empty), so that all the jobs take the same space */}
      <tr className="job-tags-row" data-tags-of={job.jobId}>
        <td>
          {/* One line, scrolling when the tags do not fit */}
          <div className="job-tags-line">
            {job.tags.length > 0 && <TagBadges tags={job.tags} />}
          </div>
        </td>
        <td className="text-muted small text-nowrap">
          {formatDuration(job.start, job.end)}
        </td>
      </tr>
    </>
  );
};
