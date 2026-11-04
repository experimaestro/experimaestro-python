import React from "react";
import { Job } from "./reducers";
import { formatShortDate, formatDuration, formatCO2 } from "./format";
import { StatusBadge } from "./StatusBadge";
import { TagBadges } from "./ui/tags";

/** Number of columns of the job table */
export const JOB_COLUMNS = 3;

export type JobAction = "details" | "logs" | "copy-id" | "copy-folder" | "kill" | "delete";

/** Actions on a job: icon, label and key (the TUI keys, without Ctrl) */
export const JOB_ACTIONS: {
  [key in JobAction]: { icon: string; label: string; key: string };
} = {
  details: { icon: "fas fa-eye", label: "Details", key: "i" },
  logs: { icon: "fas fa-file-lines", label: "Logs", key: "l" },
  "copy-id": { icon: "fas fa-clipboard", label: "Copy the job id", key: "c" },
  "copy-folder": { icon: "fas fa-folder", label: "Copy the job folder", key: "f" },
  kill: { icon: "fa fa-skull-crossbones", label: "Kill", key: "k" },
  delete: { icon: "fa fa-trash", label: "Delete", key: "d" },
};

/** The actions available for a job */
export function jobActions(job: Job): JobAction[] {
  const actions: JobAction[] = ["details"];
  if (job.started !== false) actions.push("logs");
  actions.push("copy-id", "copy-folder");
  if (job.status === "running") actions.push("kill");
  actions.push("delete");
  return actions;
}

export default ({
  job,
  onAction,
  onHover,
  hiddenTags,
}: {
  job: Job;
  onAction: (action: JobAction) => void;
  // Called with true/false when the pointer enters/leaves the job
  onHover: (hovered: boolean) => void;
  // Keys of the tags not shown
  hiddenTags: Set<string>;
}) => {
  const tags = job.tags.filter(([key]) => !hiddenTags.has(key));
  const progress = job.progress.length > 0 ? job.progress[0].progress : 0;
  return (
    <>
      <tr
        className="job-row"
        data-job-id={job.jobId}
        onMouseEnter={() => onHover(true)}
        onMouseLeave={() => onHover(false)}
      >
        <td className="text-nowrap">
          <span className="task-id">{job.taskId}</span>
          <span className="job-id ms-2" title={`${job.taskId}/${job.jobId}`}>
            {job.jobId.substring(0, 12)}
          </span>
        </td>

        {/* Cells spanning the two lines of the job */}
        <td rowSpan={2} className="job-status">
          <StatusBadge status={job.status} progress={progress} compact failure={job} />
        </td>
        <td className="job-date text-muted small text-nowrap">
          {formatShortDate(job.submitted)}
          {/* Shown over the end of the row when the job is hovered */}
          <div className="job-actions">
            {jobActions(job).map((action) => {
              const { icon, label, key } = JOB_ACTIONS[action];
              return (
                <i
                  key={action}
                  className={`${icon} action`}
                  data-action={action}
                  title={`${label} (${key})`}
                  aria-label={label}
                  onClick={() => onAction(action)}
                />
              );
            })}
          </div>
        </td>
      </tr>
      {/* Always present (even empty), so that all the jobs take the same space */}
      <tr
        className="job-tags-row"
        data-tags-of={job.jobId}
        onMouseEnter={() => onHover(true)}
        onMouseLeave={() => onHover(false)}
      >
        <td>
          {/* One line, scrolling when the tags do not fit */}
          <div className="job-tags-line">{tags.length > 0 && <TagBadges tags={tags} />}</div>
        </td>
        <td className="text-muted small text-nowrap">
          {formatDuration(job.start, job.end)}
          {(job.carbon?.co2kg ?? 0) > 0 && (
            <span className="job-co2" title="CO₂ emissions">
              {" · "}
              {formatCO2(job.carbon?.co2kg)}
            </span>
          )}
        </td>
      </tr>
    </>
  );
};
