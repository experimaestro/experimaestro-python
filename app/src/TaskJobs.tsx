import React from "react";
import { copyToClibpoard } from "./clipboard";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";
import { useAppSelector } from "./store";

export default ({
  job,
  onKill,
  onShow,
}: {
  job: Job;
  onKill: () => void;
  onShow: () => void;
}) => {
  const { success, error } = useMessages();
  const progress = job.progress.length > 0 ? job.progress[0].progress : 0;
  const allJobs = useAppSelector((state) => state.db.jobs);

  // Get dependency job info (taskId and status)
  const dependencies = (job.dependsOn || []).map((depId) => {
    const depJob = allJobs.byId[depId];
    return {
      jobId: depId,
      taskId: depJob?.taskId || depId.substring(0, 8),
      status: depJob?.status || "unknown",
    };
  });

  return (
    <div className="resource">
      {job.status === "running" ? (
        <>
          <span
            className="status progressbar-container"
            title={`${progress * 100}%`}
          >
            <span
              style={{ right: `${(1 - progress) * 100}%` }}
              className="progressbar"
            ></span>
            <div className="status-running">{job.status}</div>
          </span>
          <i className="fa fa-skull-crossbones action" onClick={onKill} />
        </>
      ) : (
        <span className={`status status-${job.status}`}>{job.status}</span>
      )}

      <i className="fas fa-eye action" title="Details" onClick={onShow} />
      <span className="job-id" />
      <span
        className="clipboard"
        onClick={(event) =>
          copyToClibpoard(job.locator)
            .then(() => success("Job path copied"))
            .catch((e) => error("Error when copying job path: " + e))
        }
      >
        {job.taskId}
      </span>
      {job.tags.map((tag) => (
        <span key={tag[0]} className="tag">
          <span className="name">{tag[0]}</span>
          <span className="value">{tag[1]}</span>
        </span>
      ))}
      {dependencies.length > 0 && (
        <span className="dependencies">
          <span className="dependencies-label">deps:</span>
          {dependencies.map((dep) => (
            <span
              key={dep.jobId}
              className={`dependency status-${dep.status}`}
              title={`${dep.taskId} (${dep.status})`}
            >
              {dep.taskId.split(".").pop()}
            </span>
          ))}
        </span>
      )}
    </div>
  );
};
