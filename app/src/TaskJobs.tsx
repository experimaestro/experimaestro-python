import React from "react";
import { copyToClibpoard } from "./clipboard";
import { Job } from "./reducers";
import { useMessages } from "./ui/messages";

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
    </div>
  );
};
